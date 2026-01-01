# -*- coding: utf-8 -*-
"""
demo_router_qwen_soft_v7.py
---------------------------
- 基于 demo_v6，增加虚拟 batch (梯度累积)
- 其余结构基本不变，进度条/soft router 保留
"""

import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_accum_steps(loader: DataLoader, virtual_batch_size: int):
    unit = getattr(loader, "batch_size", 1) or 1
    steps = max(1, virtual_batch_size // unit)
    return steps


# ============================================================
# 1. 数据集等内容同 demo_v6
# ============================================================

def build_addition_dataset(num_samples=20000, seed=42):
    random.seed(seed)
    samples = []

    def sum_to_domain(s):
        if 2 <= s <= 20:
            return "range_2_20"
        if 21 <= s <= 40:
            return "range_21_40"
        if 41 <= s <= 60:
            return "range_41_60"
        if 61 <= s <= 80:
            return "range_61_80"
        if 81 <= s <= 100:
            return "range_81_100"
        return None

    for _ in range(num_samples):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        s = a + b
        domain = sum_to_domain(s)
        q = f"What is {a} plus {b}?"
        samples.append({"question": q, "answer": str(s), "domain": domain})

    label_vocab = [str(i) for i in range(2, 101)]
    return samples, label_vocab


class QADataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, label_vocab, max_len=32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_id = {a: i for i, a in enumerate(label_vocab)}
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self.label_to_id[s["answer"]]

        enc = self.tokenizer(
            s["question"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
            "domain": s["domain"],
        }


class QwenEncoderFrozen(nn.Module):
    def __init__(self, model_path, max_len=64, trainable_last_n_layers=1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        for p in self.model.parameters():
            p.requires_grad = False
        self.unfreeze_last_layers(trainable_last_n_layers)

    def unfreeze_last_layers(self, n):
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        if layers:
            for block in layers[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            for p in self.model.model.norm.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class ModularAddModelWithRouter(nn.Module):
    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size

        self.base_head = nn.Linear(H, num_classes)
        self.experts = nn.ModuleList([nn.Linear(H, num_classes) for _ in range(num_experts)])
        self.router = nn.Linear(H, num_experts)
        self.num_experts = num_experts
        self.router_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, attention_mask, mode="base", expert_mask=None):
        h = self.encoder(input_ids, attention_mask)
        base = self.base_head(h)
        if mode == "base":
            return base

        exp_logits = torch.stack([exp(h) for exp in self.experts], dim=1)
        if mode == "all_fixed":
            mask = torch.ones(self.num_experts, device=h.device) if expert_mask is None else expert_mask
            weighted = (exp_logits * mask.view(1, -1, 1)).sum(dim=1)
            return base + 0.1 * weighted

        if mode == "router":
            α = torch.softmax(self.router(h), dim=-1).unsqueeze(-1)
            weighted = (exp_logits * α).sum(dim=1)
            return base + self.router_scale * weighted

        raise ValueError("Unknown mode")


def _train_loop(model, loader, optim, criterion, train_step_fn, desc, accum_steps):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    iters = 0
    grad_step = 0

    pbar = tqdm(loader, desc=desc, ncols=120)
    optim.zero_grad()
    for batch in pbar:
        logits, label = train_step_fn(batch)
        loss = criterion(logits, label) / accum_steps

        loss.backward()
        grad_step += 1

        if grad_step % accum_steps == 0:
            optim.step()
            optim.zero_grad()

        pred = logits.argmax(-1)
        acc = (pred == label).float().mean().item()

        total_loss += loss.item() * accum_steps
        total_acc += acc
        iters += 1
        pbar.set_postfix(loss="{:.4f}".format(total_loss / iters), acc="{:.4f}".format(total_acc / iters))

    if grad_step % accum_steps != 0:
        optim.step()
        optim.zero_grad()

    return total_loss / iters, total_acc / iters


def train_base(model, loader, epochs=6, lr=5e-5, virtual_batch_size=64):
    for exp in model.experts:
        for p in exp.parameters():
            p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = True

    params = list(model.base_head.parameters()) + [p for p in model.encoder.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def base_step(batch):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        logits = model(ids, mask, mode="base")
        return logits, label

    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=base_step,
            desc=f"[Base] epoch {ep}/{epochs}", accum_steps=accum_steps)
        print(f"=> Base epoch {ep} avg_loss={loss:.4f} avg_acc={acc:.4f}")


def train_expert(model, loader, ei, epochs=5, lr=5e-4, virtual_batch_size=64):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = False
    for i, exp in enumerate(model.experts):
        for p in exp.parameters():
            p.requires_grad = (i == ei)
    for p in model.router.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(model.experts[ei].parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def expert_step(batch):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        expert_mask = torch.zeros(model.num_experts, device=DEVICE)
        expert_mask[ei] = 1.0
        logits = model(ids, mask, mode="all_fixed", expert_mask=expert_mask)
        return logits, label

    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            forward_fn,
            desc=f"[Expert {ei}] epoch {ep}/{epochs}", accum_steps=accum_steps)
        print(f"=> Expert {ei} epoch {ep} avg_loss={loss:.4f} avg_acc={acc:.4f}")


def train_router(model, loader, domain_to_id, epochs=8, lr=5e-4, virtual_batch_size=64):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = False
    for exp in model.experts:
        for p in exp.parameters():
            p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = True

    optim = torch.optim.Adam(model.router.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def router_step(batch):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        domains = batch["domain"]
        domain_ids = torch.tensor([domain_to_id[d] for d in domains], device=DEVICE)
        logits = model(ids, mask, mode="router")
        return logits, domain_ids

    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=router_step,
            desc=f"[Router] epoch {ep}/{epochs}", accum_steps=accum_steps)
        print(f"=> Router epoch {ep} avg_loss={loss:.4f} avg_acc={acc:.4f}")


@torch.no_grad()
def evaluate(model, loader, mode, desc):
    model.eval()
    total = correct = 0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        logits = model(ids, mask, mode=mode)
        pred = logits.argmax(-1)

        correct += (pred == label).sum().item()
        total += label.size(0)

    acc = correct / total
    print(f"[Eval] {desc} acc={acc:.4f}")
    return acc


def main():
    samples, label_vocab = build_addition_dataset(num_samples=20000)
    random.shuffle(samples)

    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    test_samples = samples[split:]

    TOKENIZER_PATH = r"E:\dev\test2\Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    train_ds = QADataset(train_samples, tokenizer, label_vocab)
    test_ds = QADataset(test_samples, tokenizer, label_vocab)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    domain_to_expert = {
        "range_2_20": 0,
        "range_21_40": 1,
        "range_41_60": 2,
        "range_61_80": 3,
        "range_81_100": 4,
    }

    expert_train = [[] for _ in range(5)]
    expert_test = [[] for _ in range(5)]
    for s in train_samples:
        expert_train[domain_to_expert[s["domain"]]].append(s)
    for s in test_samples:
        expert_test[domain_to_expert[s["domain"]]].append(s)

    expert_train_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab), batch_size=16, shuffle=True)
        for sub in expert_train
    ]
    expert_test_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab), batch_size=16, shuffle=False)
        for sub in expert_test
    ]

    encoder = QwenEncoderFrozen(TOKENIZER_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = ModularAddModelWithRouter(encoder, num_classes=len(label_vocab)).to(DEVICE)

    train_base(model, train_loader, epochs=6, virtual_batch_size=64)
    evaluate(model, train_loader, "base", "Base-only Train")
    evaluate(model, test_loader, "base", "Base-only Test")

    for ei in range(5):
        train_expert(model, expert_train_loaders[ei], ei, epochs=8, virtual_batch_size=64)

    evaluate(model, train_loader, "all_fixed", "AllExperts Train")
    evaluate(model, test_loader, "all_fixed", "AllExperts Test")

    train_router(model, train_loader, domain_to_expert, epochs=6, virtual_batch_size=64)

    evaluate(model, train_loader, "router", "Router Train")
    evaluate(model, test_loader, "router", "Router Test")


if __name__ == "__main__":
    main()
