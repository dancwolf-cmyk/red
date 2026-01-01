# -*- coding: utf-8 -*-
"""
demo_router_qwen_soft_v4.py
---------------------------
- 使用 Qwen3-0.6B 作为 encoder （温和微调：仅解冻最后 1–2 层）
- 加法任务范围：1–50 => 和 s=2..100（永不超范围）
- Residual experts + soft router
- 全阶段加入 tqdm 进度条，不爆显存版本
"""

import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 数据集，范围改为 1–50
# ============================================================

def build_addition_dataset(num_samples=20000, seed=42):
    """
    自然语言加法：
      Q: "What is 13 plus 7?"
      A: "20"

    新设定：
      - a,b ∈ [1,50]
      - s = a+b ∈ [2,100]（不会超范围）
    """

    random.seed(seed)
    samples = []

    def sum_to_domain(s):
        if 2 <= s <= 20:   return "range_2_20"
        if 21 <= s <= 40:  return "range_21_40"
        if 41 <= s <= 60:  return "range_41_60"
        if 61 <= s <= 80:  return "range_61_80"
        if 81 <= s <= 100: return "range_81_100"
        return None

    for _ in range(num_samples):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        s = a + b                        # 保证 2–100
        domain = sum_to_domain(s)

        q = f"What is {a} plus {b}?"
        samples.append({"question": q, "answer": str(s), "domain": domain})

    # 标签：2..100 共 99 类
    label_vocab = [str(i) for i in range(2, 101)]
    return samples, label_vocab


# ============================================================
# 2. Dataset
# ============================================================

class QADataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, label_vocab, max_len=32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_id = {a: i for i, a in enumerate(label_vocab)}
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = self.label_to_id[s["answer"]]

        enc = self.tokenizer(
            s["question"],
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
            "domain": s["domain"],
        }


# ============================================================
# 3. Qwen encoder（温和微调）
# ============================================================

class QwenEncoderFrozen(nn.Module):
    def __init__(self, model_path, max_len=64, trainable_last_n_layers=1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        # 全部冻结
        for p in self.model.parameters():
            p.requires_grad = False

        # 解冻最后 n 层
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

        # 最终 norm 一般也可适当解冻
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            for p in self.model.model.norm.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        h = out.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1)
        h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return h  # (B,H)


# ============================================================
# 4. Residual Experts + Router
# ============================================================

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


# ============================================================
# 5. 训练（带 tqdm）
# ============================================================

def train_base(model, loader, epochs=6, lr=5e-5):
    """训练 Qwen 的最后几层 + base_head"""
    for exp in model.experts:
        for p in exp.parameters(): p.requires_grad = False
    for p in model.router.parameters(): p.requires_grad = False
    for p in model.base_head.parameters(): p.requires_grad = True

    params = list(model.base_head.parameters()) + \
             [p for p in model.encoder.parameters() if p.requires_grad]

    optim = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        total_loss = total_acc = 0
        pbar = tqdm(loader, desc=f"[Base] epoch {ep}/{epochs}", ncols=120)

        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            logits = model(ids, mask, mode="base")
            loss = ce(logits, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pred = logits.argmax(-1)
            acc = (pred == label).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=loss.item(), acc=acc)

        print(f"=> Base epoch {ep} avg_loss={total_loss/len(loader):.4f} avg_acc={total_acc/len(loader):.4f}")


def train_expert(model, loader, ei, epochs=5, lr=5e-4):
    """训练单个 expert（encoder 完全冻结）"""
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.base_head.parameters(): p.requires_grad = False
    for i, exp in enumerate(model.experts):
        for p in exp.parameters(): p.requires_grad = (i == ei)
    for p in model.router.parameters(): p.requires_grad = False

    optim = torch.optim.Adam(model.experts[ei].parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        total_loss = total_acc = 0
        pbar = tqdm(loader, desc=f"[Expert {ei}] epoch {ep}/{epochs}", ncols=120)

        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            mask_e = torch.zeros(model.num_experts, device=DEVICE)
            mask_e[ei] = 1

            logits = model(ids, mask, mode="all_fixed", expert_mask=mask_e)
            loss = ce(logits, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pred = logits.argmax(-1)
            acc = (pred == label).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=loss.item(), acc=acc)

        print(f"=> Expert {ei} epoch {ep} avg_loss={total_loss/len(loader):.4f} avg_acc={total_acc/len(loader):.4f}")


def train_router(model, loader, domain_to_id, epochs=8, lr=5e-4):
    """训练 router（encoder 和 experts 冻结）"""
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.base_head.parameters(): p.requires_grad = False
    for exp in model.experts:
        for p in exp.parameters(): p.requires_grad = False
    for p in model.router.parameters(): p.requires_grad = True

    optim = torch.optim.Adam(model.router.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        total_loss = total_acc = 0
        pbar = tqdm(loader, desc=f"[Router] epoch {ep}/{epochs}", ncols=120)

        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            domains = batch["domain"]
            d_ids = torch.tensor([domain_to_id[d] for d in domains], device=DEVICE)

            h = model.encoder(ids, mask)
            logits = model.router(h)

            loss = ce(logits, d_ids)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pred = logits.argmax(-1)
            acc = (pred == d_ids).float().mean().item()

            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=loss.item(), acc=acc)

        print(f"=> Router epoch {ep} avg_loss={total_loss/len(loader):.4f} avg_acc={total_acc/len(loader):.4f}")


# ============================================================
# 6. 评估
# ============================================================

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


# ============================================================
# 7. 主流程
# ============================================================

def main():
    samples, label_vocab = build_addition_dataset(num_samples=20000)
    random.shuffle(samples)

    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    test_samples = samples[split:]

    TOKENIZER_PATH = r"E:\dev\test2\Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    train_ds = QADataset(train_samples, tokenizer, label_vocab)
    test_ds  = QADataset(test_samples, tokenizer, label_vocab)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

    # domain -> expert
    domain_to_expert = {
        "range_2_20": 0, "range_21_40": 1, "range_41_60": 2,
        "range_61_80": 3, "range_81_100": 4,
    }

    # per-range loader
    expert_train_loaders = [[] for _ in range(5)]
    expert_test_loaders = [[] for _ in range(5)]
    for s in train_samples: expert_train_loaders[domain_to_expert[s["domain"]]].append(s)
    for s in test_samples:  expert_test_loaders[domain_to_expert[s["domain"]]].append(s)

    expert_train_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab), batch_size=16, shuffle=True)
        for sub in expert_train_loaders
    ]
    expert_test_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab), batch_size=16, shuffle=False)
        for sub in expert_test_loaders
    ]

    encoder = QwenEncoderFrozen(TOKENIZER_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = ModularAddModelWithRouter(encoder, num_classes=len(label_vocab)).to(DEVICE)

    # Stage 1
    train_base(model, train_loader, epochs=6)
    evaluate(model, train_loader, "base", "Base-only Train")
    evaluate(model, test_loader,  "base", "Base-only Test")

    # Stage 2
    for ei in range(5):
        train_expert(model, expert_train_loaders[ei], ei, epochs=8)

    evaluate(model, train_loader, "all_fixed", "AllExperts Train")
    evaluate(model, test_loader,  "all_fixed", "AllExperts Test")

    # Stage 3
    train_router(model, train_loader, domain_to_expert, epochs=6)

    evaluate(model, train_loader, "router", "Router Train")
    evaluate(model, test_loader,  "router", "Router Test")


if __name__ == "__main__":
    main()
