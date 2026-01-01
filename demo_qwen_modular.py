# -*- coding: utf-8 -*-
"""
demo_qwen_modular.py
--------------------
Use Qwen3-0.6B as a frozen feature extractor and train only five domain-specific
expert heads on the addition QA task. The base model and a residual bias head
stay untouched, while each expert is trained sequentially on its own range.
"""

import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_addition_dataset(num_samples=20000, seed=42):
    """Build natural-language addition pairs split into five ranges."""
    random.seed(seed)
    samples = []

    def sum_to_domain(s: int):
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

    while len(samples) < num_samples:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        s = a + b
        if s < 2 or s > 100:
            continue
        domain = sum_to_domain(s)
        if domain is None:
            continue

        question = f"What is {a} plus {b}?"
        answer = str(s)
        samples.append({"question": question, "answer": answer, "domain": domain})

    label_vocab = [str(i) for i in range(2, 201)]
    return samples, label_vocab


class QADataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, label_vocab, max_len=32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.label_to_id = {entry: idx for idx, entry in enumerate(label_vocab)}
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer(
            sample["question"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        label = self.label_to_id[sample["answer"]]
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "domain": sample["domain"],
        }


class FrozenQwenEncoder(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            pretrained_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths


class QwenModularExperts(nn.Module):
    def __init__(self, pretrained_path: str, num_classes: int, num_experts=5, alpha=0.1, router_scale=0.1):
        super().__init__()
        self.encoder = FrozenQwenEncoder(pretrained_path)
        hidden_size = self.encoder.base.config.hidden_size
        self.base_bias = nn.Linear(hidden_size, num_classes)
        nn.init.zeros_(self.base_bias.weight)
        nn.init.zeros_(self.base_bias.bias)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_experts)])
        self.alpha = alpha
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        self.router_scale = nn.Parameter(torch.tensor(router_scale))

    def forward(self, input_ids, attention_mask, mode="base", expert_mask=None):
        h = self.encoder(input_ids, attention_mask)
        logits = self.base_bias(h)
        if mode == "base":
            return logits

        if mode == "experts":
            if expert_mask is None:
                return logits
            scaled = torch.zeros_like(logits)
            active = expert_mask.to(logits.dtype)
            for idx, expert in enumerate(self.experts):
                if active[idx] == 1:
                    scaled = scaled + self.alpha * expert(h)
            return logits + scaled

        if mode == "router":
            gate_logits = self.router(h)
            gate = torch.softmax(gate_logits, dim=-1)
            expert_logits = torch.stack([expert(h) for expert in self.experts], dim=1)
            weighted = (expert_logits * gate.unsqueeze(-1)).sum(dim=1)
            return logits + self.router_scale * weighted

        raise ValueError(f"Unknown mode: {mode}")


def train_expert(model, dataloader, expert_idx, epochs=8, lr=5e-4):
    model.train()
    for p in model.experts.parameters():
        p.requires_grad = False
    for idx, expert in enumerate(model.experts):
        for p in expert.parameters():
            p.requires_grad = (idx == expert_idx)
    for p in model.base_bias.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(model.experts[expert_idx].parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            mask = torch.zeros(model.num_experts, dtype=torch.long, device=DEVICE)
            mask[expert_idx] = 1
            logits = model(input_ids, attention_mask, mode="experts", expert_mask=mask)

            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        print(f"[Expert {expert_idx}] epoch {ep:3d}, loss={total_loss / total_count:.4f}, "
              f"acc={total_correct / total_count:.4f}")


@torch.no_grad()
def evaluate(model, dataloader, mode="base", expert_mask=None, desc=""):
    model.eval()
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask, mode=mode, expert_mask=expert_mask)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    acc = total_correct / total_count
    print(f"[Eval] {desc} accuracy: {acc:.4f}")
    return acc


@torch.no_grad()
def evaluate_per_range(model, loaders, descs):
    print("\n=== Per-range evaluation ===")
    for mask_idx, loader in enumerate(loaders):
        mask = torch.zeros(model.num_experts, dtype=torch.long, device=DEVICE)
        mask[mask_idx] = 1
        acc = evaluate(model, loader, mode="experts", expert_mask=mask,
                       desc=f"Expert {mask_idx} ({descs[mask_idx]})")
        print(f"Range {descs[mask_idx]}: expert {mask_idx} acc={acc:.4f}")


def train_router(model, dataloader, domain_to_id, epochs=6, lr=5e-4):
    model.train()
    for p in model.base_bias.parameters():
        p.requires_grad = False
    for expert in model.experts:
        for p in expert.parameters():
            p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = True

    optim = torch.optim.Adam(model.router.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            domains = batch["domain"]
            domain_ids = torch.tensor(
                [domain_to_id[d] for d in domains],
                dtype=torch.long,
                device=DEVICE,
            )

            h = model.encoder(input_ids, attention_mask)
            gate_logits = model.router(h)
            loss = criterion(gate_logits, domain_ids)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * domain_ids.size(0)
            preds = gate_logits.argmax(dim=-1)
            total_correct += (preds == domain_ids).sum().item()
            total_count += domain_ids.size(0)

        print(f"[Router] epoch {ep:3d}, loss={total_loss / total_count:.4f}, "
              f"acc={total_correct / total_count:.4f}")


def main():
    samples, label_vocab = build_addition_dataset(num_samples=20000, seed=42)
    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    test_samples = samples[split:]

    tokenizer_path = r"E:\dev\test2\Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    max_len = 32
    train_loader = DataLoader(
        QADataset(train_samples, tokenizer, label_vocab, max_len=max_len),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        QADataset(test_samples, tokenizer, label_vocab, max_len=max_len),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    domain_to_idx = {
        "range_2_20": 0,
        "range_21_40": 1,
        "range_41_60": 2,
        "range_61_80": 3,
        "range_81_100": 4,
    }
    expert_train_samples = [[] for _ in range(5)]
    expert_test_samples = [[] for _ in range(5)]
    for sample in train_samples:
        expert_train_samples[domain_to_idx[sample["domain"]]].append(sample)
    for sample in test_samples:
        expert_test_samples[domain_to_idx[sample["domain"]]].append(sample)

    expert_train_loaders = [
        DataLoader(
            QADataset(subset, tokenizer, label_vocab, max_len=max_len),
            batch_size=32,
            shuffle=True,
        )
        for subset in expert_train_samples
    ]
    expert_test_loaders = [
        DataLoader(
            QADataset(subset, tokenizer, label_vocab, max_len=max_len),
            batch_size=64,
            shuffle=False,
        )
        for subset in expert_test_samples
    ]

    model = QwenModularExperts(
        pretrained_path=tokenizer_path,
        num_classes=len(label_vocab),
        num_experts=5,
        alpha=0.1,
    ).to(DEVICE)

    print("=== Training five domain experts (base frozen) ===")
    for expert_idx in range(5):
        if len(expert_train_loaders[expert_idx].dataset) == 0:
            print(f"Expert {expert_idx} has no samples, skipping.")
            continue
        train_expert(model, expert_train_loaders[expert_idx], expert_idx, epochs=8, lr=5e-4)

    mask_all = torch.ones(model.num_experts, dtype=torch.long, device=DEVICE)
    print("=== Evaluate Base+Experts on TRAIN ===")
    evaluate(model, train_loader, mode="experts", expert_mask=mask_all, desc="Base+AllExperts-Train")
    print("=== Evaluate Base+Experts on TEST ===")
    evaluate(model, test_loader, mode="experts", expert_mask=mask_all, desc="Base+AllExperts-Test")

    print("=== Train Router on domain labels ===")
    train_router(model, train_loader, domain_to_idx, epochs=6, lr=5e-4)

    print("=== Evaluate Router on TRAIN ===")
    evaluate(model, train_loader, mode="router", desc="Router-Train")
    print("=== Evaluate Router on TEST ===")
    evaluate(model, test_loader, mode="router", desc="Router-Test")

    evaluate_per_range(
        model,
        expert_test_loaders,
        descs=("2-20", "21-40", "41-60", "61-80", "81-100"),
    )


if __name__ == "__main__":
    main()
