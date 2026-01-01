# -*- coding: utf-8 -*-
"""
modular_addition_transformer_v2.py
----------------------------------
大号 toy 实验：自然语言加法问答 (1–100)，
Transformer + Base + 5 Residual Experts

依赖:
  pip install torch transformers
"""

import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# 1. 构造加法 QA 数据集
# ========================

def build_addition_dataset(num_samples=20000, seed=42):
    """
    构造类似：
      Q: "What is 13 plus 7?"
      A: "20"
    的数据集。

    限制：
      1 <= a,b <= 100
      2 <= a+b <= 100
    并按 sum 分 5 个 domain：
      [2,20], [21,40], [41,60], [61,80], [81,100]
    """
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
        return None  # 超出我们关心的区间就丢弃

    # 生成随机 (a,b)，丢弃和不在 2–100 的
    while len(samples) < num_samples:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        s = a + b
        if s < 2 or s > 100:
            continue
        domain = sum_to_domain(s)
        if domain is None:
            continue

        q = "What is {} plus {}?".format(a, b)
        a_str = str(s)
        samples.append({"question": q, "answer": a_str, "domain": domain})

    # label_vocab 固定为 "2".."100"
    label_vocab = [str(i) for i in range(2, 101)]

    return samples, label_vocab


# ========================
# 2. Dataset & DataLoader
# ========================

class QADataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, label_vocab, max_len=32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.label_to_id = {a: i for i, a in enumerate(label_vocab)}
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        q = sample["question"]
        a = sample["answer"]
        if a not in self.label_to_id:
            raise ValueError("Unknown answer: {}".format(a))
        label = self.label_to_id[a]

        enc = self.tokenizer(
            q,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "domain": sample["domain"],
        }


# ========================
# 3. 模型定义
# ========================

class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, num_layers=4, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask):
        # 保证 long 型
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        B, L = input_ids.size()

        # 安全检查：token 索引是否越界
        max_id = int(input_ids.max().item())
        min_id = int(input_ids.min().item())
        if min_id < 0 or max_id >= self.token_emb.num_embeddings:
            raise ValueError(
                "input_ids out of range for token_emb: "
                "min_id={} max_id={} vocab_size={}".format(
                    min_id, max_id, self.token_emb.num_embeddings
                )
            )

        # 位置编码长度检查
        if L > self.pos_emb.num_embeddings:
            raise ValueError(
                "sequence length L={} exceeds pos_emb size {}".format(
                    L, self.pos_emb.num_embeddings
                )
            )

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)  # (B, L, d_model)

        src_key_padding_mask = attention_mask == 0  # (B, L)
        h = self.encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, L, d_model)

        mask = attention_mask.unsqueeze(-1)
        h_sum = (h * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        h_pooled = h_sum / lengths
        return h_pooled


class ModularAddModel(nn.Module):
    def __init__(self, vocab_size, num_classes, num_experts=5, d_model=256, n_heads=4, num_layers=4, max_len=64):
        super().__init__()
        self.encoder = TinyTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            max_len=max_len,
        )
        self.base_head = nn.Linear(d_model, num_classes)
        self.experts = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(num_experts)]
        )

    def forward(self, input_ids, attention_mask, expert_mask=None):
        h = self.encoder(input_ids, attention_mask)  # (B, d_model)
        

        logits = self.base_head(h)
        if expert_mask is not None:
            alpha = 0.1  # 残差缩放系数，控制专家的影响力
            for i, expert in enumerate(self.experts):
                if expert_mask[i] == 1:
                    logits = logits + alpha * expert(h)

        return logits


# ========================
# 4. 训练 & 评估
# ========================

def train_base(model, dataloader, epochs=10, lr=1e-4):
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    for p in model.experts.parameters():
        p.requires_grad = False

    optim = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.base_head.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, expert_mask=None)
            loss = criterion(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        print("[Base] epoch {:3d}, loss={:.4f}, acc={:.4f}".format(
            ep, total_loss / total_count, total_correct / total_count
        ))


def train_expert(model, dataloader, expert_idx, epochs=5, lr=5e-4):
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = False
    for i, expert in enumerate(model.experts):
        for p in expert.parameters():
            p.requires_grad = (i == expert_idx)

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

            mask = torch.zeros(len(model.experts), dtype=torch.long, device=DEVICE)
            mask[expert_idx] = 1

            logits = model(input_ids, attention_mask, expert_mask=mask)
            
            ce = criterion(logits, labels)

            # L2 penalty on current expert
            h = model.encoder(input_ids, attention_mask)
            logits_cur = model.experts[expert_idx](h)
            reg = (logits_cur ** 2).mean()

            loss = ce + 1e-4 * reg


            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

        print("[Expert {}] epoch {:3d}, loss={:.4f}, acc={:.4f}".format(
            expert_idx, ep, total_loss / total_count, total_correct / total_count
        ))


@torch.no_grad()
def evaluate(model, dataloader, expert_mask=None, desc=""):
    model.eval()
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(input_ids, attention_mask, expert_mask=expert_mask)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    acc = total_correct / total_count
    print("[Eval] {} accuracy: {:.4f}".format(desc, acc))
    return acc


# ========================
# 5. 主流程
# ========================

def main():
    # 1) 数据 & label
    samples, label_vocab = build_addition_dataset(num_samples=20000, seed=42)
    print("Total samples:", len(samples))
    print("Label vocab size:", len(label_vocab))

    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    test_samples = samples[split:]

    # 2) tokenizer
    TOKENIZER_PATH = r"C:\temp\lunwen\Qwen3-0.6B"  # TODO: 改成你本地路径
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    max_len = 32
    train_ds = QADataset(train_samples, tokenizer, label_vocab, max_len=max_len)
    test_ds = QADataset(test_samples, tokenizer, label_vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # domain -> expert idx 映射
    domain_to_expert = {
        "range_2_20": 0,
        "range_21_40": 1,
        "range_41_60": 2,
        "range_61_80": 3,
        "range_81_100": 4,
    }

    # 为每个 expert 构建自己的 train loader
    expert_samples = [[] for _ in range(5)]
    for s in train_samples:
        idx = domain_to_expert[s["domain"]]
        expert_samples[idx].append(s)

    expert_train_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab, max_len=max_len),
                   batch_size=64, shuffle=True)
        for sub in expert_samples
    ]

    num_classes = len(label_vocab)
    vocab_size = len(tokenizer)
    print("Using vocab_size =", vocab_size)

    model = ModularAddModel(
        vocab_size=vocab_size,
        num_classes=num_classes,
        num_experts=5,
        d_model=512,
        n_heads=4,
        num_layers=4,
        max_len=max_len,
    ).to(DEVICE)

    # 3) 训练 Base
    print("=== Train Base on all sums ===")
    train_base(model, train_loader, epochs=10, lr=1e-4)

    print("=== Evaluate Base-only on TRAIN ===")
    evaluate(model, train_loader, expert_mask=None, desc="Base-only-Train")
    print("=== Evaluate Base-only on TEST ===")
    evaluate(model, test_loader, expert_mask=None, desc="Base-only-Test")

    # 4) 训练 5 个 residual experts
    for ei in range(5):
        print("=== Train Expert {} on its range ===".format(ei))
        train_expert(model, expert_train_loaders[ei], expert_idx=ei, epochs=5, lr=5e-4)

    # 5) 评估 Base + All Experts
    print("=== Evaluate Base + All Experts on TRAIN ===")
    mask_all = torch.ones(5, dtype=torch.long, device=DEVICE)
    evaluate(model, train_loader, expert_mask=mask_all, desc="Base+AllExperts-Train")

    print("=== Evaluate Base + All Experts on TEST ===")
    evaluate(model, test_loader, expert_mask=mask_all, desc="Base+AllExperts-Test")


if __name__ == "__main__":
    main()
