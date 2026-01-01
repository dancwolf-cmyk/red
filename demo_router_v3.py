# -*- coding: utf-8 -*-
"""
demo_router_qwen_base.py
------------------------
使用 Qwen3-0.6B 作为“冻结的共享 encoder”，在其 pooled 表达上叠加：
  - base_head（全局分类头）
  - 5 个 residual experts（线性层）
  - 一个简化版 router：α(x) = softmax(Linear(h(x)))

支持三种模式：
  1) Base-only
  2) Base + AllExperts (固定 residual scale = 0.1)
  3) Base + Router (learned gating, residual scale = router_scale)
"""

import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================
# 1. 构造加法 QA 数据集
# ========================

def build_addition_dataset(num_samples=20000, seed=42):
    """
    自然语言加法：
      Q: "What is 13 plus 7?"
      A: "20"

    这里保持和之前 toy 一致：
      1 <= a,b <= 100
      2 <= s = a+b <= 100   （大于 100 的和直接丢弃）
    并按 sum 分 5 个区间：
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
        return None

    while len(samples) < num_samples:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        s = a + b
        # 这里只保留和在 [2,100] 的样本
        if s < 2 or s > 100:
            continue
        domain = sum_to_domain(s)
        if domain is None:
            continue

        q = "What is {} plus {}?".format(a, b)
        a_str = str(s)
        samples.append({"question": q, "answer": a_str, "domain": domain})

    # 标签空间：2..100 共 99 类
    label_vocab = [str(i) for i in range(2, 101)]
    return samples, label_vocab


# ========================
# 2. Dataset
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
# 3. Qwen Encoder 封装（冻结）
# ========================

class QwenEncoderFrozen(nn.Module):
    """
    用 Qwen3-0.6B 做编码器：
      - 冻结所有参数
      - forward 中使用 torch.no_grad()
      - 输出 mean-pooled 句向量 h_pooled (B, hidden_size)
    """
    def __init__(self, model_path_or_name, max_len=64):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_path_or_name,
            trust_remote_code=True
        )
        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        # 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        # Qwen 通常可以接受较长序列，这里简单截断保护一下
        if input_ids.size(1) > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            attention_mask = attention_mask[:, :self.max_len]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 大部分 transformer 模型都有 last_hidden_state
            h = outputs.last_hidden_state  # (B, L, hidden_size)

        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        h_sum = (h * mask).sum(dim=1)       # (B, hidden_size)
        lengths = mask.sum(dim=1).clamp(min=1)
        h_pooled = h_sum / lengths          # (B, hidden_size)
        return h_pooled


# ========================
# 4. 模型定义：Base + Experts + Router
# ========================

class ModularAddModelWithRouter(nn.Module):
    def __init__(self, encoder: QwenEncoderFrozen, num_classes,
                 num_experts=5):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.hidden_size

        # 全局分类头（base）
        self.base_head = nn.Linear(d_model, num_classes)

        # 5 个 residual expert 头
        self.experts = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(num_experts)]
        )

        # 简化版 router：Linear(h) -> logits(E) -> softmax
        self.router = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts

        # residual 缩放因子（初始 0.1，可训练）
        self.router_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, attention_mask,
                mode="base", expert_mask=None):
        """
        mode:
          - 'base': 只用 base_head
          - 'all_fixed': Base + alpha * sum_i expert_i (expert_mask 决定哪些 expert 激活)
          - 'router': Base + Σ_i α_i(x) * expert_i
        """
        h = self.encoder(input_ids, attention_mask)  # (B, d_model)
        logits_base = self.base_head(h)              # (B, C)

        if mode == "base":
            return logits_base

        # 计算所有 experts 的 logits
        expert_logits_list = [exp(h) for exp in self.experts]   # list of (B, C)
        expert_logits = torch.stack(expert_logits_list, dim=1)  # (B, E, C)

        if mode == "all_fixed":
            alpha = 0.1
            if expert_mask is None:
                mask = torch.ones(self.num_experts, dtype=torch.long, device=h.device)
            else:
                mask = expert_mask.long().to(h.device)
            mask_f = mask.view(1, self.num_experts, 1).float()  # (1, E, 1)
            exp_sum = (expert_logits * mask_f).sum(dim=1)       # (B, C)
            logits = logits_base + alpha * exp_sum
            return logits

        if mode == "router":
            # Router: α(x) = softmax(Linear(h))
            gate_logits = self.router(h)                 # (B, E)
            alpha = torch.softmax(gate_logits, dim=-1)   # (B, E)
            alpha_expanded = alpha.unsqueeze(-1)         # (B, E, 1)
            weighted = (expert_logits * alpha_expanded).sum(dim=1)  # (B, C)
            logits = logits_base + self.router_scale * weighted
            return logits

        raise ValueError("Unknown mode: {}".format(mode))


# ========================
# 5. 训练函数（只训头，不训 Qwen）
# ========================

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def train_base(model, dataloader, epochs=6, lr=1e-4):
    """
    仅训练 base_head，Qwen encoder 和 experts/router 全部冻结。
    """
    model.train()
    # 冻结 encoder, experts, router
    set_requires_grad(model.encoder, False)
    for exp in model.experts:
        set_requires_grad(exp, False)
    set_requires_grad(model.router, False)
    # 只训练 base_head
    set_requires_grad(model.base_head, True)

    optim = torch.optim.Adam(model.base_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask, mode="base")
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
    """
    冻结 encoder + base_head + 其他 experts + router，只训练第 expert_idx 个 expert。
    """
    model.train()
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.base_head, False)
    for i, exp in enumerate(model.experts):
        set_requires_grad(exp, i == expert_idx)
    set_requires_grad(model.router, False)

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

            # 用 all_fixed 模式，同时只激活当前 expert
            mask = torch.zeros(model.num_experts, dtype=torch.long, device=DEVICE)
            mask[expert_idx] = 1
            logits = model(input_ids, attention_mask,
                           mode="all_fixed", expert_mask=mask)

            ce = criterion(logits, labels)

            # 轻微 L2 正则：防止 expert 太爆
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


def train_router_domain(model, dataloader, domain_to_id, epochs=5, lr=5e-4):
    """
    冻结 encoder + base_head + experts，只训练 router。
    用 domain label 做 5 分类，监督 softmax gate。
    """
    model.train()
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.base_head, False)
    for exp in model.experts:
        set_requires_grad(exp, False)
    set_requires_grad(model.router, True)

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
            gate_logits = model.router(h)  # (B, 5)
            loss = criterion(gate_logits, domain_ids)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * domain_ids.size(0)
            preds = gate_logits.argmax(dim=-1)
            total_correct += (preds == domain_ids).sum().item()
            total_count += domain_ids.size(0)

        print("[Router-domain] epoch {:3d}, loss={:.4f}, acc={:.4f}".format(
            ep, total_loss / total_count, total_correct / total_count
        ))


# ========================
# 6. 评估
# ========================

@torch.no_grad()
def evaluate(model, dataloader, mode="base", desc="", expert_mask=None):
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
    print("[Eval] {} accuracy: {:.4f}".format(desc, acc))
    return acc


@torch.no_grad()
def evaluate_range_with_mode(model, loaders_per_range, mode, tag, num_experts=5):
    """
    用某个 mode 在每个区间上评估。
    对于 'all_fixed'，统一使用全 1 mask。
    对于 'router' / 'base'，不需要 mask。
    """
    print("\n=== Per-range evaluation: {} (mode={}) ===".format(tag, mode))
    ranges = ("2-20", "21-40", "41-60", "61-80", "81-100")
    for i, loader in enumerate(loaders_per_range):
        if mode == "all_fixed":
            mask = torch.ones(num_experts, dtype=torch.long, device=DEVICE)
            acc = evaluate(model, loader, mode=mode,
                           desc="{} [{}]".format(tag, ranges[i]),
                           expert_mask=mask)
        else:
            acc = evaluate(model, loader, mode=mode,
                           desc="{} [{}]".format(tag, ranges[i]))
        print("Range {}: acc={:.4f}".format(ranges[i], acc))


# ========================
# 7. 主流程
# ========================

def main():
    # 数据
    samples, label_vocab = build_addition_dataset(num_samples=20000, seed=42)
    print("Total samples:", len(samples))
    print("Label vocab size:", len(label_vocab))

    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    test_samples = samples[split:]

    # tokenizer
    TOKENIZER_PATH = r"E:\dev\test2\Qwen3-0.6B"  # 改成你的本地路径
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    max_len = 32
    train_ds = QADataset(train_samples, tokenizer, label_vocab, max_len=max_len)
    test_ds = QADataset(test_samples, tokenizer, label_vocab, max_len=max_len)

    # Qwen 比较大，batch_size 建议 16 左右
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # domain -> expert
    domain_to_expert = {
        "range_2_20": 0,
        "range_21_40": 1,
        "range_41_60": 2,
        "range_61_80": 3,
        "range_81_100": 4,
    }

    # 每个 expert 的 train/test 子集
    expert_train = [[] for _ in range(5)]
    expert_test = [[] for _ in range(5)]
    for s in train_samples:
        idx = domain_to_expert[s["domain"]]
        expert_train[idx].append(s)
    for s in test_samples:
        idx = domain_to_expert[s["domain"]]
        expert_test[idx].append(s)

    expert_train_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab, max_len=max_len),
                   batch_size=16, shuffle=True)
        for sub in expert_train
    ]
    expert_test_loaders = [
        DataLoader(QADataset(sub, tokenizer, label_vocab, max_len=max_len),
                   batch_size=16, shuffle=False)
        for sub in expert_test
    ]

    num_classes = len(label_vocab)

    # 构建冻结 Qwen encoder + modular heads
    encoder = QwenEncoderFrozen(TOKENIZER_PATH, max_len=max_len).to(DEVICE)
    model = ModularAddModelWithRouter(
        encoder=encoder,
        num_classes=num_classes,
        num_experts=5,
    ).to(DEVICE)

    # Stage 1: Base
    print("=== Train Base (Qwen frozen) on all sums ===")
    train_base(model, train_loader, epochs=6, lr=1e-4)
    print("=== Eval Base-only (global) ===")
    evaluate(model, train_loader, mode="base", desc="Base-only-Train")
    evaluate(model, test_loader, mode="base", desc="Base-only-Test")

    # Stage 2: Experts
    for ei in range(5):
        print("=== Train Expert {} on its range ===".format(ei))
        train_expert(model, expert_train_loaders[ei],
                     expert_idx=ei, epochs=10, lr=5e-4)

    print("=== Eval Base+AllExperts (fixed alpha=0.1, global) ===")
    evaluate(model, train_loader, mode="all_fixed", desc="AllExperts-Train")
    evaluate(model, test_loader, mode="all_fixed", desc="AllExperts-Test")

    # Stage 3: Router
    print("=== Train Router on full train set ===")
    train_router_domain(model, train_loader, domain_to_expert,
                        epochs=10, lr=5e-4)

    print("=== Eval Base+Router (global) ===")
    evaluate(model, train_loader, mode="router", desc="Router-Train")
    evaluate(model, test_loader, mode="router", desc="Router-Test")

    # 分区评估：Base / AllFixed / Router
    evaluate_range_with_mode(model, expert_test_loaders,
                             mode="base", tag="Base-only")
    evaluate_range_with_mode(model, expert_test_loaders,
                             mode="all_fixed", tag="AllExperts-fixed")
    evaluate_range_with_mode(model, expert_test_loaders,
                             mode="router", tag="Router")


if __name__ == "__main__":
    main()
