# -*- coding: utf-8 -*-
"""
demo_router_qwen3.py
--------------------
使用 Qwen3-0.6B 作为共享 Transformer encoder，
在其顶部加一个分类头（base_head）+ 5 个 residual expert + 简化 router：

  logits_base(x) = W_base * h(x)
  logits_i(x)    = W_i * h(x)        # 5 个 expert 的 residual logit
  Base-only      : logits_base(x)
  AllExperts     : logits_base(x) + alpha * sum_i logits_i(x)
  Router         : logits_base(x) + beta  * sum_i alpha_i(x) * logits_i(x)

数据集：自然语言加法 QA
  Q: "What is 13 plus 7?"
  A: "20"
1 <= a,b <= 100, 2 <= s <= 100
按 s 分成 5 个区间：
  [2,20], [21,40], [41,60], [61,80], [81,100]
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
    1 <= a,b <= 100; 2 <= a+b <= 100
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
        if s < 2 or s > 100:
            continue
        domain = sum_to_domain(s)
        if domain is None:
            continue

        q = "What is {} plus {}?".format(a, b)
        a_str = str(s)
        samples.append({"question": q, "answer": a_str, "domain": domain})

    # label 空间为 "2" ~ "100"
    label_vocab = [str(i) for i in range(2, 201)]
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
# 3. Qwen3 作为 Encoder
# ========================

class QwenEncoder(nn.Module):
    """
    用 Qwen3-0.6B 作为 encoder：
      h(x) = masked average pooling(last_hidden_state)
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        # 这里用 AutoModel 即可（只要是 CausalLM 检查点也能被当成 BaseModel 加载）
        self.model = AutoModel.from_pretrained(model_name_or_path)

        # 一般 Qwen 不带 pad_token，需要手动在 tokenizer 那里设置 pad_token=eos_token；
        # 这里不处理 tokenizer，只做 forward。
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: (B, L, hidden_size)
        h = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        h_sum = (h * mask).sum(dim=1)       # (B, hidden_size)
        lengths = mask.sum(dim=1).clamp(min=1)
        h_pooled = h_sum / lengths          # (B, hidden_size)
        return h_pooled


# ========================
# 4. Base + Experts + Router
# ========================

class ModularAddModelWithRouter(nn.Module):
    def __init__(self, encoder: QwenEncoder, num_classes: int, num_experts: int = 5):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.hidden_size

        # base head：Qwen embedding -> logit(2~100 的 99 类)
        self.base_head = nn.Linear(d_model, num_classes)

        # 5 个 residual expert（线性头）
        self.experts = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(num_experts)]
        )

        # Router: Linear(h) -> 5 维 gating logits
        self.router = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts

        # 残差缩放系数（可调 / 可训练），先固定为 0.1
        self.router_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, attention_mask,
                mode: str = "base", expert_mask=None):
        """
        mode:
          - 'base'      : 只用 base_head
          - 'all_fixed' : Base + alpha * sum_i expert_i (expert_mask 决定哪些 expert 激活，默认全激活)
          - 'router'    : Base + beta  * Σ_i α_i(x) * expert_i
        """
        h = self.encoder(input_ids, attention_mask)   # (B, d_model)
        logits_base = self.base_head(h)               # (B, C)

        if mode == "base":
            return logits_base

        # 计算所有 experts 的 logits
        expert_logits_list = [exp(h) for exp in self.experts]  # list of (B, C)
        expert_logits = torch.stack(expert_logits_list, dim=1) # (B, E, C)

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
# 5. 训练函数
# ========================

def train_base(model, dataloader, epochs=6, lr=1e-5):
    """
    训练 base：Qwen encoder + base_head
    小心 Qwen 比较大，lr 可以设小一点，batch 也可酌情调小（比如 16）。
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    # 不训练 experts、router
    for p in model.experts.parameters():
        p.requires_grad = False
    for p in model.router.parameters():
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


def train_expert(model, dataloader, expert_idx, epochs=10, lr=5e-4):
    """
    冻结 Qwen encoder + base，仅训练第 expert_idx 个 residual 头，
    使用 all_fixed 模式 + 轻微 L2 正则。
    """
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = False
    for i, expert in enumerate(model.experts):
        for p in expert.parameters():
            p.requires_grad = (i == expert_idx)
    for p in model.router.parameters():
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
            logits = model(
                input_ids, attention_mask,
                mode="all_fixed", expert_mask=mask
            )

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


def train_router_domain(model, dataloader, domain_to_id, epochs=10, lr=5e-4):
    """
    冻结 encoder/base/experts，只训练 router。
    用 domain label 做 5 分类，监督 softmax gate。
    """
    model.train()
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
    TOKENIZER_PATH = r"E:\dev\test2\Qwen3-0.6B"  # 改成你的本地 Qwen3-0.6B 路径
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 若 tokenizer 没有 pad_token，设为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = 32
    train_ds = QADataset(train_samples, tokenizer, label_vocab, max_len=max_len)
    test_ds = QADataset(test_samples, tokenizer, label_vocab, max_len=max_len)

    # Qwen 比较大，建议 batch_size 先用 16，看显存情况再调
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

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
                   batch_size=32, shuffle=False)
        for sub in expert_test
    ]

    num_classes = len(label_vocab)

    # 构建 Qwen encoder + 模块化头
    encoder = QwenEncoder(TOKENIZER_PATH).to(DEVICE)
    model = ModularAddModelWithRouter(
        encoder=encoder,
        num_classes=num_classes,
        num_experts=5,
    ).to(DEVICE)

    # Stage 1: Base
    print("=== Train Base on all sums ===")
    train_base(model, train_loader, epochs=6, lr=1e-5)
    print("=== Eval Base-only (global) ===")
    evaluate(model, train_loader, mode="base", desc="Base-only-Train")
    evaluate(model, test_loader, mode="base", desc="Base-only-Test")

    # Stage 2: Experts
    for ei in range(5):
        print("=== Train Expert {} on its range ===".format(ei))
        train_expert(model, expert_train_loaders[ei], expert_idx=ei, epochs=10, lr=5e-4)

    print("=== Eval Base+AllExperts (fixed alpha=0.1, global) ===")
    evaluate(model, train_loader, mode="all_fixed", desc="AllExperts-Train")
    evaluate(model, test_loader, mode="all_fixed", desc="AllExperts-Test")

    # Stage 3: Router
    print("=== Train Router on full train set ===")
    train_router_domain(model, train_loader, domain_to_expert, epochs=10, lr=5e-4)

    print("=== Eval Base+Router (global) ===")
    evaluate(model, train_loader, mode="router", desc="Router-Train")
    evaluate(model, test_loader, mode="router", desc="Router-Test")

    # 分区评估：Base / AllFixed / Router
    evaluate_range_with_mode(model, expert_test_loaders, mode="base", tag="Base-only")
    evaluate_range_with_mode(model, expert_test_loaders, mode="all_fixed", tag="AllExperts-fixed")
    evaluate_range_with_mode(model, expert_test_loaders, mode="router", tag="Router")


if __name__ == "__main__":
    main()
