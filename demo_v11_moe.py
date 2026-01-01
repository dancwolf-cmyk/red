# -*- coding: utf-8 -*-
"""
demo_v11_moe.py
----------------
在 demo_v11 已经训练好的 base + experts 上，
新增一个“经典 MoE Gating”基线：

- 输出：p(y|x) = sum_i alpha_i(x) * logits_i(x)
- 不带 residual（没有 base + ...）
- 只训练 router（gating），其余部分全部冻结
- 损失：对最终分类输出做 CrossEntropyLoss（标签为类 y）
"""

from pathlib import Path
import json
import random
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ========================
# 配置
# ========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_SAMPLES = 80000
DEFAULT_SEED = 42

VAL_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64

MAX_EPOCHS_MOE = 1
PATIENCE_MOE = 3

# 这里仍然复用原来的 checkpoints 目录和模型路径
SAVE_DIR = "checkpoints"
LOG_FILE = Path(SAVE_DIR) / "demo_v11_moe.log"
MODEL_PATH = r"c:/temp/lunwen/Qwen3-0.6B"

SAMPLES_CACHE_FILE = Path(SAVE_DIR) / "demo_v11_samples.json"


# ========================
# 日志
# ========================

def setup_logger():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo_v11_moe")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


logger = setup_logger()


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# 数据构造 / 缓存
# ========================

DOMAIN_ORDER = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]


def save_samples_to_cache(samples: List[Dict], label_vocab: List[str], path: Path = SAMPLES_CACHE_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "label_vocab": label_vocab}, f, ensure_ascii=False)
    logger.info("=> cached samples to {} (n={})".format(path, len(samples)))


def load_samples_from_cache(path: Path = SAMPLES_CACHE_FILE):
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples"), data.get("label_vocab")
    return data, None


def build_addition_dataset(num_samples=20000, seed=DEFAULT_SEED):
    random.seed(seed)
    ranges = [
        ("range_2_20", 2, 20),
        ("range_21_40", 21, 40),
        ("range_41_60", 41, 60),
        ("range_61_80", 61, 80),
        ("range_81_100", 81, 100),
    ]
    per_domain = num_samples // len(ranges)
    remainder = num_samples - per_domain * len(ranges)
    target_counts = {name: per_domain for name, _, _ in ranges}
    idx = 0
    while remainder > 0:
        name, _, _ = ranges[idx % len(ranges)]
        target_counts[name] += 1
        remainder -= 1
        idx += 1

    samples = []
    for name, lo, hi in ranges:
        collected = 0
        while collected < target_counts[name]:
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            s = a + b
            if lo <= s <= hi:
                q = "What is {} plus {}?".format(a, b)
                samples.append({"question": q, "answer": str(s), "domain": name})
                collected += 1

    random.shuffle(samples)
    label_vocab = [str(i) for i in range(2, 101)]
    return samples, label_vocab


def load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED, cache_path: Path = SAMPLES_CACHE_FILE):
    cached_samples, cached_vocab = load_samples_from_cache(cache_path)
    if cached_samples is not None and len(cached_samples) == num_samples:
        label_vocab = cached_vocab or [str(i) for i in range(2, 101)]
        logger.info("=> loaded cached samples from {} (n={})".format(cache_path, len(cached_samples)))
        return cached_samples, label_vocab
    samples, label_vocab = build_addition_dataset(num_samples=num_samples, seed=seed)
    save_samples_to_cache(samples, label_vocab, cache_path)
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


# ========================
# 模型定义
# ========================

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


class ModularAddModelWithMoE(nn.Module):
    """
    从 demo_v11 的 ModularAddModelWithRouter 精简而来：

    - base_head 仍然保留，仅用于加载已有 checkpoint（不参与 MoE 输出）。
    - experts: 5 个全连接头，每个输出 num_classes 维 logits。
    - router: 线性层，输出 5 维 gating logits。
    - forward(mode="moe")：
        1) h = encoder(...)
        2) experts_logits: [B, 5, C]
        3) alpha = softmax(router(h))  # [B, 5]
        4) out = sum_i alpha_i * logits_i  # [B, C]
    """

    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size
        self.base_head = nn.Linear(H, num_classes)
        self.experts = nn.ModuleList([nn.Linear(H, num_classes) for _ in range(num_experts)])
        self.router = nn.Linear(H, num_experts)
        self.num_experts = num_experts

    def forward(self, input_ids, attention_mask, mode="moe", expert_mask=None):
        h = self.encoder(input_ids, attention_mask)  # [B, H]

        # 只实现我们需要的两种模式：base / moe
        if mode == "base":
            return self.base_head(h)

        if mode == "moe":
            # 经典 MoE：softmax gating + experts logits 的加权和
            exp_logits = torch.stack([exp(h) for exp in self.experts], dim=1)  # [B, K, C]
            gate_logits = self.router(h)                                       # [B, K]
            alpha = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)           # [B, K, 1]
            mixed = (exp_logits * alpha).sum(dim=1)                            # [B, C]
            return mixed

        raise ValueError("Unknown mode: {}".format(mode))


# ========================
# Checkpoint 辅助函数
# ========================

def get_best_checkpoint(stage_name: str):
    # 复用 demo_v11 的命名规则： demo_v11_{stage_name}_best*.pt
    matches = sorted(Path(SAVE_DIR).glob("demo_v11_{}_best*.pt".format(stage_name)))
    if not matches:
        return None
    # 优先非 slim
    non_slim = [m for m in matches if ".slim" not in m.name]
    if non_slim:
        return non_slim[-1]
    return matches[-1]


def load_best_checkpoint(stage_name: str, model: nn.Module, strict: bool = False):
    path = get_best_checkpoint(stage_name)
    if not path or not path.exists():
        logger.warning("=> no checkpoint found for stage {} (path pattern: demo_v11_{}_best*.pt)".format(
            stage_name, stage_name
        ))
        return None
    try:
        state = torch.load(path, map_location=DEVICE)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if not strict:
            logger.info("=> loaded best checkpoint for {}: {} (missing={} unexpected={})".format(
                stage_name, path, len(missing), len(unexpected)
            ))
        else:
            logger.info("=> loaded best checkpoint for {}: {}".format(stage_name, path))
        return path
    except RuntimeError as e:
        logger.warning("=> failed to load checkpoint {}: {}".format(path, e))
        return None


# ========================
# 训练 / 评估
# ========================

def _get_accum_steps(loader: DataLoader, virtual_batch_size: int):
    unit = getattr(loader, "batch_size", 1) or 1
    return max(1, virtual_batch_size // unit)


def train_moe_router(model: ModularAddModelWithMoE,
                     loader: DataLoader,
                     epochs: int = MAX_EPOCHS_MOE,
                     lr: float = 5e-4,
                     virtual_batch_size: int = VIRTUAL_BATCH_SIZE):
    """
    只训练 router，冻结 encoder / base_head / experts：
    - loss: CE(moe_logits, labels)
    """
    # 冻结除 router 以外的所有参数
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

    best_loss = float("inf")
    best_epoch = 0
    patience_cnt = 0
    best_path = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        iters = 0
        grad_step = 0

        pbar = tqdm(loader, desc="[MoE] epoch {}/{}".format(epoch, epochs), ncols=120)
        optim.zero_grad()

        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            logits = model(ids, mask, mode="moe")
            loss = ce(logits, label) / float(accum_steps)
            loss.backward()
            grad_step += 1

            if grad_step % accum_steps == 0:
                optim.step()
                optim.zero_grad()

            pred = logits.argmax(-1)
            acc = (pred == label).float().mean().item()

            total_loss += loss.item() * float(accum_steps)
            total_acc += acc
            iters += 1

            pbar.set_postfix(
                loss="{:.4f}".format(total_loss / float(iters)),
                acc="{:.4f}".format(total_acc / float(iters)),
            )

        if grad_step % accum_steps != 0:
            optim.step()
            optim.zero_grad()

        avg_loss = total_loss / float(iters if iters > 0 else 1)
        avg_acc = total_acc / float(iters if iters > 0 else 1)
        logger.info("=> MoE epoch {} avg_loss={:.4f} avg_acc={:.4f}".format(epoch, avg_loss, avg_acc))

        # early stop
        if avg_loss < best_loss:
            if best_path and best_path.exists():
                best_path.unlink()
            best_loss = avg_loss
            best_epoch = epoch
            best_path = Path(SAVE_DIR) / "demo_v11_moe_best_ep{}.pt".format(epoch)
            torch.save({"state_dict": model.state_dict(),
                        "stage": "moe",
                        "epoch": epoch},
                       best_path)
            logger.info("=> new best MoE checkpoint: {} (loss={:.4f})".format(best_path, best_loss))
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_MOE:
                logger.info("=> Early stopping MoE at epoch {} (patience={})".format(epoch, PATIENCE_MOE))
                break

    logger.info("=> MoE training finished. best_epoch={} best_loss={:.4f}".format(best_epoch, best_loss))
    return best_path


@torch.no_grad()
def evaluate(model: ModularAddModelWithMoE, loader: DataLoader, mode: str, desc: str):
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        logits = model(ids, mask, mode=mode)
        pred = logits.argmax(-1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())
    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0)
    labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0)
    acc = float(correct) / float(total if total > 0 else 1)
    # macro F1 over full label space
    num_classes = len(loader.dataset.label_to_id)
    f1s = []
    for c in range(num_classes):
        tp = ((preds_cat == c) & (labels_cat == c)).sum().item()
        fp = ((preds_cat == c) & (labels_cat != c)).sum().item()
        fn = ((preds_cat != c) & (labels_cat == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0
    logger.info("[Eval] {} mode={} acc={:.4f} macro_f1={:.4f} n={}".format(desc, mode, acc, macro_f1, total))
    return acc, macro_f1


@torch.no_grad()
def evaluate_per_range(model: ModularAddModelWithMoE,
                       loaders_per_range: List[DataLoader],
                       mode: str,
                       tag: str):
    model.eval()
    ranges = ["2-20", "21-40", "41-60", "61-80", "81-100"]
    logger.info("=== Per-range evaluation: {} (mode={}) ===".format(tag, mode))
    for i, loader in enumerate(loaders_per_range):
        total_correct = 0
        total_count = 0
        all_preds = []
        all_labels = []
        pbar = tqdm(loader, desc="[{}] Range {}".format(tag, ranges[i]), ncols=120)
        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids, mask, mode=mode)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        acc = float(total_correct) / float(total_count if total_count > 0 else 1)
        preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0)
        labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0)
        num_classes = len(loader.dataset.label_to_id)
        f1s = []
        for c in range(num_classes):
            tp = ((preds_cat == c) & (labels_cat == c)).sum().item()
            fp = ((preds_cat == c) & (labels_cat != c)).sum().item()
            fn = ((preds_cat != c) & (labels_cat == c)).sum().item()
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
        macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0
        logger.info("Range {}: acc={:.4f} macro_f1={:.4f} (n={})".format(ranges[i], acc, macro_f1, total_count))


# ========================
# 主流程
# ========================

def main():
    set_seed(DEFAULT_SEED)

    # 1) 加载 / 生成数据
    samples, label_vocab = load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED)
    random.seed(DEFAULT_SEED + 1)
    val_size = int(len(samples) * VAL_RATIO)
    test_size = int(len(samples) * TEST_RATIO)
    train_size = len(samples) - val_size - test_size

    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def make_loader(dataset, shuffle=False, seed_offset=0):
        gen = torch.Generator()
        gen.manual_seed(DEFAULT_SEED + seed_offset)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)

    train_ds = QADataset(train_samples, tokenizer, label_vocab)
    val_ds = QADataset(val_samples, tokenizer, label_vocab)
    test_ds = QADataset(test_samples, tokenizer, label_vocab)
    all_ds = QADataset(samples, tokenizer, label_vocab)

    train_loader = make_loader(train_ds, shuffle=True, seed_offset=0)
    val_loader = make_loader(val_ds, shuffle=False, seed_offset=1)
    test_loader = make_loader(test_ds, shuffle=False, seed_offset=2)
    all_loader = make_loader(all_ds, shuffle=False, seed_offset=3)

    # 为 per-range 评估构造 loader
    domain_to_expert = {
        "range_2_20": 0,
        "range_21_40": 1,
        "range_41_60": 2,
        "range_61_80": 3,
        "range_81_100": 4,
    }
    expert_all = [[] for _ in range(5)]
    for s in samples:
        expert_all[domain_to_expert[s["domain"]]].append(s)

    expert_all_loaders = [
        make_loader(QADataset(sub, tokenizer, label_vocab), shuffle=False, seed_offset=30 + i)
        for i, sub in enumerate(expert_all)
    ]

    # 2) 构建模型 & 加载 base / experts checkpoint
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = ModularAddModelWithMoE(encoder, num_classes=len(label_vocab)).to(DEVICE)

    # 首先加载 base 最佳 checkpoint（会把 base_head 和 encoder 的权重都带进来）
    load_best_checkpoint("base", model, strict=False)
    # 再加载 expert_0 ... expert_4 最佳 checkpoint
    for ei in range(5):
        load_best_checkpoint("expert_{}".format(ei), model, strict=False)

    # 3) 可选：评估一下 base-only 的表现（复现你 log 里的 base acc=0.5460 左右）
    # evaluate(model, all_loader, mode="base", desc="Base-only All")

    # 4) 训练 MoE router（只训练 router，一切从 base+experts 现有权重出发）
    best_moe_ckpt = train_moe_router(model, train_loader,
                                     epochs=MAX_EPOCHS_MOE,
                                     lr=5e-4,
                                     virtual_batch_size=VIRTUAL_BATCH_SIZE)

    # 如果有最优 checkpoint，再载入以确保评估的是最优点
    if best_moe_ckpt is not None and best_moe_ckpt.exists():
        state = torch.load(best_moe_ckpt, map_location=DEVICE)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(sd, strict=False)
        logger.info("=> loaded best MoE checkpoint from {}".format(best_moe_ckpt))

    # 5) 评估 MoE 整体表现 + 分区间表现
    evaluate(model, all_loader, mode="moe", desc="MoE All")
    evaluate(model, train_loader, mode="moe", desc="MoE Train")
    evaluate(model, val_loader, mode="moe", desc="MoE Val")
    evaluate(model, test_loader, mode="moe", desc="MoE Test")

    evaluate_per_range(model, expert_all_loaders, mode="moe", tag="MoE All per-range")


if __name__ == "__main__":
    main()
