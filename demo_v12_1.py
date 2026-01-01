# -*- coding: utf-8 -*-
"""
demo_v12_1.py
-----------
Task 2: Mixed Arithmetic (Addition + Subtraction, Multi-step)
- 统一脚本：同时包含 RED (residual experts + router) 和经典 MoE baseline。
- 数据：2~4 步加减混合表达式，结果仍限制在 [2, 100]，并映射到 5 个区间域。

使用阶段：
  1) base: 训练全局头（少量平衡子集）
  2) experts: 为每个区间训练一个 expert（all_fixed 模式）
  3) all_fixed: 评估所有 expert 固定加权的表现
  4) router (RED): 用 domain 标签训练 router，得到 residual MoE (RED)
  5) moe: 在 base+experts 的基础上训练经典 MoE router (labels 为具体 class)

最终输出日志包括：
  - base / all_fixed / router / moe 的 overall acc + macro-F1
  - 每个方法的 per-range acc + macro-F1
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
# 全局配置
# ========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_SAMPLES = 80000
DEFAULT_SEED = 42

BASE_RATIO = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64

MAX_EPOCHS_BASE = 10
MAX_EPOCHS_EXPERT = 60
MAX_EPOCHS_ROUTER = 8
MAX_EPOCHS_MOE = 3

PATIENCE = 3
PATIENCE_MOE = 3

SAVE_DIR = "checkpoints_v12"
LOG_FILE = Path(SAVE_DIR) / "demo_v12.log"
MODEL_PATH = r"e:/dev/lunwen/Qwen3-0.6B"

STAGE_STATUS_FILE = Path(SAVE_DIR) / "stage_status.json"
SAMPLES_CACHE_FILE = Path(SAVE_DIR) / "demo_v12_samples_mixed.json"
AMP_DTYPE = torch.bfloat16
DOMAIN_ORDER = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]
try:
    from packaging import version as _pkg_version
except ImportError:
    _pkg_version = None

# ========================
# 日志 & 随机种子
# ========================

def setup_logger():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo_v12")
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

def _filter_state_dict_for_stage(model, full_sd, stage_group):
    """
    根据 stage_group（base / expert_i / router / moe），从 full_sd 里筛选出
    只需要保存 / 加载的那部分参数。
    """
    result = {}
    # 当前模型的完整 key 集，用来避免加载不存在的 key
    model_sd_keys = set(model.state_dict().keys())

    # 1) base 阶段：只动 encoder + base_head + router_scale
    if stage_group == "base":
        for k, v in full_sd.items():
            if k.startswith("encoder.") or k.startswith("base_head.") or k == "router_scale":
                if k in model_sd_keys:
                    result[k] = v
        return result

    # 2) expert_i 阶段：只动 experts.i.*
    if stage_group.startswith("expert_"):
        # 解析 expert index，例如 "expert_0" -> 0
        suffix = stage_group[len("expert_"):]  # "0", "1", ...
        idx_str = ""
        for ch in suffix:
            if ch.isdigit():
                idx_str += ch
            else:
                break
        if idx_str != "":
            idx = int(idx_str)
            prefix = "experts." + str(idx) + "."
            for k, v in full_sd.items():
                if k.startswith(prefix) and k in model_sd_keys:
                    result[k] = v
        return result

    # 3) router / moe 阶段：只动 router.*
    #    - RED router: ModularAddModelWithRouter.router
    #    - MoE router: ModularAddModelWithMoE.router
    if stage_group == "router" or stage_group == "moe":
        for k, v in full_sd.items():
            if k.startswith("router.") and k in model_sd_keys:
                result[k] = v
        return result

    # 默认：不做筛选（理论上不会走到这里）
    for k, v in full_sd.items():
        if k in model_sd_keys:
            result[k] = v
    return result

logger = setup_logger()


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# Stage 状态记录（断点控制）
# ========================

def load_stage_status():
    if STAGE_STATUS_FILE.exists():
        with open(STAGE_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_stage_status(status: Dict[str, bool]):
    STAGE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STAGE_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f)


def is_stage_completed(stage_name: str) -> bool:
    status = load_stage_status()
    return status.get(stage_name, False)


def mark_stage_completed(stage_name: str):
    status = load_stage_status()
    status[stage_name] = True
    save_stage_status(status)


# ========================
# 数据构造：混合算术（加减、多步）
# ========================

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


def build_mixed_arithmetic_dataset(num_samples=20000, seed=DEFAULT_SEED):
    """Build the mixed-arithmetic dataset and ensure each domain gets the same number of samples."""
    random.seed(seed)
    ranges = [
        ("range_2_20", 2, 20),
        ("range_21_40", 21, 40),
        ("range_41_60", 41, 60),
        ("range_61_80", 61, 80),
        ("range_81_100", 81, 100),
    ]

    num_domains = len(ranges)
    per_domain = num_samples // num_domains
    remainder = num_samples - per_domain * num_domains
    domain_targets = {}
    for idx, (name, _, _) in enumerate(ranges):
        domain_targets[name] = per_domain + (1 if idx < remainder else 0)

    domain_samples = {name: [] for name, _, _ in ranges}

    for name, lo, hi in ranges:
        target = domain_targets[name]
        while len(domain_samples[name]) < target:
            steps = random.randint(2, 4)
            nums = [random.randint(1, 50) for _ in range(steps)]
            ops = [random.choice(("plus", "minus")) for _ in range(steps - 1)]

            expr_text = str(nums[0])
            result = nums[0]
            for i in range(steps - 1):
                expr_text += " {} {}".format(ops[i], nums[i + 1])
                if ops[i] == "plus":
                    result += nums[i + 1]
                else:
                    result -= nums[i + 1]

            if result < lo or result > hi:
                continue

            q = "What is {}?".format(expr_text)
            domain_samples[name].append({"question": q, "answer": str(result), "domain": name})

    samples = []
    for bucket in domain_samples.values():
        samples.extend(bucket)
    random.shuffle(samples)
    label_vocab = [str(i) for i in range(2, 101)]
    return samples, label_vocab


def stratified_split_samples(samples: List[Dict], val_ratio: float, test_ratio: float, seed: int = DEFAULT_SEED):
    """Split the samples into train/val/test while preserving per-domain balance."""
    buckets = {domain: [] for domain in DOMAIN_ORDER}
    for s in samples:
        buckets[s["domain"]].append(s)
    rand = random.Random(seed)
    for bucket in buckets.values():
        rand.shuffle(bucket)

    train_samples = []
    val_samples = []
    test_samples = []
    for domain in DOMAIN_ORDER:
        bucket = buckets[domain]
        total = len(bucket)
        val_count = int(total * val_ratio)
        test_count = int(total * test_ratio)
        train_count = total - val_count - test_count

        idx = 0
        train_samples.extend(bucket[idx:idx + train_count])
        idx += train_count
        val_samples.extend(bucket[idx:idx + val_count])
        idx += val_count
        test_samples.extend(bucket[idx:idx + test_count])

    rand.shuffle(train_samples)
    rand.shuffle(val_samples)
    rand.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED, cache_path: Path = SAMPLES_CACHE_FILE):
    cached_samples, cached_vocab = load_samples_from_cache(cache_path)
    if cached_samples is not None and len(cached_samples) == num_samples:
        label_vocab = cached_vocab or [str(i) for i in range(2, 101)]
        logger.info("=> loaded cached samples from {} (n={})".format(cache_path, len(cached_samples)))
        return cached_samples, label_vocab
    samples, label_vocab = build_mixed_arithmetic_dataset(num_samples=num_samples, seed=seed)
    save_samples_to_cache(samples, label_vocab, cache_path)
    return samples, label_vocab


def sample_balanced_subset(samples: List[Dict], count: int):
    grouped = {domain: [] for domain in DOMAIN_ORDER}
    for s in samples:
        if s["domain"] in grouped:
            grouped[s["domain"]].append(s)
    for bucket in grouped.values():
        random.shuffle(bucket)

    per_domain = count // len(grouped)
    remainder = count - per_domain * len(grouped)

    selected = []
    taken_counts = {}
    for domain in DOMAIN_ORDER:
        bucket = grouped[domain]
        take = min(per_domain + (1 if remainder > 0 else 0), len(bucket))
        selected.extend(bucket[:take])
        taken_counts[domain] = take
        if take > per_domain:
            remainder -= 1

    for domain in DOMAIN_ORDER:
        if len(selected) >= count:
            break
        bucket = grouped[domain]
        already = taken_counts.get(domain, 0)
        extra = bucket[already:]
        need = min(count - len(selected), len(extra))
        if need > 0:
            selected.extend(extra[:need])
            taken_counts[domain] = already + need

    return selected[:count]


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
# 模型定义：Qwen encoder
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


# ========================
# 模型定义：RED (residual experts + router)
# ========================

class ModularAddModelWithRouter(nn.Module):
    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size

        # base 还是简单的一层线性头
        self.base_head = nn.Linear(H, num_classes)

        # 专家：两层 MLP（H -> 4H -> num_classes）
        hidden_exp = 4 * H
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(H, hidden_exp),
                    nn.GELU(),
                    nn.Linear(hidden_exp, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

        # router 不变
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
            alpha = torch.softmax(self.router(h), dim=-1).unsqueeze(-1)
            weighted = (exp_logits * alpha).sum(dim=1)
            return base + self.router_scale * weighted

        raise ValueError("Unknown mode: {}".format(mode))


# ========================
# 模型定义：经典 MoE baseline
# ========================

class ModularAddModelWithMoE(nn.Module):
    """
    经典 MoE gating：
      - 直接将 experts logits 按 router softmax 加权求和
      - 不包含 base residual
    """

    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size

        # 这里 base_head 主要是为了兼容从 RED 迁移参数，用不用都无所谓
        self.base_head = nn.Linear(H, num_classes)

        # 专家：两层 MLP（H -> 4H -> num_classes）
        hidden_exp = 4 * H
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(H, hidden_exp),
                    nn.GELU(),
                    nn.Linear(hidden_exp, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

        self.router = nn.Linear(H, num_experts)
        self.num_experts = num_experts

    def forward(self, input_ids, attention_mask, mode="moe"):
        h = self.encoder(input_ids, attention_mask)
        if mode == "base":
            return self.base_head(h)
        if mode == "moe":
            exp_logits = torch.stack([exp(h) for exp in self.experts], dim=1)  # [B, K, C]
            gate_logits = self.router(h)                                       # [B, K]
            alpha = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)           # [B, K, 1]
            mixed = (exp_logits * alpha).sum(dim=1)                            # [B, C]
            return mixed
        raise ValueError("Unknown mode: {}".format(mode))


# ========================
# Checkpoint 辅助函数
# ========================

def save_checkpoint(model, stage, epoch=None):
    """
    按 stage 只保存自己那一部分参数：
      - base_best: encoder + base_head + router_scale
      - expert_i_best: experts.i.*
      - router_best: router.*
      - moe_best: router.*  (MoE router)
    """
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    suffix = "_ep" + str(epoch) if epoch is not None else ""
    path = Path(SAVE_DIR) / ("demo_v12_" + stage + suffix + ".pt")

    # 从 stage 名里得到逻辑上的 group 名：base / expert_0 / router / moe
    # 例如 stage="base_best" -> stage_group="base"
    stage_group = stage
    if stage.endswith("_best"):
        stage_group = stage[: -len("_best")]

    full_sd = model.state_dict()
    # 只取本 stage 关心的那部分参数
    filtered_sd = _filter_state_dict_for_stage(model, full_sd, stage_group)

    torch.save(
        {
            "state_dict": filtered_sd,
            "stage": stage_group,
            "epoch": epoch,
        },
        path,
    )
    logger.info("=> checkpoint saved (stage_group={}): {}".format(stage_group, path))
    return path



def get_best_checkpoint(stage_name):
    matches = sorted(Path(SAVE_DIR).glob("demo_v12_{}_best*.pt".format(stage_name)))
    if not matches:
        return None
    non_slim = [m for m in matches if ".slim" not in m.name]
    if non_slim:
        return non_slim[-1]
    return matches[-1]


def load_best_checkpoint(stage_name, model, strict=False):
    """
    按 stage_name（base / expert_i / router / moe）只加载对应的参数：
      - base: encoder + base_head + router_scale
      - expert_i: experts.i.*
      - router: router.*
      - moe: router.*
    其他部分一律保持当前 model 里的数值，不会被覆盖。
    """
    path = get_best_checkpoint(stage_name)
    if (path is None) or (not path.exists()):
        return None

    try:
        state = torch.load(path, map_location=DEVICE)
        full_sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

        # 只选取与当前 stage_name 对应的部分参数
        filtered_sd = _filter_state_dict_for_stage(model, full_sd, stage_name)

        model_sd = model.state_dict()
        updated_keys = 0
        for k, v in filtered_sd.items():
            if k in model_sd:
                model_sd[k] = v
                updated_keys += 1

        # 严格模式在这里意义不大，因为我们是“局部覆盖”
        model.load_state_dict(model_sd, strict=False)

        logger.info(
            "=> loaded best checkpoint for {}: {} (updated {} params)".format(
                stage_name, path, updated_keys
            )
        )
        return path
    except RuntimeError as e:
        logger.warning("=> failed to load checkpoint {} for stage {}: {}".format(path, stage_name, e))
        return None



def run_stage_with_early_stop(stage_name, train_step_fn, model, max_epochs, patience=PATIENCE):
    best_loss = float("inf")
    patience_cnt = 0
    best_epoch = 0
    best_path = None
    for epoch in range(1, max_epochs + 1):
        loss, acc = train_step_fn(epoch)
        if loss < best_loss:
            if best_path and best_path.exists():
                best_path.unlink()
            best_loss = loss
            best_epoch = epoch
            patience_cnt = 0
            best_path = save_checkpoint(model, "{}_best".format(stage_name), epoch=epoch)
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            logger.info("=> Early stopping {} at epoch {} (patience={})".format(stage_name, epoch, patience))
            break
    logger.info("=> {} best_epoch={} best_loss={:.4f}".format(stage_name, best_epoch, best_loss))
    return best_loss, best_epoch, best_path


# ========================
# 训练通用工具
# ========================

def _get_accum_steps(loader: DataLoader, virtual_batch_size: int):
    unit = getattr(loader, "batch_size", 1) or 1
    return max(1, virtual_batch_size // unit)


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
        loss = criterion(logits, label) / float(accum_steps)
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
        pbar.set_postfix(loss="{:.4f}".format(total_loss / float(iters)),
                         acc="{:.4f}".format(total_acc / float(iters)))
    if grad_step % accum_steps != 0:
        optim.step()
        optim.zero_grad()
    avg_loss = total_loss / float(iters if iters > 0 else 1)
    avg_acc = total_acc / float(iters if iters > 0 else 1)
    return avg_loss, avg_acc


# ========================
# RED: base / expert / router 训练
# ========================

def train_base(model, loader, epochs=6, lr=5e-5, virtual_batch_size=VIRTUAL_BATCH_SIZE):
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

    loss, acc = 0.0, 0.0
    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=base_step,
            desc="[Base] epoch {}/{}".format(ep, epochs),
            accum_steps=accum_steps)
        logger.info("=> Base epoch {} avg_loss={:.4f} avg_acc={:.4f}".format(ep, loss, acc))
    return loss, acc


def train_expert(model, loader, ei, epochs=5, lr=5e-4, virtual_batch_size=VIRTUAL_BATCH_SIZE):
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

    loss, acc = 0.0, 0.0
    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=expert_step,
            desc="[Expert {}] epoch {}/{}".format(ei, ep, epochs),
            accum_steps=accum_steps)
        logger.info("=> Expert {} epoch {} avg_loss={:.4f} avg_acc={:.4f}".format(ei, ep, loss, acc))
    return loss, acc


def train_router(model, loader, domain_to_id, epochs=8, lr=5e-4, virtual_batch_size=VIRTUAL_BATCH_SIZE):
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
        h = model.encoder(ids, mask)
        gate_logits = model.router(h)
        return gate_logits, domain_ids

    loss, acc = 0.0, 0.0
    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=router_step,
            desc="[Router] epoch {}/{}".format(ep, epochs),
            accum_steps=accum_steps)
        logger.info("=> Router epoch {} avg_loss={:.4f} avg_acc={:.4f}".format(ep, loss, acc))
    return loss, acc


# ========================
# MoE router 训练
# ========================

def train_moe_router(model: ModularAddModelWithMoE,
                     loader: DataLoader,
                     epochs: int = MAX_EPOCHS_MOE,
                     lr: float = 5e-4,
                     virtual_batch_size: int = VIRTUAL_BATCH_SIZE):
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

        if avg_loss < best_loss:
            if best_path and best_path.exists():
                best_path.unlink()
            best_loss = avg_loss
            best_epoch = epoch
            best_path = Path(SAVE_DIR) / "demo_v12_moe_best_ep{}.pt".format(epoch)
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


# ========================
# 评估函数（带 Macro-F1）
# ========================

@torch.no_grad()
def evaluate(model, loader, mode, desc):
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

    num_classes = len(loader.dataset.label_to_id)
    f1s = []
    for c in range(num_classes):
        tp = ((preds_cat == c) & (labels_cat == c)).sum().item()
        fp = ((preds_cat == c) & (labels_cat != c)).sum().item()
        fn = ((preds_cat != c) & (labels_cat == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / float(denom))
    macro_f1 = sum(f1s) / float(num_classes if num_classes > 0 else 1)

    logger.info("[Eval] {} mode={} acc={:.4f} macro_f1={:.4f} n={}".format(desc, mode, acc, macro_f1, total))
    return acc, macro_f1


@torch.no_grad()
def evaluate_per_range(model, loaders_per_range: List[DataLoader], mode: str, tag: str):
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
            f1s.append(0.0 if denom == 0 else (2 * tp) / float(denom))
        macro_f1 = sum(f1s) / float(num_classes if num_classes > 0 else 1)
        logger.info("Range {}: acc={:.4f} macro_f1={:.4f} (n={})".format(ranges[i], acc, macro_f1, total_count))
@torch.no_grad()
def debug_base_vs_exp0_on_range0(model, range0_loader):
    model.eval()
    total = 0
    correct_base = 0
    correct_mix = 0

    # 专家 0 的 mask
    expert_mask = torch.zeros(model.num_experts, device=DEVICE)
    expert_mask[0] = 1.0

    for batch in range0_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # 1) 纯 base
        logits_base = model(ids, mask, mode="base")
        preds_base = logits_base.argmax(dim=-1)
        correct_base += (preds_base == labels).sum().item()

        # 2) base + expert_0
        logits_mix = model(ids, mask, mode="all_fixed", expert_mask=expert_mask)
        preds_mix = logits_mix.argmax(dim=-1)
        correct_mix += (preds_mix == labels).sum().item()

        total += labels.size(0)

    acc_base = correct_base / float(total if total > 0 else 1)
    acc_mix  = correct_mix  / float(total if total > 0 else 1)

    print("[DEBUG] Range 2-20 | base-only acc       = {:.6f}".format(acc_base))
    print("[DEBUG] Range 2-20 | base + expert_0 acc = {:.6f}".format(acc_mix))


# ========================
# 主流程
# ========================

def main():
    set_seed(DEFAULT_SEED)
    USE_FLASH_SDP = False
    if torch.cuda.is_available():
        try:
            torch_version = torch.__version__
            if _pkg_version is not None:
                if _pkg_version.parse(torch_version) >= _pkg_version.parse("2.0.0"):
                    try:
                        from torch.backends.cuda import sdp_kernel
                        sdp_kernel.enable_flash = True
                        sdp_kernel.enable_mem_efficient = True
                        sdp_kernel.enable_math = True
                        USE_FLASH_SDP = True
                        logger.info("=> SDPA Flash-Attention enabled (torch {})".format(torch_version))
                    except Exception as e:
                        logger.warning("=> Failed to enable SDPA Flash-Attention: {}".format(e))
                else:
                    logger.info("=> Torch {} < 2.0.0, skip SDPA Flash-Attention".format(torch_version))
            else:
                logger.info("=> packaging.version not available, skip precise torch version check")
        except Exception as e:
            logger.warning("=> SDPA setup exception: {}".format(e))
    # 1) 数据
    samples, label_vocab = load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED)
    train_samples, val_samples, test_samples = stratified_split_samples(
        samples, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=DEFAULT_SEED + 1
    )
    train_size = len(train_samples)
    val_size = len(val_samples)
    test_size = len(test_samples)

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

    train_base_count = max(1, int(train_size * BASE_RATIO))
    random.seed(DEFAULT_SEED + 2)
    base_subset = sample_balanced_subset(train_samples, train_base_count)
    base_loader = make_loader(QADataset(base_subset, tokenizer, label_vocab),
                              shuffle=True, seed_offset=4)

    domain_to_expert = {
        "range_2_20": 0,
        "range_21_40": 1,
        "range_41_60": 2,
        "range_61_80": 3,
        "range_81_100": 4,
    }

    expert_train = [[] for _ in range(5)]
    expert_test = [[] for _ in range(5)]
    expert_all = [[] for _ in range(5)]

    for s in train_samples:
        expert_train[domain_to_expert[s["domain"]]].append(s)
    for s in test_samples:
        expert_test[domain_to_expert[s["domain"]]].append(s)
    for s in samples:
        expert_all[domain_to_expert[s["domain"]]].append(s)

    expert_train_loaders = [
        make_loader(QADataset(sub, tokenizer, label_vocab), shuffle=True, seed_offset=10 + i)
        for i, sub in enumerate(expert_train)
    ]
    expert_test_loaders = [
        make_loader(QADataset(sub, tokenizer, label_vocab), shuffle=False, seed_offset=20 + i)
        for i, sub in enumerate(expert_test)
    ]
    base_range_loaders = [
        make_loader(QADataset(sub, tokenizer, label_vocab), shuffle=False, seed_offset=30 + i)
        for i, sub in enumerate(expert_all)
    ]

    # 2) RED 模型
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    red_model = ModularAddModelWithRouter(encoder, num_classes=len(label_vocab)).to(DEVICE)

    # 2.1 base stage
    base_stage = "base"
    # load_best_checkpoint(base_stage, red_model, strict=False)
    # load_best_checkpoint("base", red_model, strict=False)
    # load_best_checkpoint("expert_0", red_model, strict=False)

    # # # 3. 原有的 per-range base 测试（确认 0.3503）
    # # evaluate_per_range(red_model, base_range_loaders, mode="base", tag="Base All per-range")

    # # 4. 调试：同一批 loader 上比较 base vs base+expert_0
    # debug_base_vs_exp0_on_range0(red_model, base_range_loaders[0])
    # evaluate_per_range(red_model, base_range_loaders, mode="base", tag="Base All per-range")
    # exit()
    if not is_stage_completed(base_stage):
        load_best_checkpoint(base_stage, red_model, strict=False)
        _, _, best_path = run_stage_with_early_stop(
            base_stage,
            lambda epoch: train_base(red_model, base_loader, epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE),
            red_model,
            max_epochs=MAX_EPOCHS_BASE,
        )
        mark_stage_completed(base_stage)
        load_best_checkpoint(base_stage, red_model, strict=False)
        logger.info("Starting evaluation (mode=base)")
        evaluate(red_model, all_loader, "base", "Base All")
        evaluate_per_range(red_model, base_range_loaders, mode="base", tag="Base All per-range")
    else:
        load_best_checkpoint(base_stage, red_model, strict=False)

    # 2.2 experts
    for ei in range(5):
        expert_stage = "expert_{}".format(ei)
        if not is_stage_completed(expert_stage):
            load_best_checkpoint(expert_stage, red_model, strict=False)
            _, _, best_path = run_stage_with_early_stop(
                expert_stage,
                lambda epoch, ei=ei: train_expert(red_model, expert_train_loaders[ei], ei, epochs=1,
                                                  virtual_batch_size=VIRTUAL_BATCH_SIZE),
                red_model,
                max_epochs=MAX_EPOCHS_EXPERT,
            )
            mark_stage_completed(expert_stage)
            load_best_checkpoint(expert_stage, red_model, strict=False)
        else:
            load_best_checkpoint(expert_stage, red_model, strict=False)

    logger.info("Starting evaluation (mode=all_fixed)")
    evaluate(red_model, train_loader, "all_fixed", "AllExperts Train")
    evaluate(red_model, val_loader, "all_fixed", "AllExperts Val")
    evaluate(red_model, test_loader, "all_fixed", "AllExperts Test")
    evaluate(red_model, all_loader, "all_fixed", "AllExperts All")
    evaluate_per_range(red_model, expert_test_loaders, mode="all_fixed", tag="AllExperts All per-range")

    # 2.3 router (RED)
    router_stage = "router"
    if not is_stage_completed(router_stage):
        load_best_checkpoint(router_stage, red_model, strict=False)
        _, _, best_path = run_stage_with_early_stop(
            router_stage,
            lambda epoch: train_router(red_model, train_loader, domain_to_expert,
                                       epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE),
            red_model,
            max_epochs=MAX_EPOCHS_ROUTER,
        )
        mark_stage_completed(router_stage)
        load_best_checkpoint(router_stage, red_model, strict=False)
    else:
        load_best_checkpoint(router_stage, red_model, strict=False)

    logger.info("Starting evaluation (mode=router)")
    evaluate(red_model, train_loader, "router", "Router Train")
    evaluate(red_model, val_loader, "router", "Router Val")
    evaluate(red_model, test_loader, "router", "Router Test")
    evaluate(red_model, all_loader, "router", "Router All")
    evaluate_per_range(red_model, expert_test_loaders, mode="router", tag="Router All per-range")

    # 3) 经典 MoE baseline
    moe_encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    moe_model = ModularAddModelWithMoE(moe_encoder, num_classes=len(label_vocab)).to(DEVICE)

    # 从 RED 的 base + experts checkpoint 继承参数
    load_best_checkpoint("base", moe_model, strict=False)
    for ei in range(5):
        load_best_checkpoint("expert_{}".format(ei), moe_model, strict=False)

    best_moe_ckpt = train_moe_router(moe_model, train_loader,
                                     epochs=MAX_EPOCHS_MOE,
                                     lr=5e-4,
                                     virtual_batch_size=VIRTUAL_BATCH_SIZE)
    if best_moe_ckpt is not None and best_moe_ckpt.exists():
        state = torch.load(best_moe_ckpt, map_location=DEVICE)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        moe_model.load_state_dict(sd, strict=False)
        logger.info("=> loaded best MoE checkpoint from {}".format(best_moe_ckpt))

    evaluate(moe_model, all_loader, mode="moe", desc="MoE All")
    evaluate(moe_model, train_loader, mode="moe", desc="MoE Train")
    evaluate(moe_model, val_loader, mode="moe", desc="MoE Val")
    evaluate(moe_model, test_loader, mode="moe", desc="MoE Test")
    evaluate_per_range(moe_model, expert_test_loaders, mode="moe", tag="MoE All per-range")


if __name__ == "__main__":
    main()
