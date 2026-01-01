# -*- coding: utf-8 -*-
"""
demo_v13.py
-----------
Task 2: Mixed Arithmetic (Addition + Subtraction, Multi-step)
- 统一脚本：同时包含 RED (residual experts + router) 和经典 MoE baseline。
- 数据：2~4 步加减混合表达式，结果仍限制在 [2, 100]，并映射到 5 个区间域。
- 针对 torch >= 2.0 / 尤其是 torch 2.5 做了兼容：
    * 优先启用 CUDA SDPA Flash-Attention 内核
    * 模型加载优先使用 float16
    * 训练与评估使用 autocast 以匹配 Flash-Attention 要求

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
import torch.nn.functional as F  # 新增
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from packaging import version as _pkg_version
except ImportError:
    _pkg_version = None

# ========================
# 全局配置
# ========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float32

TOTAL_SAMPLES = 80000
DEFAULT_SEED = 42

BASE_RATIO = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64

MAX_EPOCHS_BASE = 40
MAX_EPOCHS_EXPERT = 60
MAX_EPOCHS_ROUTER = 8
MAX_EPOCHS_MOE = 3

PATIENCE = 3
PATIENCE_MOE = 3

SAVE_DIR = "checkpoints_v13"
LOG_FILE = Path(SAVE_DIR) / "demo_v13.log"
MODEL_PATH = r"c:/temp/lunwen/Qwen3-0.6B"

STAGE_STATUS_FILE = Path(SAVE_DIR) / "stage_status_v13.json"
SAMPLES_CACHE_FILE = Path(SAVE_DIR) / "demo_v13_samples_mixed.json"
BALANCED_TRAIN_CACHE_FILE = Path(SAVE_DIR) / "demo_v13_train_balanced.json"
BALANCED_VAL_CACHE_FILE = Path(SAVE_DIR) / "demo_v13_val_balanced.json"
BALANCED_TEST_CACHE_FILE = Path(SAVE_DIR) / "demo_v13_test_balanced.json"


DOMAIN_ORDER = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]
# 用于域内标签均衡的范围（按需求限制到 2-99）
DOMAIN_LABEL_RANGES = {
    "range_2_20": list(range(2, 21)),
    "range_21_40": list(range(21, 41)),
    "range_41_60": list(range(41, 61)),
    "range_61_80": list(range(61, 81)),
    "range_81_100": list(range(81, 100)),  # 仅保留到 99
}


# ========================
# 日志 & 随机种子
# ========================

def setup_logger():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo_v13")
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
# 尝试启用 Flash-Attention (SDPA)
# ========================

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


def save_split_to_cache(samples: List[Dict], label_vocab: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "label_vocab": label_vocab}, f, ensure_ascii=False)
    logger.info("=> cached split to {} (n={})".format(path, len(samples)))


def load_split_from_cache(path: Path):
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples"), data.get("label_vocab")
    return data, None


def load_balanced_splits_from_cache():
    paths = [BALANCED_TRAIN_CACHE_FILE, BALANCED_VAL_CACHE_FILE, BALANCED_TEST_CACHE_FILE]
    if not all(p.exists() for p in paths):
        return None
    train_samples, train_vocab = load_split_from_cache(BALANCED_TRAIN_CACHE_FILE)
    val_samples, val_vocab = load_split_from_cache(BALANCED_VAL_CACHE_FILE)
    test_samples, test_vocab = load_split_from_cache(BALANCED_TEST_CACHE_FILE)
    cached_vocab = train_vocab or val_vocab or test_vocab
    for arr in (train_samples, val_samples, test_samples):
        for s in arr:
            if not isinstance(s.get("answer"), str):
                s["answer"] = str(s.get("answer"))
    logger.info("=> loaded balanced train/val/test splits from cache")
    return train_samples, val_samples, test_samples, cached_vocab


def save_balanced_splits_to_cache(train_samples: List[Dict],
                                  val_samples: List[Dict],
                                  test_samples: List[Dict],
                                  label_vocab: List[str]):
    save_split_to_cache(train_samples, label_vocab, BALANCED_TRAIN_CACHE_FILE)
    save_split_to_cache(val_samples, label_vocab, BALANCED_VAL_CACHE_FILE)
    save_split_to_cache(test_samples, label_vocab, BALANCED_TEST_CACHE_FILE)


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


def balance_labels(samples: List[Dict], allowed_labels: List[str], seed: int, tag: str):
    """下采样得到标签均衡的集合，仅保留 allowed_labels 中的标签。"""
    for s in samples:
        if not isinstance(s.get("answer"), str):
            s["answer"] = str(s.get("answer"))
    buckets = {lbl: [] for lbl in allowed_labels}
    for s in samples:
        ans = s.get("answer")
        if ans in buckets:
            buckets[ans].append(s)
    missing = [lbl for lbl, items in buckets.items() if not items]
    if missing:
        logger.warning("=> [{}] label balance skipped, missing labels: {}".format(tag, ",".join(missing)))
        return samples

    rand = random.Random(seed)
    target = min(len(items) for items in buckets.values())
    balanced = []
    for items in buckets.values():
        rand.shuffle(items)
        balanced.extend(items[:target])
    rand.shuffle(balanced)
    logger.info("=> [{}] label balanced to {} samples (per-label={})".format(tag, len(balanced), target))
    return balanced


def sample_label_balanced_subset(samples: List[Dict], allowed_labels: List[str], count: int, seed: int, tag: str):
    """在已知样本中下采样，尽量保持标签均衡。"""
    for s in samples:
        if not isinstance(s.get("answer"), str):
            s["answer"] = str(s.get("answer"))
    buckets = {lbl: [] for lbl in allowed_labels}
    for s in samples:
        ans = s.get("answer")
        if ans in buckets:
            buckets[ans].append(s)
    missing = [lbl for lbl, items in buckets.items() if not items]
    if missing:
        logger.warning("=> [{}] label-balanced subset skipped (missing labels: {})".format(tag, ",".join(missing)))
        return samples[:count]

    rand = random.Random(seed)
    for items in buckets.values():
        rand.shuffle(items)

    per_label = count // len(allowed_labels)
    subset = []
    remainder = count % len(allowed_labels)
    for idx, lbl in enumerate(allowed_labels):
        items = buckets[lbl]
        take = min(per_label + (1 if idx < remainder else 0), len(items))
        subset.extend(items[:take])
    rand.shuffle(subset)
    logger.info("=> [{}] label-balanced subset size={} (target per-label≈{})".format(tag, len(subset), per_label))
    return subset[:count]


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
        ans = s["answer"]
        if not isinstance(ans, str):
            ans = str(ans)
            s["answer"] = ans
        if ans not in self.label_to_id:
            raise KeyError("Answer {} not in label_to_id (size={})".format(ans, len(self.label_to_id)))
        label = self.label_to_id[ans]
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
    def __init__(self, model_path, max_len=64, trainable_last_n_layers=3):
        super().__init__()

        model_kwargs = {
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = AMP_DTYPE

        try:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            model_kwargs.pop("torch_dtype", None)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)

        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        # 先全部冻结，再解锁最后 n 层
        for p in self.model.parameters():
            p.requires_grad = False
        self.unfreeze_last_layers(trainable_last_n_layers)

        if torch.cuda.is_available():
            try:
                self.model = self.model.to(dtype=AMP_DTYPE)
            except Exception as e:
                logger.warning("=> Failed to cast encoder to {}: {}".format(AMP_DTYPE, e))

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
        h = out.last_hidden_state  # (B, T, H)
        # last-token pooling：取最后一个非 pad token 的 hidden state
        lengths = attention_mask.sum(dim=1)              # (B,)
        last_idx = (lengths - 1).clamp(min=0)            # 防止负数
        batch_idx = torch.arange(h.size(0), device=h.device)
        pooled = h[batch_idx, last_idx]                  # (B, H)
        return pooled



# ========================
# 模型定义：RED (residual experts + router)
# ========================

class ModularAddModelWithRouter(nn.Module):
    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size

        # v1 风格：更强一点的 base head
        self.base_head = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * H, num_classes),
        )

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
    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size

        # 保持和 RED 一致的 base_head 结构
        self.base_head = nn.Sequential(
            nn.Linear(H, 4 * H),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * H, num_classes),
        )

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


    def forward(self,
                input_ids,
                attention_mask=None,
                mode=None,
                return_gate: bool = False):
        """
        和 RED 模型保持相同的调用签名：
          - mode 参数只是为了兼容 evaluate/train 时的调用，当前实现里可以忽略
          - 默认只返回 logits；若 return_gate=True，则返回 (logits, gate_probs)
        """
        # 编码得到句向量
        hidden = self.encoder(input_ids, attention_mask)  # (B, H)
        target_dtype = None
        if hasattr(self.router, "weight"):
            target_dtype = self.router.weight.dtype
        elif hasattr(self, "base_head") and isinstance(self.base_head, nn.Sequential):
            target_dtype = self.base_head[0].weight.dtype
        if target_dtype is not None and hidden.dtype != target_dtype:
            hidden = hidden.to(target_dtype)

        # router 得到每个 expert 的权重
        gate_logits = self.router(hidden)                 # (B, E)
        gate_probs = torch.softmax(gate_logits, dim=-1)   # (B, E)

        # 每个 expert 输出一份 logits
        expert_logits_list = []
        for expert in self.experts:
            expert_logits_list.append(expert(hidden))     # 每个是 (B, C)

        # 堆叠后按 router 权重加权求和
        # expert_logits: (B, E, C)
        expert_logits = torch.stack(expert_logits_list, dim=1)
        # gate_probs: (B, E, 1)
        gate_probs_expanded = gate_probs.unsqueeze(-1)
        # 加权求和得到最终 logits: (B, C)
        logits = torch.sum(gate_probs_expanded * expert_logits, dim=1)

        if return_gate:
            return logits, gate_probs

        return logits




# ========================
# Checkpoint 辅助函数
# ========================

def _collect_stage_state_dict(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in trainable}
    return filtered if filtered else state


def save_checkpoint(model, stage, epoch=None):
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    suffix = "_ep{}".format(epoch) if epoch is not None else ""
    path = Path(SAVE_DIR) / "demo_v13_{}{}.pt".format(stage, suffix)
    torch.save({"state_dict": _collect_stage_state_dict(model),
                "stage": stage, "epoch": epoch}, path)
    logger.info("=> checkpoint saved: {}".format(path))
    return path


def get_best_checkpoint(stage_name):
    matches = sorted(Path(SAVE_DIR).glob("demo_v13_{}_best*.pt".format(stage_name)))
    if not matches:
        return None
    non_slim = [m for m in matches if ".slim" not in m.name]
    if non_slim:
        return non_slim[-1]
    return matches[-1]


def load_best_checkpoint(stage_name, model, strict=False):
    path = get_best_checkpoint(stage_name)
    if not path or not path.exists():
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


def _train_loop(model, loader, optim, criterion, train_step_fn, desc, accum_steps, max_norm=None):
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

        # 新增：梯度裁剪
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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

def train_base(model, loader, val_loader, epochs=6, lr=5e-5, virtual_batch_size=VIRTUAL_BATCH_SIZE):
    for exp in model.experts:
        for p in exp.parameters():
            p.requires_grad = False
    for p in model.router.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = True

    head_params = list(model.base_head.parameters())
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]

    optim = torch.optim.AdamW(
        [
            {"params": head_params, "lr": 1e-4},
            {"params": enc_params, "lr": 2e-5},
        ],
        weight_decay=0.01,
    )

    ce = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def base_step(batch):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(ids, mask, mode="base")
        else:
            logits = model(ids, mask, mode="base")
        return logits, label

    loss, acc, val_loss = 0.0, 0.0, float("inf")
    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=base_step,
            desc="[Base] epoch {}/{}".format(ep, epochs),
            accum_steps=accum_steps,
            max_norm=1.0,
        )
        val_loss, val_acc, val_f1 = evaluate_with_loss(
            model, val_loader, mode="base", desc="[Base-VAL] epoch {}/{}".format(ep, epochs)
        )
        logger.info(
            "=> Base epoch {} train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f} val_f1={:.4f}".format(
                ep, loss, acc, val_loss, val_acc, val_f1
            )
        )
    return loss, acc, val_loss





def train_expert(model, loader, val_loader, ei, epochs=5, lr=5e-4, virtual_batch_size=VIRTUAL_BATCH_SIZE):
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
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(ids, mask, mode="all_fixed", expert_mask=expert_mask)
        else:
            logits = model(ids, mask, mode="all_fixed", expert_mask=expert_mask)
        return logits, label

    loss, acc, val_loss = 0.0, 0.0, float("inf")
    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=expert_step,
            desc="[Expert {}] epoch {}/{}".format(ei, ep, epochs),
            accum_steps=accum_steps,max_norm=1.0,)
        val_loss, val_acc = evaluate_single_expert(
            model, val_loader, expert_idx=ei, desc="[Expert {} VAL] epoch {}/{}".format(ei, ep, epochs)
        )
        logger.info("=> Expert {} epoch {} train_loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}".format(
            ei, ep, loss, acc, val_loss, val_acc))
    return loss, acc, val_loss


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

    optim = torch.optim.Adam(
        model.router.parameters(),
        lr=lr,
        weight_decay=0.01,  # 新增
    )
    ce = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def router_step(batch):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        domains = batch["domain"]
        domain_ids = torch.tensor([domain_to_id[d] for d in domains], device=DEVICE)
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                h = model.encoder(ids, mask)
                h = F.dropout(h, p=0.1, training=model.training)
                gate_logits = model.router(h)
        else:
            h = model.encoder(ids, mask)
            gate_logits = model.router(h)
        return gate_logits, domain_ids

    loss, acc = 0.0, 0.0
    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=router_step,
            desc="[Router] epoch {}/{}".format(ep, epochs),
            accum_steps=accum_steps,max_norm=1.0,)
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

            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    logits = model(ids, mask, mode="moe")
            else:
                logits = model(ids, mask, mode="moe")

            loss = ce(logits, label) / float(accum_steps)
            loss.backward()
            grad_step += 1
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.0)

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
            best_path = Path(SAVE_DIR) / "demo_v13_moe_best_ep{}.pt".format(epoch)
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
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(ids, mask, mode=mode)
        else:
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
def evaluate_with_loss(model, loader, mode, desc):
    """计算 loss/acc/f1（用于每轮校验）。"""
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(ids, mask, mode=mode)
                loss = ce(logits, label)
        else:
            logits = model(ids, mask, mode=mode)
            loss = ce(logits, label)
        pred = logits.argmax(-1)
        batch_size = label.size(0)
        total_loss += loss.item() * float(batch_size)
        correct += (pred == label).sum().item()
        total += batch_size
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
    avg_loss = total_loss / float(total if total > 0 else 1)

    logger.info("[EvalLoss] {} mode={} loss={:.4f} acc={:.4f} macro_f1={:.4f} n={}".format(
        desc, mode, avg_loss, acc, macro_f1, total))
    return avg_loss, acc, macro_f1


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
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    logits = model(ids, mask, mode=mode)
            else:
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
def evaluate_single_expert(model, loader: DataLoader, expert_idx: int, desc: str):
    """
    Evaluate a single expert (plus base residual) with a fixed expert mask.
    Only expert_idx is active; others are masked out.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    expert_mask = torch.zeros(model.num_experts, device=DEVICE)
    expert_mask[expert_idx] = 1.0

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(ids, mask, mode="all_fixed", expert_mask=expert_mask)
                loss = ce(logits, labels)
        else:
            logits = model(ids, mask, mode="all_fixed", expert_mask=expert_mask)
            loss = ce(logits, labels)
        preds = logits.argmax(dim=-1)

        batch_size = labels.size(0)
        total_loss += loss.item() * float(batch_size)
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        avg_loss = total_loss / float(total_count if total_count > 0 else 1)
        avg_acc = total_correct / float(total_count if total_count > 0 else 1)
        pbar.set_postfix(loss="{:.4f}".format(avg_loss), acc="{:.4f}".format(avg_acc))

    avg_loss = total_loss / float(total_count if total_count > 0 else 1)
    avg_acc = total_correct / float(total_count if total_count > 0 else 1)
    logger.info("[Expert {} Eval] {} loss={:.4f} acc={:.4f} n={}".format(
        expert_idx, desc, avg_loss, avg_acc, total_count))
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_router_domain(model, loader: DataLoader, domain_to_id: Dict[str, int], desc: str):
    """Evaluate router head on domain labels (for early stopping)."""
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        domains = batch["domain"]
        domain_ids = torch.tensor([domain_to_id[d] for d in domains], device=DEVICE)
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                h = model.encoder(ids, mask)
                logits = model.router(h)
                loss = ce(logits, domain_ids)
        else:
            h = model.encoder(ids, mask)
            logits = model.router(h)
            loss = ce(logits, domain_ids)
        pred = logits.argmax(-1)
        batch_size = domain_ids.size(0)
        total_loss += loss.item() * float(batch_size)
        correct += (pred == domain_ids).sum().item()
        total += batch_size
        pbar.set_postfix(loss="{:.4f}".format(total_loss / float(total if total > 0 else 1)),
                         acc="{:.4f}".format(correct / float(total if total > 0 else 1)))
    avg_loss = total_loss / float(total if total > 0 else 1)
    avg_acc = correct / float(total if total > 0 else 1)
    logger.info("[RouterEval] {} loss={:.4f} acc={:.4f} n={}".format(desc, avg_loss, avg_acc, total))
    return avg_loss, avg_acc


# ========================
# 主流程
# ========================

def main():
    set_seed(DEFAULT_SEED)

    # 1) 数据
    samples, label_vocab = load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED)
    labels_2_99 = [str(i) for i in range(2, 100)]

    cached_balanced = load_balanced_splits_from_cache()
    if cached_balanced is not None:
        train_samples, val_samples, test_samples, cached_vocab = cached_balanced
        if cached_vocab:
            label_vocab = cached_vocab
        logger.info("=> using cached balanced splits for train/val/test")
    else:
        train_samples, val_samples, test_samples = stratified_split_samples(
            samples, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=DEFAULT_SEED + 1
        )
        # 标签均衡（2-99）仅对 train/val
        train_samples = balance_labels(train_samples, labels_2_99, seed=DEFAULT_SEED + 100, tag="train (labels 2-99)")
        val_samples = balance_labels(val_samples, labels_2_99, seed=DEFAULT_SEED + 101, tag="val (labels 2-99)")
        save_balanced_splits_to_cache(train_samples, val_samples, test_samples, label_vocab)

    train_size = len(train_samples)
    val_size = len(val_samples)
    test_size = len(test_samples)

    logger.info("=> train_size={} val_size={} test_size={}".format(train_size, val_size, test_size))

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
    base_subset = sample_label_balanced_subset(
        train_samples, labels_2_99, train_base_count, seed=DEFAULT_SEED + 3, tag="base subset (labels 2-99)"
    )
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
    expert_val = [[] for _ in range(5)]
    expert_test = [[] for _ in range(5)]
    expert_all = [[] for _ in range(5)]

    for s in train_samples:
        expert_train[domain_to_expert[s["domain"]]].append(s)
    for s in val_samples:
        expert_val[domain_to_expert[s["domain"]]].append(s)
    for s in test_samples:
        expert_test[domain_to_expert[s["domain"]]].append(s)
    for s in samples:
        expert_all[domain_to_expert[s["domain"]]].append(s)

    domain_by_idx = {v: k for k, v in domain_to_expert.items()}
    for i in range(len(expert_train)):
        domain_name = domain_by_idx.get(i, f"domain_{i}")
        domain_labels = [str(x) for x in DOMAIN_LABEL_RANGES.get(domain_name, [])]
        expert_train[i] = balance_labels(
            expert_train[i], domain_labels, seed=DEFAULT_SEED + 200 + i, tag="expert_train[{}]-{}".format(i, domain_name)
        )
        expert_val[i] = balance_labels(
            expert_val[i], domain_labels, seed=DEFAULT_SEED + 210 + i, tag="expert_val[{}]-{}".format(i, domain_name)
        )
        expert_test[i] = balance_labels(
            expert_test[i], domain_labels, seed=DEFAULT_SEED + 220 + i, tag="expert_test[{}]-{}".format(i, domain_name)
        )

    expert_train_loaders = [
        make_loader(QADataset(sub, tokenizer, label_vocab), shuffle=True, seed_offset=10 + i)
        for i, sub in enumerate(expert_train)
    ]
    expert_val_loaders = [
        make_loader(QADataset(sub, tokenizer, label_vocab), shuffle=False, seed_offset=15 + i)
        for i, sub in enumerate(expert_val)
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
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=3).to(DEVICE)
    red_model = ModularAddModelWithRouter(encoder, num_classes=len(label_vocab)).to(DEVICE)

    # 2.1 base stage
    base_stage = "base"
    if not is_stage_completed(base_stage):
        load_best_checkpoint(base_stage, red_model, strict=False)
        best_val = float("inf")
        patience_cnt = 0
        best_path = None
        for epoch in range(1, MAX_EPOCHS_BASE + 1):
            train_loss, train_acc, val_loss = train_base(
                red_model, base_loader, val_loader, epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE
            )
            if val_loss < best_val:
                if best_path and best_path.exists():
                    best_path.unlink()
                best_val = val_loss
                best_path = save_checkpoint(red_model, "{}_best".format(base_stage), epoch=epoch)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    logger.info("=> Early stopping {} at epoch {} by val_loss (patience={})".format(
                        base_stage, epoch, PATIENCE))
                    break
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
            best_val = float("inf")
            patience_cnt = 0
            best_path = None
            for epoch in range(1, MAX_EPOCHS_EXPERT + 1):
                train_loss, train_acc, val_loss = train_expert(
                    red_model, expert_train_loaders[ei], expert_val_loaders[ei], ei, epochs=1,
                    virtual_batch_size=VIRTUAL_BATCH_SIZE)
                if val_loss < best_val:
                    if best_path and best_path.exists():
                        best_path.unlink()
                    best_val = val_loss
                    best_path = save_checkpoint(red_model, "{}_best".format(expert_stage), epoch=epoch)
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        logger.info("=> Early stopping {} at epoch {} by val_loss (patience={})".format(
                            expert_stage, epoch, PATIENCE))
                        break
            mark_stage_completed(expert_stage)
            load_best_checkpoint(expert_stage, red_model, strict=False)
        else:
            load_best_checkpoint(expert_stage, red_model, strict=False)

    logger.info("Starting evaluation (mode=all_fixed)")
    # evaluate(red_model, train_loader, "all_fixed", "AllExperts Train")
    # evaluate(red_model, val_loader, "all_fixed", "AllExperts Val")
    # evaluate(red_model, test_loader, "all_fixed", "AllExperts Test")
    # evaluate(red_model, all_loader, "all_fixed", "AllExperts All")
    # evaluate_per_range(red_model, expert_test_loaders, mode="all_fixed", tag="AllExperts All per-range")

    # # 单独测试 expert 0（只加载 base 与 expert_0 的 checkpoint）
    # logger.info("Starting single expert[0] evaluation (base + expert_0 checkpoints only)")
    # expert0_encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=3).to(DEVICE)
    # expert0_model = ModularAddModelWithRouter(expert0_encoder, num_classes=len(label_vocab)).to(DEVICE)
    # load_best_checkpoint("base", expert0_model, strict=False)
    # load_best_checkpoint("expert_0", expert0_model, strict=False)
    # evaluate_single_expert(expert0_model, expert_val_loaders[0], expert_idx=0,
    #                        desc="Expert0 Val (range_2_20)")
    # evaluate_single_expert(expert0_model, expert_test_loaders[0], expert_idx=0,
    #                        desc="Expert0 Test (range_2_20)")

    # 2.3 router (RED)
    router_stage = "router"
    if not is_stage_completed(router_stage):
        load_best_checkpoint(router_stage, red_model, strict=False)
        best_val = float("inf")
        patience_cnt = 0
        best_path = None
        for epoch in range(1, MAX_EPOCHS_ROUTER + 1):
            train_loss, train_acc = train_router(
                red_model, train_loader, domain_to_expert, epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE
            )
            val_loss, val_acc = evaluate_router_domain(
                red_model, val_loader, domain_to_expert, desc="[Router VAL] epoch {}/{}".format(epoch, MAX_EPOCHS_ROUTER)
            )
            if val_loss < best_val:
                if best_path and best_path.exists():
                    best_path.unlink()
                best_val = val_loss
                best_path = save_checkpoint(red_model, "{}_best".format(router_stage), epoch=epoch)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    logger.info("=> Early stopping {} at epoch {} by val_loss (patience={})".format(
                        router_stage, epoch, PATIENCE))
                    break
        mark_stage_completed(router_stage)
        load_best_checkpoint(router_stage, red_model, strict=False)
    else:
        load_best_checkpoint(router_stage, red_model, strict=False)

    logger.info("Starting evaluation (mode=router)")
    # evaluate(red_model, train_loader, "router", "Router Train")
    # evaluate(red_model, val_loader, "router", "Router Val")
    # evaluate(red_model, test_loader, "router", "Router Test")
    # evaluate(red_model, all_loader, "router", "Router All")
    # evaluate_per_range(red_model, expert_test_loaders, mode="router", tag="Router All per-range")

    # 3) 经典 MoE baseline
    moe_encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=3).to(DEVICE)
    moe_model = ModularAddModelWithMoE(moe_encoder, num_classes=len(label_vocab)).to(DEVICE)

    # 从 RED 的 base + experts checkpoint 继承参数
    load_best_checkpoint("base", moe_model, strict=False)
    for ei in range(5):
        load_best_checkpoint("expert_{}".format(ei), moe_model, strict=False)

    # best_moe_ckpt = train_moe_router(moe_model, train_loader,
    #                                  epochs=MAX_EPOCHS_MOE,
    #                                  lr=5e-4,
    #                                  virtual_batch_size=VIRTUAL_BATCH_SIZE)
    # if best_moe_ckpt is not None and best_moe_ckpt.exists():
    #     state = torch.load(best_moe_ckpt, map_location=DEVICE)
    #     sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    #     moe_model.load_state_dict(sd, strict=False)
    #     logger.info("=> loaded best MoE checkpoint from {}".format(best_moe_ckpt))
    # state = torch.load(best_moe_ckpt, map_location=DEVICE)
    # sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    # moe_model.load_state_dict(sd, strict=False)
    load_best_checkpoint("moe", moe_model, strict=False)
    # evaluate(moe_model, all_loader, mode="moe", desc="MoE All")
    # evaluate(moe_model, train_loader, mode="moe", desc="MoE Train")
    # evaluate(moe_model, val_loader, mode="moe", desc="MoE Val")
    evaluate(moe_model, test_loader, mode="moe", desc="MoE Test")
    evaluate_per_range(moe_model, expert_test_loaders, mode="moe", tag="MoE All per-range")


if __name__ == "__main__":
    main()
