# -*- coding: utf-8 -*-
"""
demo_mnist_v11_label5.py  (v11-style, label-group domains, full metrics + legacy per-domain acc dicts)
---------------------------------------------------------------------------------------------------
Domain grouping (fixed 5 domains):
  D0: labels {0,1}
  D1: labels {2,3}
  D2: labels {4,5}
  D3: labels {6,7}
  D4: labels {8,9}

RED stages (v11 style):
  1) base     : train encoder + base_head on a small base_subset (balanced by label)
  2) expert_i : train only expert_i on its domain subset
  3) router   : train only router with CE(router(h), domain_id)
  4) eval     : evaluate base / all_fixed / router

MoE baseline (v11 style):
  - moe : train only router with CE(moe_logits, y), experts shared

Metrics (added for each split and per-domain on TEST):
  - loss (CrossEntropy), acc
  - macro_precision / macro_recall / macro_f1 (classes with support>0 only)
  - weighted_f1
  - per_class_precision / per_class_recall / per_class_f1 / support (10 classes)
  - confusion_matrix (10x10, saved to json)

Critical requirement (user): keep legacy "per-domain acc dict" lines (v11 logs).
This script prints BOTH:
  - legacy per-domain acc dicts
  - richer per-domain metrics (loss/acc/macro_f1 in logs, full metrics in json)
"""

import os
import json
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import datasets
from torchvision.transforms import functional as TF

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------
# Defaults (v11-style)
# -------------------------
DEFAULT_SEED = 42
DEFAULT_NUM_DOMAINS = 5
DEFAULT_HIDDEN_DIM = 128

DEFAULT_VAL_RATIO = 0.1
DEFAULT_BASE_RATIO = 0.10

DEFAULT_BATCH_SIZE = 128
DEFAULT_VIRTUAL_BATCH_SIZE = 128

DEFAULT_MAX_EPOCHS_BASE = 2
DEFAULT_MAX_EPOCHS_EXPERT = 30
DEFAULT_MAX_EPOCHS_ROUTER = 30
DEFAULT_MAX_EPOCHS_MOE = 30

DEFAULT_PATIENCE = 3

DEFAULT_LR_BASE = 1e-3
DEFAULT_LR_EXPERT = 3e-3
DEFAULT_LR_ROUTER = 1e-3
DEFAULT_LR_MOE = 1e-3

DEFAULT_WEIGHT_DECAY = 1e-4

NUM_CLASSES = 10


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logger(save_dir: str, name: str = "mnist_v11_label5") -> logging.Logger:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    log_file = Path(save_dir) / (name + ".log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _get_accum_steps(loader: DataLoader, virtual_batch_size: int) -> int:
    unit = getattr(loader, "batch_size", 1) or 1
    return max(1, int(virtual_batch_size) // int(unit))


def _tqdm(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, ncols=120)


# -------------------------
# Stage status (v11-style)
# -------------------------
def _stage_status_path(save_dir: str) -> Path:
    return Path(save_dir) / "stage_status.json"


def load_stage_status(save_dir: str) -> Dict[str, bool]:
    p = _stage_status_path(save_dir)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_stage_status(save_dir: str, status: Dict[str, bool]) -> None:
    p = _stage_status_path(save_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def is_stage_completed(save_dir: str, stage_name: str) -> bool:
    status = load_stage_status(save_dir)
    return bool(status.get(stage_name, False))


def mark_stage_completed(save_dir: str, stage_name: str) -> None:
    status = load_stage_status(save_dir)
    status[stage_name] = True
    save_stage_status(save_dir, status)


# -------------------------
# Checkpoints (v11-style)
# -------------------------
def save_checkpoint(
    save_dir: str,
    prefix: str,
    model: nn.Module,
    stage: str,
    epoch: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Path:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    suffix = ("_ep" + str(epoch)) if epoch is not None else ""
    path = Path(save_dir) / (prefix + "_" + stage + suffix + ".pt")
    torch.save({"state_dict": model.state_dict(), "stage": stage, "epoch": epoch}, path)
    if logger is not None:
        logger.info("=> checkpoint saved: {}".format(path))
    return path


def get_best_checkpoint(save_dir: str, prefix: str, stage_name: str) -> Optional[Path]:
    matches = sorted(Path(save_dir).glob(prefix + "_" + stage_name + "_best*.pt"))
    if not matches:
        return None
    non_slim = [m for m in matches if ".slim" not in m.name]
    if non_slim:
        return non_slim[-1]
    return matches[-1]


def load_best_checkpoint(
    save_dir: str,
    prefix: str,
    stage_name: str,
    model: nn.Module,
    device: str,
    strict: bool = False,
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    path = get_best_checkpoint(save_dir, prefix, stage_name)
    if path is None or (not path.exists()):
        return None
    try:
        state = torch.load(path, map_location=device)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if logger is not None:
            logger.info("=> loaded best checkpoint for {}: {} (missing={} unexpected={})".format(
                stage_name, path, len(missing), len(unexpected)
            ))
        return path
    except RuntimeError as e:
        if logger is not None:
            logger.warning("=> failed to load checkpoint {}: {}".format(path, e))
        return None


def attempt_resume(save_dir: str, prefix: str, stage_name: str, model: nn.Module, device: str, logger: logging.Logger) -> bool:
    path = load_best_checkpoint(save_dir, prefix, stage_name, model, device=device, strict=False, logger=logger)
    if path is None:
        return False
    logger.info("=> resumed {} from {}".format(stage_name, path))
    return True


def run_stage_with_early_stop(
    save_dir: str,
    prefix: str,
    stage_name: str,
    model: nn.Module,
    max_epochs: int,
    patience: int,
    train_step_fn,
    device: str,
    logger: logging.Logger,
) -> Tuple[float, int, Optional[Path]]:
    best_loss = float("inf")
    patience_cnt = 0
    best_epoch = 0
    best_path: Optional[Path] = None

    for epoch in range(1, max_epochs + 1):
        loss, acc = train_step_fn(epoch)
        if loss < best_loss:
            if best_path is not None and best_path.exists():
                try:
                    best_path.unlink()
                except Exception:
                    pass
            best_loss = float(loss)
            best_epoch = int(epoch)
            patience_cnt = 0
            best_path = save_checkpoint(save_dir, prefix, model, stage_name + "_best", epoch=epoch, logger=logger)
        else:
            patience_cnt += 1

        logger.info("=> {} epoch {} loss={:.6f} acc={:.6f} best_loss={:.6f} patience={}/{}".format(
            stage_name, epoch, float(loss), float(acc), float(best_loss), int(patience_cnt), int(patience)
        ))
        if patience_cnt >= patience:
            logger.info("=> Early stopping {} at epoch {} (patience={})".format(stage_name, epoch, patience))
            break

    return best_loss, best_epoch, best_path


# -------------------------
# Label->Domain mapping (fixed 5)
# -------------------------
def label_to_domain_fixed5(label: int) -> int:
    y = int(label)
    if y < 0 or y > 9:
        raise ValueError("MNIST label must be 0..9, got {}".format(y))
    return int(y // 2)


# -------------------------
# Splits cache
# -------------------------
@dataclass
class SplitPack:
    train_pairs: List[Tuple[int, int]]  # (idx, domain)
    val_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]


def build_label5_domain_pairs(mnist_ds, seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    for idx in range(len(mnist_ds)):
        _img, y = mnist_ds[idx]
        d = label_to_domain_fixed5(int(y))
        pairs.append((int(idx), int(d)))
    rng.shuffle(pairs)
    return pairs


def split_train_val_by_domain(pairs: List[Tuple[int, int]], val_ratio: float, seed: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    rng = random.Random(seed)
    dom_to_pairs: Dict[int, List[Tuple[int, int]]] = {}
    for idx, d in pairs:
        dom_to_pairs.setdefault(int(d), []).append((int(idx), int(d)))

    train_out: List[Tuple[int, int]] = []
    val_out: List[Tuple[int, int]] = []
    for d, lst in dom_to_pairs.items():
        rng.shuffle(lst)
        n_val = int(len(lst) * float(val_ratio))
        if n_val >= len(lst):
            n_val = (len(lst) - 1) if len(lst) > 1 else 0
        val_out.extend(lst[:n_val])
        train_out.extend(lst[n_val:])

    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


def build_or_load_splits(save_dir: str, train_raw, test_raw, val_ratio: float, seed: int, logger: logging.Logger) -> SplitPack:
    split_path = Path(save_dir) / "splits.json"
    if split_path.exists():
        try:
            data = load_json(str(split_path))
            meta = data.get("meta", {})
            mode = str(meta.get("domain_mode", "")) if isinstance(meta, dict) else ""
            cached_k = int(meta.get("num_domains", -1)) if isinstance(meta, dict) else -1

            train_pairs = [(int(a), int(b)) for a, b in data["train_pairs"]]
            val_pairs = [(int(a), int(b)) for a, b in data["val_pairs"]]
            test_pairs = [(int(a), int(b)) for a, b in data["test_pairs"]]

            doms = [d for _i, d in train_pairs] + [d for _i, d in val_pairs] + [d for _i, d in test_pairs]
            d_min = min(doms) if doms else 0
            d_max = max(doms) if doms else -1

            if mode != "label_pair_01_23_45_67_89" or cached_k != 5 or d_min < 0 or d_max >= 5:
                logger.warning("=> splits.json incompatible. Rebuilding. (mode={}, cachedK={}, dom=[{},{}])".format(
                    mode, cached_k, d_min, d_max
                ))
                raise ValueError("splits mismatch")

            logger.info("=> loaded splits from {} (train={}, val={}, test={})".format(
                split_path, len(train_pairs), len(val_pairs), len(test_pairs)
            ))
            return SplitPack(train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs)
        except Exception as e:
            logger.warning("=> failed to use cached splits.json ({}). Rebuilding.".format(e))

    train_pairs_all = build_label5_domain_pairs(train_raw, seed=seed)
    train_pairs, val_pairs = split_train_val_by_domain(train_pairs_all, val_ratio=val_ratio, seed=seed + 1)
    test_pairs = build_label5_domain_pairs(test_raw, seed=seed + 2)

    save_json({
        "meta": {
            "seed": int(seed),
            "num_domains": 5,
            "val_ratio": float(val_ratio),
            "domain_mode": "label_pair_01_23_45_67_89",
            "domain_map": {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]},
        },
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
    }, str(split_path))

    logger.info("=> built splits and saved to {} (train={}, val={}, test={})".format(
        split_path, len(train_pairs), len(val_pairs), len(test_pairs)
    ))
    return SplitPack(train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs)


def build_base_subset_balanced_by_label(mnist_ds, train_pairs: List[Tuple[int, int]], base_ratio: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {c: [] for c in range(10)}
    for pos, (idx, _d) in enumerate(train_pairs):
        _img, y = mnist_ds[int(idx)]
        buckets[int(y)].append(int(pos))

    selected: List[int] = []
    for c in range(10):
        lst = buckets.get(c, [])
        if len(lst) == 0:
            continue
        rng.shuffle(lst)
        k = int(len(lst) * float(base_ratio))
        if k <= 0:
            k = 1
        selected.extend(lst[:k])

    rng.shuffle(selected)
    return selected


# -------------------------
# Dataset
# -------------------------
class MNISTLabel5DomainDataset(Dataset):
    def __init__(self, mnist_ds, pairs: List[Tuple[int, int]]):
        super().__init__()
        self.mnist = mnist_ds
        self.pairs = pairs
        self.mean = 0.1307
        self.std = 0.3081

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        idx, domain = self.pairs[i]
        img, y = self.mnist[int(idx)]
        x = TF.to_tensor(img)
        x = (x - self.mean) / self.std
        return x, int(y), int(domain)


# -------------------------
# Model (RED + Router + MoE)
# -------------------------
class SmallCNNEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ModularMNISTModelV11(nn.Module):
    """
    v11-aligned:
      - base     : base_head(h)
      - all_fixed: base + 0.1 * sum(mask_i * expert_i(h))
      - router   : base + router_scale * sum(alpha_i * expert_i(h))
      - moe      : sum(alpha_i * expert_i(h))   (baseline)
    """
    def __init__(self, num_classes: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = SmallCNNEncoder(hidden_dim=hidden_dim)
        self.base_head = nn.Linear(hidden_dim, num_classes)
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_experts)])
        self.router = nn.Linear(hidden_dim, num_experts)
        self.num_experts = int(num_experts)
        self.router_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor, mode: str = "base", expert_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(x)
        base_logits = self.base_head(h)

        if mode == "base":
            return base_logits

        expert_logits = torch.stack([e(h) for e in self.experts], dim=1)  # [B,K,C]

        if mode == "all_fixed":
            if expert_mask is None:
                mask = torch.ones(self.num_experts, device=h.device, dtype=torch.float32)
            else:
                mask = expert_mask.to(h.device).to(torch.float32)
            mixed = (expert_logits * mask.view(1, -1, 1)).sum(dim=1)
            return base_logits + 0.1 * mixed

        if mode == "router":
            alpha = torch.softmax(self.router(h), dim=-1).unsqueeze(-1)
            mixed = (expert_logits * alpha).sum(dim=1)
            return base_logits + self.router_scale * mixed

        if mode == "moe":
            alpha = torch.softmax(self.router(h), dim=-1).unsqueeze(-1)
            mixed = (expert_logits * alpha).sum(dim=1)
            return mixed

        raise ValueError("Unknown mode: {}".format(mode))


def set_all_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = bool(flag)


def set_module_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(flag)


# -------------------------
# Training loop (one-epoch)
# -------------------------
def _train_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    train_step_fn,
    desc: str,
    accum_steps: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    iters = 0
    grad_step = 0

    optimizer.zero_grad()
    for batch in _tqdm(loader, desc=desc):
        logits, labels = train_step_fn(batch)
        loss = criterion(logits, labels) / float(accum_steps)
        loss.backward()
        grad_step += 1

        if grad_step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        pred = torch.argmax(logits, dim=-1)
        acc = (pred == labels).float().mean().item()

        total_loss += float(loss.item()) * float(accum_steps)
        total_acc += float(acc)
        iters += 1

    if grad_step % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    denom = float(iters if iters > 0 else 1)
    return total_loss / denom, total_acc / denom


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    try:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, generator=gen)
    except TypeError:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


# -------------------------
# Metrics (loss, acc, macro-F1, recall, etc.)
# -------------------------
@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    mode: str,
    device: str,
    num_classes: int,
    expert_mask: Optional[torch.Tensor] = None
) -> Dict:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_n = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for x, y, _d in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, mode=mode, expert_mask=expert_mask)

        loss = crit(logits, y)
        total_loss += float(loss.item())
        total_n += int(y.numel())

        pred = torch.argmax(logits, dim=-1)

        y_cpu = y.view(-1).detach().to(torch.long).cpu()
        p_cpu = pred.view(-1).detach().to(torch.long).cpu()
        idx = y_cpu * num_classes + p_cpu
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += binc.view(num_classes, num_classes)

    denom = float(total_n if total_n > 0 else 1)
    avg_loss = total_loss / denom

    tp = conf.diag().to(torch.float32)
    row_sum = conf.sum(dim=1).to(torch.float32)  # support (true)
    col_sum = conf.sum(dim=0).to(torch.float32)

    fp = col_sum - tp
    fn = row_sum - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    support = row_sum
    mask = support > 0
    if int(mask.sum().item()) > 0:
        macro_p = float(precision[mask].mean().item())
        macro_r = float(recall[mask].mean().item())
        macro_f1 = float(f1[mask].mean().item())
    else:
        macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0

    weighted_f1 = float((f1 * support).sum().item() / (support.sum().item() + eps))
    acc = float(tp.sum().item() / (support.sum().item() + eps))

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": [float(v) for v in precision.tolist()],
        "per_class_recall": [float(v) for v in recall.tolist()],
        "per_class_f1": [float(v) for v in f1.tolist()],
        "per_class_support": [int(v) for v in support.to(torch.long).tolist()],
        "confusion_matrix": conf.tolist(),
    }


@torch.no_grad()
def evaluate_per_domain_metrics(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: str,
    mode: str,
    num_domains: int,
    num_classes: int,
    expert_mask: Optional[torch.Tensor] = None,
    num_workers: int = 0
) -> Dict[int, Dict]:
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        raise ValueError("dataset must have .pairs for per-domain evaluation")

    out: Dict[int, Dict] = {}
    for d in range(int(num_domains)):
        indices = [i for i, (_idx, dom) in enumerate(pairs) if int(dom) == int(d)]
        if len(indices) == 0:
            out[int(d)] = {
                "loss": 0.0, "acc": 0.0,
                "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "per_class_precision": [0.0] * num_classes,
                "per_class_recall": [0.0] * num_classes,
                "per_class_f1": [0.0] * num_classes,
                "per_class_support": [0] * num_classes,
                "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
            }
            continue

        sub = Subset(dataset, indices)
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        out[int(d)] = evaluate_metrics(
            model=model, loader=loader, mode=mode, device=device, num_classes=num_classes, expert_mask=expert_mask
        )
    return out


# -------------------------
# One-epoch stage trainers (v11 style)
# -------------------------
def train_base_one_epoch(model: ModularMNISTModelV11, loader: DataLoader, device: str, lr: float, weight_decay: float, virtual_batch_size: int) -> Tuple[float, float]:
    set_all_requires_grad(model, False)
    set_module_requires_grad(model.encoder, True)
    set_module_requires_grad(model.base_head, True)
    model.router_scale.requires_grad = False
    model.to(device)

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def step(batch):
        x, y, _d = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x, mode="base")
        return logits, y

    return _train_loop(model, loader, opt, crit, step, desc="[Base] one-epoch", accum_steps=accum_steps)


def train_expert_one_epoch(model: ModularMNISTModelV11, loader: DataLoader, expert_id: int, device: str, lr: float, weight_decay: float, virtual_batch_size: int) -> Tuple[float, float]:
    set_all_requires_grad(model, False)
    set_module_requires_grad(model.experts[int(expert_id)], True)
    model.router_scale.requires_grad = False
    model.to(device)

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    expert_mask = torch.zeros(model.num_experts, dtype=torch.float32)
    expert_mask[int(expert_id)] = 1.0

    def step(batch):
        x, y, _d = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x, mode="all_fixed", expert_mask=expert_mask)
        return logits, y

    return _train_loop(model, loader, opt, crit, step, desc="[Expert {}] one-epoch".format(expert_id), accum_steps=accum_steps)


def train_router_domain_one_epoch(model: ModularMNISTModelV11, loader: DataLoader, device: str, lr: float, weight_decay: float, virtual_batch_size: int) -> Tuple[float, float]:
    set_all_requires_grad(model, False)
    set_module_requires_grad(model.router, True)
    model.router_scale.requires_grad = False
    model.to(device)

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def step(batch):
        x, _y, d = batch
        x = x.to(device)
        d = d.to(device)
        h = model.encoder(x)
        gate_logits = model.router(h)
        return gate_logits, d

    return _train_loop(model, loader, opt, crit, step, desc="[Router(domain CE)] one-epoch", accum_steps=accum_steps)


def train_moe_router_one_epoch(model: ModularMNISTModelV11, loader: DataLoader, device: str, lr: float, weight_decay: float, virtual_batch_size: int) -> Tuple[float, float]:
    set_all_requires_grad(model, False)
    set_module_requires_grad(model.router, True)
    model.router_scale.requires_grad = False
    model.to(device)

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    crit = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(loader, virtual_batch_size)

    def step(batch):
        x, y, _d = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x, mode="moe")
        return logits, y

    return _train_loop(model, loader, opt, crit, step, desc="[MoE(CE on y)] one-epoch", accum_steps=accum_steps)


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data_mnist")
    parser.add_argument("--save_dir", type=str, default="./ckpt_mnist_v11_label5")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--num_domains", type=int, default=DEFAULT_NUM_DOMAINS)  # must be 5
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)

    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--base_ratio", type=float, default=DEFAULT_BASE_RATIO)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--virtual_batch_size", type=int, default=DEFAULT_VIRTUAL_BATCH_SIZE)

    parser.add_argument("--lr_base", type=float, default=DEFAULT_LR_BASE)
    parser.add_argument("--lr_expert", type=float, default=DEFAULT_LR_EXPERT)
    parser.add_argument("--lr_router", type=float, default=DEFAULT_LR_ROUTER)
    parser.add_argument("--lr_moe", type=float, default=DEFAULT_LR_MOE)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    parser.add_argument("--max_epochs_base", type=int, default=DEFAULT_MAX_EPOCHS_BASE)
    parser.add_argument("--max_epochs_expert", type=int, default=DEFAULT_MAX_EPOCHS_EXPERT)
    parser.add_argument("--max_epochs_router", type=int, default=DEFAULT_MAX_EPOCHS_ROUTER)
    parser.add_argument("--max_epochs_moe", type=int, default=DEFAULT_MAX_EPOCHS_MOE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--run", type=str, choices=["red", "moe", "all"], default="all")

    args = parser.parse_args()

    if int(args.num_domains) != 5:
        raise ValueError("This script is fixed for 5 domains (0/1,2/3,4/5,6/7,8/9). Please set --num_domains 5.")

    ensure_dir(args.save_dir)
    logger = setup_logger(args.save_dir, name="mnist_v11_label5")
    logger.info("Args: {}".format(vars(args)))

    set_seed(int(args.seed))

    prefix = "mnist_v11_label5"
    device = str(args.device)
    k = 5

    # raw MNIST
    train_raw = datasets.MNIST(root=args.data_dir, train=True, download=True)
    test_raw = datasets.MNIST(root=args.data_dir, train=False, download=True)

    # splits
    splits = build_or_load_splits(
        save_dir=args.save_dir,
        train_raw=train_raw,
        test_raw=test_raw,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        logger=logger,
    )

    train_ds = MNISTLabel5DomainDataset(train_raw, splits.train_pairs)
    val_ds = MNISTLabel5DomainDataset(train_raw, splits.val_pairs)
    test_ds = MNISTLabel5DomainDataset(test_raw, splits.test_pairs)

    # base subset
    base_pos = build_base_subset_balanced_by_label(
        mnist_ds=train_raw,
        train_pairs=splits.train_pairs,
        base_ratio=float(args.base_ratio),
        seed=int(args.seed) + 3,
    )
    if len(base_pos) == 0:
        raise ValueError("base_pos is empty. Check splits/base_ratio/val_ratio.")
    base_subset = Subset(train_ds, base_pos)

    # loaders
    train_loader = make_loader(train_ds, batch_size=int(args.batch_size), shuffle=True, seed=int(args.seed) + 10, num_workers=int(args.num_workers))
    val_loader = make_loader(val_ds, batch_size=int(args.batch_size), shuffle=False, seed=int(args.seed) + 11, num_workers=int(args.num_workers))
    test_loader = make_loader(test_ds, batch_size=int(args.batch_size), shuffle=False, seed=int(args.seed) + 12, num_workers=int(args.num_workers))
    base_loader = make_loader(base_subset, batch_size=int(args.batch_size), shuffle=True, seed=int(args.seed) + 13, num_workers=int(args.num_workers))

    # expert train loaders (per-domain)
    expert_train_loaders: List[DataLoader] = []
    for d in range(k):
        tr_idx = [i for i, (_idx, dom) in enumerate(train_ds.pairs) if int(dom) == int(d)]
        tr_sub = Subset(train_ds, tr_idx)
        expert_train_loaders.append(
            make_loader(tr_sub, batch_size=int(args.batch_size), shuffle=True, seed=int(args.seed) + 100 + d, num_workers=int(args.num_workers))
        )

    # RED model
    model = ModularMNISTModelV11(num_classes=NUM_CLASSES, num_experts=k, hidden_dim=int(args.hidden_dim)).to(device)

    def log_split_metrics(tag: str, metrics: Dict) -> None:
        logger.info(
            "{} loss={:.6f} acc={:.4f} macro_f1={:.4f} macro_rec={:.4f} macro_prec={:.4f} w_f1={:.4f}".format(
                tag,
                float(metrics["loss"]),
                float(metrics["acc"]),
                float(metrics["macro_f1"]),
                float(metrics["macro_recall"]),
                float(metrics["macro_precision"]),
                float(metrics["weighted_f1"]),
            )
        )

    def red_pipeline_and_eval() -> Dict:
        # base
        base_stage = "base"
        if not is_stage_completed(args.save_dir, base_stage):
            attempt_resume(args.save_dir, prefix, base_stage, model, device=device, logger=logger)
            run_stage_with_early_stop(
                save_dir=args.save_dir,
                prefix=prefix,
                stage_name=base_stage,
                model=model,
                max_epochs=int(args.max_epochs_base),
                patience=int(args.patience),
                device=device,
                logger=logger,
                train_step_fn=lambda _ep: train_base_one_epoch(
                    model, base_loader, device=device,
                    lr=float(args.lr_base),
                    weight_decay=float(args.weight_decay),
                    virtual_batch_size=int(args.virtual_batch_size),
                ),
            )
            mark_stage_completed(args.save_dir, base_stage)
        load_best_checkpoint(args.save_dir, prefix, base_stage, model, device=device, strict=False, logger=logger)

        # experts
        for ei in range(k):
            st = "expert_" + str(ei)
            if not is_stage_completed(args.save_dir, st):
                attempt_resume(args.save_dir, prefix, st, model, device=device, logger=logger)
                run_stage_with_early_stop(
                    save_dir=args.save_dir,
                    prefix=prefix,
                    stage_name=st,
                    model=model,
                    max_epochs=int(args.max_epochs_expert),
                    patience=int(args.patience),
                    device=device,
                    logger=logger,
                    train_step_fn=lambda _ep, ei=ei: train_expert_one_epoch(
                        model, expert_train_loaders[ei], expert_id=ei, device=device,
                        lr=float(args.lr_expert),
                        weight_decay=float(args.weight_decay),
                        virtual_batch_size=int(args.virtual_batch_size),
                    ),
                )
                mark_stage_completed(args.save_dir, st)
            load_best_checkpoint(args.save_dir, prefix, st, model, device=device, strict=False, logger=logger)

        # router (domain CE)
        router_stage = "router"
        if not is_stage_completed(args.save_dir, router_stage):
            attempt_resume(args.save_dir, prefix, router_stage, model, device=device, logger=logger)
            run_stage_with_early_stop(
                save_dir=args.save_dir,
                prefix=prefix,
                stage_name=router_stage,
                model=model,
                max_epochs=int(args.max_epochs_router),
                patience=int(args.patience),
                device=device,
                logger=logger,
                train_step_fn=lambda _ep: train_router_domain_one_epoch(
                    model, train_loader, device=device,
                    lr=float(args.lr_router),
                    weight_decay=float(args.weight_decay),
                    virtual_batch_size=int(args.virtual_batch_size),
                ),
            )
            mark_stage_completed(args.save_dir, router_stage)
        load_best_checkpoint(args.save_dir, prefix, router_stage, model, device=device, strict=False, logger=logger)

        all_ones_mask = torch.ones(k, dtype=torch.float32)

        # -------- overall metrics (train/val/test) --------
        red_overall: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}}
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            m_base = evaluate_metrics(model, loader, mode="base", device=device, num_classes=NUM_CLASSES, expert_mask=None)
            m_all = evaluate_metrics(model, loader, mode="all_fixed", device=device, num_classes=NUM_CLASSES, expert_mask=all_ones_mask)
            m_router = evaluate_metrics(model, loader, mode="router", device=device, num_classes=NUM_CLASSES, expert_mask=None)

            red_overall[split_name]["base"] = m_base
            red_overall[split_name]["all_fixed"] = m_all
            red_overall[split_name]["router"] = m_router

        logger.info("========== [RED] OVERALL METRICS ==========")
        for split_name in ["train", "val", "test"]:
            log_split_metrics("[{}][base]".format(split_name), red_overall[split_name]["base"])
            log_split_metrics("[{}][all_fixed]".format(split_name), red_overall[split_name]["all_fixed"])
            log_split_metrics("[{}][router]".format(split_name), red_overall[split_name]["router"])

        # -------- per-domain full metrics on TEST --------
        per_domain_test = {
            "base": evaluate_per_domain_metrics(model, test_ds, batch_size=int(args.batch_size), device=device, mode="base",
                                                num_domains=k, num_classes=NUM_CLASSES, expert_mask=None, num_workers=int(args.num_workers)),
            "all_fixed": evaluate_per_domain_metrics(model, test_ds, batch_size=int(args.batch_size), device=device, mode="all_fixed",
                                                     num_domains=k, num_classes=NUM_CLASSES, expert_mask=all_ones_mask, num_workers=int(args.num_workers)),
            "router": evaluate_per_domain_metrics(model, test_ds, batch_size=int(args.batch_size), device=device, mode="router",
                                                  num_domains=k, num_classes=NUM_CLASSES, expert_mask=None, num_workers=int(args.num_workers)),
        }

        # -------- legacy v11-style summaries (keep: per-domain acc dicts) --------
        test_base_acc = float(red_overall["test"]["base"]["acc"])
        test_all_acc = float(red_overall["test"]["all_fixed"]["acc"])
        test_router_acc = float(red_overall["test"]["router"]["acc"])

        per_base_acc = {dd: float(per_domain_test["base"][dd]["acc"]) for dd in range(k)}
        per_all_acc = {dd: float(per_domain_test["all_fixed"][dd]["acc"]) for dd in range(k)}
        per_router_acc = {dd: float(per_domain_test["router"][dd]["acc"]) for dd in range(k)}

        logger.info("========== [RED] TEST RESULTS (legacy acc only) ==========")
        logger.info("TEST acc: base={:.4f} all_fixed={:.4f} router={:.4f}".format(
            test_base_acc, test_all_acc, test_router_acc
        ))
        logger.info("TEST per-domain base:   {}".format({kk: round(vv, 4) for kk, vv in per_base_acc.items()}))
        logger.info("TEST per-domain all:    {}".format({kk: round(vv, 4) for kk, vv in per_all_acc.items()}))
        logger.info("TEST per-domain router: {}".format({kk: round(vv, 4) for kk, vv in per_router_acc.items()}))

        # -------- concise per-domain key metrics in logs (avoid spam) --------
        logger.info("========== [RED] TEST PER-DOMAIN (key metrics) ==========")
        for d in range(k):
            mb = per_domain_test["base"][d]
            ma = per_domain_test["all_fixed"][d]
            mr = per_domain_test["router"][d]
            logger.info(
                "[domain {}] base(acc={:.4f},loss={:.4f},mf1={:.4f}) all(acc={:.4f},loss={:.4f},mf1={:.4f}) router(acc={:.4f},loss={:.4f},mf1={:.4f})".format(
                    d,
                    float(mb["acc"]), float(mb["loss"]), float(mb["macro_f1"]),
                    float(ma["acc"]), float(ma["loss"]), float(ma["macro_f1"]),
                    float(mr["acc"]), float(mr["loss"]), float(mr["macro_f1"]),
                )
            )

        return {
            "overall": red_overall,
            "test_per_domain": per_domain_test,  # full metrics per domain
            "test_per_domain_acc": {             # legacy acc-only dicts
                "base": per_base_acc,
                "all_fixed": per_all_acc,
                "router": per_router_acc,
            },
        }

    def moe_pipeline_and_eval() -> Dict:
        moe_model = ModularMNISTModelV11(num_classes=NUM_CLASSES, num_experts=k, hidden_dim=int(args.hidden_dim)).to(device)

        # ensure base+experts exist
        if not is_stage_completed(args.save_dir, "base"):
            _ = red_pipeline_and_eval()

        load_best_checkpoint(args.save_dir, prefix, "base", moe_model, device=device, strict=False, logger=logger)
        for ei in range(k):
            st = "expert_" + str(ei)
            if not is_stage_completed(args.save_dir, st):
                _ = red_pipeline_and_eval()
                break
            load_best_checkpoint(args.save_dir, prefix, st, moe_model, device=device, strict=False, logger=logger)

        # moe router training
        moe_stage = "moe"
        if not is_stage_completed(args.save_dir, moe_stage):
            attempt_resume(args.save_dir, prefix, moe_stage, moe_model, device=device, logger=logger)
            run_stage_with_early_stop(
                save_dir=args.save_dir,
                prefix=prefix,
                stage_name=moe_stage,
                model=moe_model,
                max_epochs=int(args.max_epochs_moe),
                patience=int(args.patience),
                device=device,
                logger=logger,
                train_step_fn=lambda _ep: train_moe_router_one_epoch(
                    moe_model, train_loader, device=device,
                    lr=float(args.lr_moe),
                    weight_decay=float(args.weight_decay),
                    virtual_batch_size=int(args.virtual_batch_size),
                ),
            )
            mark_stage_completed(args.save_dir, moe_stage)
        load_best_checkpoint(args.save_dir, prefix, moe_stage, moe_model, device=device, strict=False, logger=logger)

        # overall metrics (train/val/test)
        moe_overall: Dict[str, Dict] = {"train": {}, "val": {}, "test": {}}
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            moe_overall[split_name]["moe"] = evaluate_metrics(
                moe_model, loader, mode="moe", device=device, num_classes=NUM_CLASSES, expert_mask=None
            )

        logger.info("========== [MoE] OVERALL METRICS ==========")
        for split_name in ["train", "val", "test"]:
            log_split_metrics("[{}][moe]".format(split_name), moe_overall[split_name]["moe"])

        # per-domain full metrics on TEST
        per_domain_test = {
            "moe": evaluate_per_domain_metrics(moe_model, test_ds, batch_size=int(args.batch_size), device=device, mode="moe",
                                               num_domains=k, num_classes=NUM_CLASSES, expert_mask=None, num_workers=int(args.num_workers))
        }

        # legacy per-domain acc dict
        test_moe_acc = float(moe_overall["test"]["moe"]["acc"])
        per_moe_acc = {dd: float(per_domain_test["moe"][dd]["acc"]) for dd in range(k)}

        logger.info("========== [MoE] TEST RESULTS (legacy acc only) ==========")
        logger.info("TEST acc: moe={:.4f}".format(test_moe_acc))
        logger.info("TEST per-domain moe: {}".format({kk: round(vv, 4) for kk, vv in per_moe_acc.items()}))

        # concise per-domain key metrics
        logger.info("========== [MoE] TEST PER-DOMAIN (key metrics) ==========")
        for d in range(k):
            mm = per_domain_test["moe"][d]
            logger.info("[domain {}] moe(acc={:.4f},loss={:.4f},mf1={:.4f})".format(
                d, float(mm["acc"]), float(mm["loss"]), float(mm["macro_f1"])
            ))

        return {
            "overall": moe_overall,
            "test_per_domain": per_domain_test,  # full metrics per domain
            "test_per_domain_acc": {             # legacy acc-only dict
                "moe": per_moe_acc,
            },
        }

    results: Dict = {
        "meta": {
            "domain_mode": "label_pair_01_23_45_67_89",
            "num_domains": 5,
            "domain_map": {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]},
            "base_ratio": float(args.base_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "num_classes": NUM_CLASSES,
        }
    }

    if args.run in ["red", "all"]:
        results["red"] = red_pipeline_and_eval()
    if args.run in ["moe", "all"]:
        results["moe"] = moe_pipeline_and_eval()

    out_path = Path(args.save_dir) / "mnist_v11_label5_all_results.json"
    save_json(results, str(out_path))
    logger.info("=> saved combined results to {}".format(out_path))


if __name__ == "__main__":
    main()
