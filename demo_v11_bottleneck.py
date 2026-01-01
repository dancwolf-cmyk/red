# -*- coding: utf-8 -*-
"""
demo_router_qwen_soft_v11_bottleneck.py
---------------------------------------
在 demo_v11.py 基础上增加“低秩瓶颈专家”（Low-Rank Experts）版本，用于 4.9.3 的 r 消融实验。

- 复用内容：
    - encoder（QwenEncoderFrozen）
    - base_head
    - router（range 分类器）
- 需要重训：
    - experts（现在为 H -> r -> C 的低秩结构）

使用方法：
    1. 确保原 v11 训练过 base 和 router，对应 checkpoint 在 SAVE_DIR 下。
    2. 修改 BOTTLENECK_RATIO 为 0.5 / 0.25 / 0.125 分别运行三次。
"""

from pathlib import Path
import json
import random
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ---------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_SAMPLES = 40000
DEFAULT_SEED = 42
BASE_RATIO = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64
MAX_EPOCHS = 60           # base / expert 最长 epoch（配合 early stop）
MAX_EPOCHS2 = 0         # base 首轮训练最多 epoch
PATIENCE = 3
SAVE_DIR = "checkpoints"  # 复用原 v11 的目录
LOG_FILE = Path(SAVE_DIR) / "demo_v11_bottleneck.log"
MODEL_PATH = r"e:/dev/lunwen/Qwen3-0.6B"
STAGE_STATUS_FILE = Path(SAVE_DIR) / "stage_status_v11_b.json"
SAMPLES_CACHE_FILE = Path(SAVE_DIR) / "demo_v11_samples.json"

# 瓶颈比例：例如 0.5, 0.25, 0.125 分别对应 r=d/2, d/4, d/8
BOTTLENECK_RATIO = 0.5
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if USE_AMP else torch.float32

def autocast_ctx():
    if USE_AMP:
        return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


# ---------------------------------------------------------------------
# 日志与随机数
# ---------------------------------------------------------------------
def setup_logger():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo_v11_bottleneck")
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


# ---------------------------------------------------------------------
# 数据缓存与 stage 状态
# ---------------------------------------------------------------------
def save_samples_to_cache(samples: List[Dict], label_vocab: List[str], path: Path = SAMPLES_CACHE_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "label_vocab": label_vocab}, f, ensure_ascii=False)
    logger.info("=> cached samples to {} (n={})".format(path, len(samples)))


def load_samples_from_cache(path: Path = SAMPLES_CACHE_FILE) -> Tuple[List[Dict], List[str]]:
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples"), data.get("label_vocab")
    return data, None


def _collect_stage_state_dict(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in trainable}
    return filtered if filtered else state


def save_checkpoint(model, stage, epoch=None):
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    suffix = "_ep{}".format(epoch) if epoch is not None else ""
    path = Path(SAVE_DIR) / "demo_v11_bottleneck_{}{}.pt".format(stage, suffix)
    torch.save({"state_dict": _collect_stage_state_dict(model), "stage": stage, "epoch": epoch}, path)
    logger.info("=> checkpoint saved: {}".format(path))
    return path


def _stage_checkpoint_glob(stage_name: str, suffix: str) -> List[Path]:
    return sorted(Path(SAVE_DIR).glob(f"demo_v11_bottleneck_{stage_name}_{suffix}*.pt"))


def get_best_checkpoint(stage_name: str) -> Path:
    matches = _stage_checkpoint_glob(stage_name, "best")
    if matches:
        return matches[-1]
    # fallback to old naming
    old_matches = sorted(Path(SAVE_DIR).glob("demo_v11_{}_best*.pt".format(stage_name)))
    if old_matches:
        return old_matches[-1]
    return None


def get_last_checkpoint(stage_name: str) -> Path:
    matches = _stage_checkpoint_glob(stage_name, "last")
    if matches:
        return matches[-1]
    return None


def save_last_checkpoint(model, stage, epoch, best_loss=float("inf")):
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(SAVE_DIR) / "demo_v11_bottleneck_{}_last_ep{}.pt".format(stage, epoch)
    if path.exists():
        path.unlink()
    torch.save({"state_dict": _collect_stage_state_dict(model), "stage": stage, "epoch": epoch,
                "best_loss": best_loss}, path)
    logger.info("=> last checkpoint saved: {}".format(path))


def _get_accum_steps(loader: DataLoader, virtual_batch_size: int):
    unit = getattr(loader, "batch_size", 1) or 1
    return max(1, virtual_batch_size // unit)


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


def get_best_checkpoint(stage_name: str) -> Path:
    matches_new = _stage_checkpoint_glob(stage_name, "best")
    if matches_new:
        return matches_new[-1]
    matches_old = sorted(Path(SAVE_DIR).glob("demo_v11_{}_best*.pt".format(stage_name)))
    if matches_old:
        return matches_old[-1]
    return None


def load_best_checkpoint(stage_name: str, model, strict: bool = False):
    path = get_best_checkpoint(stage_name)
    if not path or not path.exists():
        return None
    try:
        # ✅ 始终先加载到 CPU
        state = torch.load(path, map_location="cpu")
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if not strict:
            logger.info(
                "=> loaded best checkpoint for {}: {} (missing={} unexpected={})"
                .format(stage_name, path, len(missing), len(unexpected))
            )
        else:
            logger.info("=> loaded best checkpoint for {}: {}".format(stage_name, path))

        # ✅ 手动释放临时 state / sd，顺便清空 CUDA 缓存
        del state
        del sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return path
    except RuntimeError as e:
        logger.warning("=> failed to load checkpoint {}: {}".format(path, e))
        return None



def run_stage_with_early_stop(stage_name, train_step_fn, val_loss_fn, model,
                              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    best_loss = float("inf")
    patience_cnt = 0
    best_epoch = 0
    best_path = None

    last_path = get_last_checkpoint(stage_name)
    start_epoch = 1
    if last_path and last_path.exists():
        state = torch.load(last_path, map_location="cpu")
        sd = state.get("state_dict", state)
        model.load_state_dict(sd, strict=False)
        start_epoch = state.get("epoch", 0) + 1
        best_loss = state.get("best_loss", best_loss)
        logger.info(
            "=> Resuming {} from epoch {} (best_loss={:.4f})"
            .format(stage_name, start_epoch, best_loss)
        )
        del state
        del sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for epoch in range(start_epoch, max_epochs + 1):
        loss, acc = train_step_fn(epoch)
        val_loss = val_loss_fn() if val_loss_fn else loss

        # 👇 每一轮结束，明确打印当前阶段的 val_loss
        logger.info(
            "[Stage: {}] epoch {} train_loss={:.4f} train_acc={:.4f} val_loss={:.4f}"
            .format(stage_name, epoch, loss, acc, val_loss)
        )

        if val_loss < best_loss:
            if best_path and best_path.exists():
                best_path.unlink()
            best_loss = val_loss
            best_epoch = epoch
            patience_cnt = 0
            best_path = save_checkpoint(model, "{}_best".format(stage_name), epoch=epoch)
        else:
            patience_cnt += 1

        save_last_checkpoint(model, stage_name, epoch, best_loss)

        if patience_cnt >= patience:
            logger.info("=> Early stopping {} at epoch {} (patience={})"
                        .format(stage_name, epoch, patience))
            break

    return best_loss, best_epoch, best_path



# ---------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------
DOMAIN_ORDER = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]


def build_addition_dataset(num_samples=20000, seed=DEFAULT_SEED) -> List[Dict]:
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
    samples: List[Dict] = []
    for i, (domain, lo, hi) in enumerate(ranges):
        count = per_domain + (1 if i < remainder else 0)
        for _ in range(count):
            while True:
                a = random.randint(0, hi)
                b = random.randint(0, hi)
                s = a + b
                if lo <= s <= hi:
                    break
            question = "{} plus {}".format(a, b)
            answer = str(s)
            samples.append({"question": question, "answer": answer, "domain": domain})
    random.shuffle(samples)
    return samples


def sample_balanced_subset(samples: List[Dict], count: int) -> List[Dict]:
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
    def __init__(self, samples: List[Dict], tokenizer, label_vocab: List[str], max_len: int = 32):
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


def load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED):
    samples, label_vocab = load_samples_from_cache()
    if samples is not None and label_vocab is not None:
        logger.info("=> Loaded samples from cache (n={})".format(len(samples)))
        return samples, label_vocab
    samples = build_addition_dataset(num_samples=num_samples, seed=seed)
    # label_vocab: 所有可能答案（字符串）排序后
    answers = sorted({s["answer"] for s in samples}, key=lambda x: int(x))
    save_samples_to_cache(samples, answers)
    logger.info("=> Built new dataset (n={})".format(len(samples)))
    return samples, answers


# ---------------------------------------------------------------------
# 模型定义：Encoder + 低秩专家 + Router
# ---------------------------------------------------------------------
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


class LowRankExpert(nn.Module):
    """
    低秩专家：H -> r -> C
    r 远小于 H 时，参数量大幅下降。
    """
    def __init__(self, d: int, c: int, r: int):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, c, bias=False)

    def forward(self, h):
        return self.up(self.down(h))


class ModularAddModelWithRouterBottleneck(nn.Module):
    """
    带低秩专家的 RED 模型：
      - base_head: H -> C
      - experts:   H -> r -> C (LowRankExpert)
      - router:    H -> K
    """
    def __init__(self, encoder: QwenEncoderFrozen, num_classes: int, num_experts: int = 5,
                 bottleneck_ratio: float = 0.25):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size
        self.base_head = nn.Linear(H, num_classes)
        # 计算瓶颈维度 r
        r = max(1, int(H * bottleneck_ratio))
        self.bottleneck_ratio = bottleneck_ratio
        self.bottleneck_dim = r
        self.experts = nn.ModuleList([LowRankExpert(H, num_classes, r) for _ in range(num_experts)])
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

        raise ValueError("Unknown mode: {}".format(mode))


# ---------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------
def _train_loop(model, loader, optim, criterion, train_step_fn, desc, accum_steps):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    iters = 0
    grad_step = 0
    pbar = tqdm(loader, desc=desc, ncols=120)
    optim.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    for batch in pbar:
        with autocast_ctx():
            logits, label = train_step_fn(batch)
            loss = criterion(logits, label) / accum_steps
        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_step += 1
        if grad_step % accum_steps == 0:
            if USE_AMP:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad()
        pred = logits.argmax(-1)
        acc = (pred == label).float().mean().item()
        total_loss += loss.item() * accum_steps
        total_acc += acc
        iters += 1
        pbar.set_postfix(
            loss="{:.4f}".format(total_loss / iters),
            acc="{:.4f}".format(total_acc / iters),
        )
    if grad_step % accum_steps != 0:
        if USE_AMP:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()
    return total_loss / iters, total_acc / iters


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

    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=base_step,
            desc="[Base] epoch {}/{}".format(ep, epochs),
            accum_steps=accum_steps,
        )
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

    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=expert_step,
            desc="[Expert {}] epoch {}/{}".format(ei, ep, epochs),
            accum_steps=accum_steps,
        )
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

    for ep in range(1, epochs + 1):
        loss, acc = _train_loop(
            model, loader, optim, ce,
            train_step_fn=router_step,
            desc="[Router] epoch {}/{}".format(ep, epochs),
            accum_steps=accum_steps,
        )
        logger.info("=> Router epoch {} avg_loss={:.4f} avg_acc={:.4f}".format(ep, loss, acc))
    return loss, acc


def _compute_loss_and_preds(model, batch, mode):
    ids = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    label = batch["label"].to(DEVICE)
    logits = model(ids, mask, mode=mode)
    return logits, label


@torch.no_grad()
def evaluate(model, loader, mode, desc):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        logits, label = _compute_loss_and_preds(model, batch, mode)
        loss = criterion(logits, label)
        pred = logits.argmax(-1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())
    preds = torch.cat(all_preds) if all_preds else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    num_classes = len(loader.dataset.label_to_id)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / float(denom))
    macro_f1 = sum(f1s) / float(num_classes if num_classes > 0 else 1)
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else float("inf")
    logger.info("[Eval] {} acc={:.4f} loss={:.4f} macro_f1={:.4f}".format(desc, acc, avg_loss, macro_f1))
    return acc, avg_loss, macro_f1


@torch.no_grad()
def evaluate_per_range(model, loaders_per_range, mode, tag):
    model.eval()
    ranges = ["2-20", "21-40", "41-60", "61-80", "81-100"]
    logger.info("=== Per-range evaluation: {} (mode={}) ===".format(tag, mode))
    for i, loader in enumerate(loaders_per_range):
        total_correct = 0
        total_count = 0
        pbar = tqdm(loader, desc="[{}] Range {}".format(tag, ranges[i]), ncols=120)
        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids, mask, mode=mode)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
        acc = float(total_correct) / float(total_count if total_count > 0 else 1)
        logger.info("Range {}: acc={:.4f} (n={})".format(ranges[i], acc, total_count))


@torch.no_grad()
def compute_loader_loss(model, loader, mode, criterion, device, expert_mask=None, domain_to_id=None,
                        desc: str = ""):
    model.eval()
    total_loss = 0.0
    total = 0
    total_correct = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=desc or f"Val {mode}", ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        if mode == "router":
            domains = batch["domain"]
            domain_ids = torch.tensor([domain_to_id[d] for d in domains], device=device)
            logits = model(ids, mask, mode=mode)
            loss = criterion(logits, domain_ids)
            count = domain_ids.size(0)
            preds = logits.argmax(-1)
            labels = domain_ids
        else:
            label = batch["label"].to(device)
            logits = model(ids, mask, mode=mode, expert_mask=expert_mask)
            loss = criterion(logits, label)
            count = label.size(0)
            preds = logits.argmax(dim=-1)
            labels = label
        total_loss += loss.item() * count
        total += count
        total_correct += (preds == labels).sum().item()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    result = total_loss / total if total else float("inf")
    preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    num_classes = len(domain_to_id) if mode == "router" else getattr(loader.dataset, "label_to_id", {})
    num_classes = len(num_classes) if isinstance(num_classes, (list, dict)) else num_classes
    num_classes = num_classes or (preds.max().item() + 1 if preds.numel() else 0)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / float(denom))
    macro_f1 = sum(f1s) / float(num_classes if num_classes > 0 else 1)
    acc = total_correct / total if total > 0 else 0.0
    logger.info("=> {} val_loss={:.4f} acc={:.4f} macro_f1={:.4f}".format(desc or mode, result, acc, macro_f1))
    return result


# ---------------------------------------------------------------------
# 主流程（与 v11 基本一致，只是模型换成 bottleneck 版本）
# ---------------------------------------------------------------------
def main():
    set_seed(DEFAULT_SEED)

    samples, label_vocab = load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED)
    random.seed(DEFAULT_SEED + 1)
    val_size = int(len(samples) * VAL_RATIO)
    test_size = int(len(samples) * TEST_RATIO)
    train_size = len(samples) - val_size - test_size
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    train_ds = QADataset(train_samples, tokenizer, label_vocab)
    test_ds = QADataset(test_samples, tokenizer, label_vocab)
    all_ds = QADataset(samples, tokenizer, label_vocab)

    def make_loader(dataset, shuffle=False, seed_offset=0):
        gen = torch.Generator()
        gen.manual_seed(DEFAULT_SEED + seed_offset)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)

    train_loader = make_loader(train_ds, shuffle=True, seed_offset=0)
    val_loader = make_loader(QADataset(val_samples, tokenizer, label_vocab), shuffle=False, seed_offset=1)
    test_loader = make_loader(test_ds, shuffle=False, seed_offset=2)
    all_loader = make_loader(all_ds, shuffle=False, seed_offset=3)

    train_base_count = max(1, int(train_size * BASE_RATIO))
    random.seed(DEFAULT_SEED + 2)
    base_subset = sample_balanced_subset(train_samples, train_base_count)
    base_loader = make_loader(QADataset(base_subset, tokenizer, label_vocab), shuffle=True, seed_offset=4)

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

    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = ModularAddModelWithRouterBottleneck(
        encoder,
        num_classes=len(label_vocab),
        num_experts=5,
        bottleneck_ratio=BOTTLENECK_RATIO,
    ).to(DEVICE)

    logger.info("=> Bottleneck ratio = {} (r = {})".format(
        BOTTLENECK_RATIO, model.bottleneck_dim
    ))

    # 1) 复用/训练 base
    base_stage = "base"
    ce = nn.CrossEntropyLoss()
    base_val_loss_fn = lambda: compute_loader_loss(model, val_loader, "base", ce, DEVICE)
    if not is_stage_completed(base_stage):
        # 尝试从原 v11 或 v11_best 中加载 base 权重
        load_best_checkpoint(base_stage, model, strict=False)
        _, _, _ = run_stage_with_early_stop(
            base_stage,
            lambda epoch: train_base(model, base_loader, epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE),
            base_val_loss_fn,
            model,
            max_epochs=MAX_EPOCHS2,
        )
        mark_stage_completed(base_stage)
        load_best_checkpoint(base_stage, model, strict=False)
    else:
        load_best_checkpoint(base_stage, model, strict=False)

    # evaluate(model, all_loader, "base", "Base-only All")
    # evaluate_per_range(model, base_range_loaders, mode="base", tag="Base-only")

    # 2) 复用 router（如果已有），否则也可以重新训练
    router_stage = "router"
    router_ce = nn.CrossEntropyLoss()
    router_val_loss_fn = lambda: compute_loader_loss(
        model, val_loader, "router", router_ce, DEVICE, domain_to_id=domain_to_expert
    )
    if get_best_checkpoint(router_stage) is None:
        logger.info("=> No existing router checkpoint found, training router...")
        _ = run_stage_with_early_stop(
            router_stage,
            lambda epoch: train_router(model, train_loader, domain_to_expert, epochs=1,
                                       virtual_batch_size=VIRTUAL_BATCH_SIZE),
            router_val_loss_fn,
            model,
        )
        mark_stage_completed(router_stage)
        load_best_checkpoint(router_stage, model, strict=False)
    else:
        load_best_checkpoint(router_stage, model, strict=False)

    # 3) 针对当前 bottleneck_ratio 重新训练所有 experts
    expert_ce = nn.CrossEntropyLoss()
    for ei in range(5):
        expert_stage = "expert_{}_r{}".format(ei, int(BOTTLENECK_RATIO * 1000))
        if not is_stage_completed(expert_stage):
            mask = torch.zeros(model.num_experts, device=DEVICE)
            mask[ei] = 1.0
            val_loss_fn = lambda ei=ei, mask=mask: compute_loader_loss(
                model,
                expert_test_loaders[ei],
                "all_fixed",
                expert_ce,
                DEVICE,
                expert_mask=mask,
            )
            _, _, _ = run_stage_with_early_stop(
                expert_stage,
                lambda epoch, ei=ei: train_expert(
                    model,
                    expert_train_loaders[ei],
                    ei,
                    epochs=1,
                    virtual_batch_size=VIRTUAL_BATCH_SIZE,
                ),
                val_loss_fn,
                model,
            )
            mark_stage_completed(expert_stage)
        else:
            # 如果你希望从之前的瓶颈版本继续，也可以在这里 load_best_checkpoint
            load_best_checkpoint(expert_stage, model, strict=False)

    # 4) 评估 AllExperts & Router（对应论文 4.6 / 4.7 / 4.9）
    evaluate(model, train_loader, "all_fixed", "AllExperts Train (bottleneck)")
    evaluate(model, val_loader, "all_fixed", "AllExperts Val (bottleneck)")
    evaluate(model, test_loader, "all_fixed", "AllExperts Test (bottleneck)")
    evaluate_per_range(model, expert_test_loaders, mode="all_fixed", tag="AllExperts-fixed (bottleneck)")

    evaluate(model, train_loader, "router", "Router Train (bottleneck)")
    evaluate(model, val_loader, "router", "Router Val (bottleneck)")
    evaluate(model, test_loader, "router", "Router Test (bottleneck)")
    evaluate_per_range(model, expert_test_loaders, mode="router", tag="Router (bottleneck)")


if __name__ == "__main__":
    main()
