# -*- coding: utf-8 -*-
"""
demo_router_qwen_soft_v11.py
----------------------------
- 在 demo_v10 基础上变更数据规模：总样本增加到 40000，base 训练只采 1/5 的 train split。
- 其余 stages/open evaluations 保持一致，依然支持虚拟 batch + validation + checkpoints。
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_SAMPLES = 40000
DEFAULT_SEED = 42
BASE_RATIO = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64
MAX_EPOCHS = 60
MAX_EPOCHS2 = 10
PATIENCE = 3
SAVE_DIR = "checkpoints"
LOG_FILE = Path(SAVE_DIR) / "demo_v11.log"
MODEL_PATH = r"e:/dev/lunwen/Qwen3-0.6B"
STAGE_STATUS_FILE = Path(SAVE_DIR) / "stage_status.json"
SAMPLES_CACHE_FILE = Path(SAVE_DIR) / "demo_v11_samples.json"


def setup_logger():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo_v11")
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


def save_samples_to_cache(samples: List[Dict], label_vocab: List[str], path: Path = SAMPLES_CACHE_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "label_vocab": label_vocab}, f, ensure_ascii=False)
    logger.info(f"=> cached samples to {path} (n={len(samples)})")


def load_samples_from_cache(path: Path = SAMPLES_CACHE_FILE):
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples"), data.get("label_vocab")
    return data, None


def save_checkpoint(model, stage, epoch=None):
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    suffix = f"_ep{epoch}" if epoch is not None else ""
    path = Path(SAVE_DIR) / f"demo_v11_{stage}{suffix}.pt"
    torch.save({"state_dict": model.state_dict(),
                "stage": stage, "epoch": epoch}, path)
    logger.info(f"=> checkpoint saved: {path}")
    return path


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

DOMAIN_ORDER = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]


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
                q = f"What is {a} plus {b}?"
                samples.append({"question": q, "answer": str(s), "domain": name})
                collected += 1

    random.shuffle(samples)
    label_vocab = [str(i) for i in range(2, 101)]
    return samples, label_vocab


def load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED, cache_path: Path = SAMPLES_CACHE_FILE):
    cached_samples, cached_vocab = load_samples_from_cache(cache_path)
    if cached_samples is not None and len(cached_samples) == num_samples:
        label_vocab = cached_vocab or [str(i) for i in range(2, 101)]
        logger.info(f"=> loaded cached samples from {cache_path} (n={len(cached_samples)})")
        return cached_samples, label_vocab
    samples, label_vocab = build_addition_dataset(num_samples=num_samples, seed=seed)
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
        loss = criterion(logits, label) / accum_steps
        loss.backward()
        grad_step += 1
        if grad_step % accum_steps == 0:
            optim.step()
            optim.zero_grad()
        pred = logits.argmax(-1)
        acc = (pred == label).float().mean().item()
        total_loss += loss.item() * accum_steps
        total_acc += acc
        iters += 1
        pbar.set_postfix(loss="{:.4f}".format(total_loss / iters),
                         acc="{:.4f}".format(total_acc / iters))
    if grad_step % accum_steps != 0:
        optim.step()
        optim.zero_grad()
    return total_loss / iters, total_acc / iters


def attempt_resume(stage_name, model, loaders=None):
    path = load_best_checkpoint(stage_name, model, strict=False)
    if path is None:
        return False
    logger.info(f"=> resumed {stage_name} from {path}")
    if loaders is not None and stage_name.startswith("expert_"):
        ei = int(stage_name.split("_")[1])
        if loaders[ei] is None:
            return False
    return True


def get_best_checkpoint(stage_name):
    matches = sorted(Path(SAVE_DIR).glob(f"demo_v11_{stage_name}_best*.pt"))
    if not matches:
        return None
    # Prefer non-slim if both exist; otherwise take latest.
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
            logger.info(f"=> loaded best checkpoint for {stage_name}: {path} (missing={len(missing)} unexpected={len(unexpected)})")
        else:
            logger.info(f"=> loaded best checkpoint for {stage_name}: {path}")
        return path
    except RuntimeError as e:
        logger.warning(f"=> failed to load checkpoint {path}: {e}")
        return None


def run_stage_with_early_stop(stage_name, train_step_fn, model, max_epochs=MAX_EPOCHS, patience=PATIENCE):
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
            best_path = save_checkpoint(model, f"{stage_name}_best", epoch=epoch)
        else:
            patience_cnt += 1
        if patience_cnt >= patience:
            logger.info(f"=> Early stopping {stage_name} at epoch {epoch} (patience={patience})")
            break
    return best_loss, best_epoch, best_path


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
            desc=f"[Base] epoch {ep}/{epochs}", accum_steps=accum_steps)
        logger.info(f"=> Base epoch {ep} avg_loss={loss:.4f} avg_acc={acc:.4f}")
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
            desc=f"[Expert {ei}] epoch {ep}/{epochs}", accum_steps=accum_steps)
        logger.info(f"=> Expert {ei} epoch {ep} avg_loss={loss:.4f} avg_acc={acc:.4f}")
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
            desc=f"[Router] epoch {ep}/{epochs}", accum_steps=accum_steps)
        logger.info(f"=> Router epoch {ep} avg_loss={loss:.4f} avg_acc={acc:.4f}")
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, mode, desc):
    model.eval()
    total = correct = 0
    pbar = tqdm(loader, desc=desc, ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        logits = model(ids, mask, mode=mode)
        pred = logits.argmax(-1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    acc = correct / total if total > 0 else 0.0
    logger.info(f"[Eval] {desc} acc={acc:.4f}")
    return acc


@torch.no_grad()
def evaluate_per_range(model, loaders_per_range, mode, tag):
    model.eval()
    ranges = ["2-20", "21-40", "41-60", "61-80", "81-100"]
    logger.info("=== Per-range evaluation: {} (mode={}) ===".format(tag, mode))
    for i, loader in enumerate(loaders_per_range):
        total_correct = 0
        total_count = 0
        pbar = tqdm(loader, desc=f"[{tag}] Range {ranges[i]}", ncols=120)
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
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = ModularAddModelWithRouter(encoder, num_classes=len(label_vocab)).to(DEVICE)
    base_stage = "base"
    if not is_stage_completed(base_stage):
        attempt_resume(base_stage, model)
        _, _, best_path = run_stage_with_early_stop(
            base_stage,
            lambda epoch: train_base(model, base_loader, epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE),
            model,
            max_epochs=MAX_EPOCHS2,
        )
        mark_stage_completed(base_stage)
        load_best_checkpoint(base_stage, model, strict=False)
        evaluate(model, all_loader, "base", "Base-only All")
        evaluate_per_range(model, base_range_loaders, mode="base", tag="Base-only")
    else:
        load_best_checkpoint(base_stage, model)
    for ei in range(5):
        expert_stage = f"expert_{ei}"
        if not is_stage_completed(expert_stage):
            attempt_resume(expert_stage, model)
            _, _, best_path = run_stage_with_early_stop(
                expert_stage,
                lambda epoch, ei=ei: train_expert(model, expert_train_loaders[ei], ei, epochs=1,
                                                  virtual_batch_size=VIRTUAL_BATCH_SIZE),
                model,
            )
            mark_stage_completed(expert_stage)
            load_best_checkpoint(expert_stage, model, strict=False)
        else:
            load_best_checkpoint(expert_stage, model, strict=False)
    evaluate(model, train_loader, "all_fixed", "AllExperts Train")
    evaluate(model, val_loader, "all_fixed", "AllExperts Val")
    evaluate(model, test_loader, "all_fixed", "AllExperts Test")
    evaluate_per_range(model, expert_test_loaders, mode="all_fixed", tag="AllExperts-fixed")
    router_stage = "router"
    if not is_stage_completed(router_stage):
        attempt_resume(router_stage, model)
        _, _, best_path = run_stage_with_early_stop(
            router_stage,
            lambda epoch: train_router(model, train_loader, domain_to_expert, epochs=1, virtual_batch_size=VIRTUAL_BATCH_SIZE),
            model,
        )
        mark_stage_completed(router_stage)
        load_best_checkpoint(router_stage, model, strict=False)
    else:
        load_best_checkpoint(router_stage, model, strict=False)
    evaluate(model, train_loader, "router", "Router Train")
    evaluate(model, val_loader, "router", "Router Val")
    evaluate(model, test_loader, "router", "Router Test")
    evaluate_per_range(model, expert_test_loaders, mode="router", tag="Router")


if __name__ == "__main__":
    main()
