"""
Evaluate base / all_fixed / router modes with full checkpoints (recommended; slim allowed but may degrade).
Computes accuracy and macro-F1; runs per-range acc; logs to checkpoints/eval_all_fixed.log.

Example (full ckpts):
python eval_all_fixed_multi.py 
  --base-ckpt checkpoints/demo_v11_base_best_ep10.pt 
  --expert-ckpt checkpoints/demo_v11_expert_0_best_ep60.pt 
  --expert-ckpt checkpoints/demo_v11_expert_1_best_ep60.pt 
  --expert-ckpt checkpoints/demo_v11_expert_2_best_ep60.pt 
  --expert-ckpt checkpoints/demo_v11_expert_3_best_ep60.pt 
  --expert-ckpt checkpoints/demo_v11_expert_4_best_ep60.pt 
  --router-ckpt checkpoints/demo_v11_router_best_ep8.pt 
  --model-path c:/temp/lunwen/Qwen3-0.6B 
  --trainable-last-n-layers 1

If you must use slim, keep the same args; loader will fallback to non-strict load with a warning.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from demo_v11 import (
    BATCH_SIZE,
    DEVICE,
    DEFAULT_SEED,
    TOTAL_SAMPLES,
    VAL_RATIO,
    TEST_RATIO,
    load_or_create_samples,
    set_seed,
    QADataset,
    ModularAddModelWithRouter,
    QwenEncoderFrozen,
)


LOG_FILE = Path("checkpoints") / "eval_all_fixed.log"


def setup_logger():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_all_fixed")
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


def _load_sd(path: Path) -> Dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    return raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw


def _apply_base(model: torch.nn.Module, sd: Dict[str, torch.Tensor]):
    try:
        missing, unexpected = model.load_state_dict(sd, strict=True)
        logger.info(f"Loaded base ckpt (strict); missing={len(missing)} unexpected={len(unexpected)}")
    except RuntimeError as e:
        logger.warning(f"Strict load failed for base ckpt, falling back to non-strict. Error: {e}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded base ckpt (non-strict); missing={len(missing)} unexpected={len(unexpected)}")


def _apply_expert(model: torch.nn.Module, sd: Dict[str, torch.Tensor], idx: int):
    prefix = f"experts.{idx}."
    filtered = {k: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded expert {idx}; keys={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")


def _apply_router(model: torch.nn.Module, sd: Dict[str, torch.Tensor]):
    filtered = {k: v for k, v in sd.items() if k.startswith("router") or k.startswith("router_scale")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(f"Loaded router; keys={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return sum(f1s) / num_classes


@torch.no_grad()
def evaluate(model, loader, desc, mode: str, num_classes: int):
    model.eval()
    total = correct = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=f"{desc} ({mode})", ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(ids, mask, mode=mode)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        pbar.set_postfix(acc="{:.4f}".format(correct / total if total else 0.0))
    acc = correct / total if total else 0.0
    preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0)
    labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0)
    f1 = macro_f1(preds_cat, labels_cat, num_classes=num_classes) if total else 0.0
    logger.info(f"[{desc}] mode={mode} acc={acc:.4f} macro_f1={f1:.4f} n={total}")
    return acc, f1


@torch.no_grad()
def evaluate_per_range(model, loaders_per_range, tag, mode: str, num_classes: int):
    model.eval()
    ranges = ["2-20", "21-40", "41-60", "61-80", "81-100"]
    logger.info(f"=== Per-range evaluation: {tag} (mode={mode}) ===")
    for i, loader in enumerate(loaders_per_range):
        total_correct = 0
        total_count = 0
        all_preds = []
        all_labels = []
        pbar = tqdm(loader, desc=f"[{tag}] Range {ranges[i]} ({mode})", ncols=120)
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
            pbar.set_postfix(acc="{:.4f}".format(acc))
        acc = float(total_correct) / float(total_count if total_count > 0 else 1)
        preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0)
        labels_cat = torch.cat(all_labels) if all_labels else torch.empty(0)
        f1 = macro_f1(preds_cat, labels_cat, num_classes=num_classes) if total_count else 0.0
        logger.info("Range {}: acc={:.4f} macro_f1={:.4f} (n={})".format(ranges[i], acc, f1, total_count))


def build_loaders(model_path: str):
    set_seed(DEFAULT_SEED)
    samples, label_vocab = load_or_create_samples(num_samples=TOTAL_SAMPLES, seed=DEFAULT_SEED)
    val_size = int(len(samples) * VAL_RATIO)
    test_size = int(len(samples) * TEST_RATIO)
    train_size = len(samples) - val_size - test_size
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def dl(data, shuffle):
        return DataLoader(QADataset(data, tokenizer, label_vocab), batch_size=BATCH_SIZE, shuffle=shuffle)

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

    loaders = {
        "train": dl(train_samples, False),
        "val": dl(val_samples, False),
        "test": dl(test_samples, False),
        "all": dl(samples, False),
        "per_range": [dl(sub, False) for sub in expert_all],
        "num_classes": len(label_vocab),
    }
    return loaders


def main():
    parser = argparse.ArgumentParser(description="Eval base/all_fixed/router with provided checkpoints.")
    parser.add_argument("--base-ckpt", required=True, help="Base checkpoint (slim or full).")
    parser.add_argument(
        "--expert-ckpt",
        action="append",
        required=True,
        help="Expert checkpoint paths; provide 5 times in order 0..4.",
    )
    parser.add_argument("--model-path", default=None, help="Base model path (Qwen).")
    parser.add_argument("--trainable-last-n-layers", type=int, default=1, help="Must match slim creation.")
    parser.add_argument("--router-ckpt", default=None, help="Optional router checkpoint (slim or full).")
    args = parser.parse_args()

    if len(args.expert_ckpt) != 5:
        raise SystemExit("Please provide exactly 5 --expert-ckpt paths in order 0..4.")

    from demo_v11 import MODEL_PATH as DEFAULT_MODEL_PATH  # local import to avoid circular refs
    model_path = args.model_path or DEFAULT_MODEL_PATH

    loaders = build_loaders(model_path)

    encoder = QwenEncoderFrozen(model_path=model_path, trainable_last_n_layers=args.trainable_last_n_layers)
    model = ModularAddModelWithRouter(encoder, num_classes=loaders["num_classes"]).to(DEVICE)

    # base
    base_sd = _load_sd(Path(args.base_ckpt))
    _apply_base(model, base_sd)
    logger.info("Starting evaluation (mode=base)")
    evaluate(model, loaders["all"], "All data", mode="base", num_classes=loaders["num_classes"])
    evaluate_per_range(model, loaders["per_range"], tag="All data per-range", mode="base", num_classes=loaders["num_classes"])
    evaluate(model, loaders["train"], "Train", mode="base", num_classes=loaders["num_classes"])
    evaluate(model, loaders["val"], "Val", mode="base", num_classes=loaders["num_classes"])
    evaluate(model, loaders["test"], "Test", mode="base", num_classes=loaders["num_classes"])

    # experts → all_fixed
    for idx, ckpt_path in enumerate(args.expert_ckpt):
        logger.info(f"Loading expert {idx} from {ckpt_path} ...")
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start and end:
            start.record()
        sd = _load_sd(Path(ckpt_path))
        _apply_expert(model, sd, idx)
        if start and end:
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            logger.info(f"Expert {idx} loaded in {elapsed_ms/1000:.2f}s")

    logger.info("Starting evaluation (mode=all_fixed)")
    evaluate(model, loaders["all"], "All data", mode="all_fixed", num_classes=loaders["num_classes"])
    evaluate_per_range(model, loaders["per_range"], tag="All data per-range", mode="all_fixed", num_classes=loaders["num_classes"])
    evaluate(model, loaders["train"], "Train", mode="all_fixed", num_classes=loaders["num_classes"])
    evaluate(model, loaders["val"], "Val", mode="all_fixed", num_classes=loaders["num_classes"])
    evaluate(model, loaders["test"], "Test", mode="all_fixed", num_classes=loaders["num_classes"])

    # router (optional)
    if args.router_ckpt:
        router_sd = _load_sd(Path(args.router_ckpt))
        _apply_router(model, router_sd)
        logger.info("Starting evaluation (mode=router)")
        evaluate(model, loaders["all"], "All data", mode="router", num_classes=loaders["num_classes"])
        evaluate_per_range(model, loaders["per_range"], tag="All data per-range", mode="router", num_classes=loaders["num_classes"])
        evaluate(model, loaders["train"], "Train", mode="router", num_classes=loaders["num_classes"])
        evaluate(model, loaders["val"], "Val", mode="router", num_classes=loaders["num_classes"])
        evaluate(model, loaders["test"], "Test", mode="router", num_classes=loaders["num_classes"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
