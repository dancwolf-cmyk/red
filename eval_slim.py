"""
Evaluate a slim checkpoint (created by shrink_checkpoint.py) on train/val/test with
accuracy and macro-F1, logging to file and showing tqdm progress bars.

Example:
py eval_slim.py --ckpt checkpoints/demo_v11_router_best_ep60.pt.slim.pt --model-path /home/cwadmin/lunwen/Qwen3-0.6B
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from demo_v11 import (
    BATCH_SIZE,
    TEST_RATIO,
    TOTAL_SAMPLES,
    VAL_RATIO,
    DEVICE,
    DEFAULT_SEED,
    load_or_create_samples,
    set_seed,
    QADataset,
)
from shrink_checkpoint import load_model_with_slim_checkpoint
from transformers import AutoTokenizer


LOG_FILE = Path("checkpoints") / "eval_slim.log"


def setup_logger():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_slim")
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
def evaluate(model, loader, mode: str, desc: str, num_classes: int):
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=desc, ncols=120)
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

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = correct / total if total else 0.0
    f1 = macro_f1(preds, labels, num_classes)
    logger.info(f"[{desc}] mode={mode} acc={acc:.4f} macro_f1={f1:.4f} n={total}")
    return acc, f1


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
    train_loader = DataLoader(QADataset(train_samples, tokenizer, label_vocab), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(QADataset(val_samples, tokenizer, label_vocab), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(QADataset(test_samples, tokenizer, label_vocab), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, len(label_vocab)


def main():
    parser = argparse.ArgumentParser(description="Evaluate slim checkpoint with acc and macro-F1.")
    parser.add_argument("--ckpt", required=True, help="Path to slim checkpoint (.slim.pt).")
    parser.add_argument("--model-path", default=None, help="Path to base model (Qwen). Overrides demo_v11.MODEL_PATH.")
    parser.add_argument("--trainable-last-n-layers", type=int, default=1,
                        help="Must match how slim checkpoint was created.")
    parser.add_argument("--modes", nargs="+", default=["base", "all_fixed", "router"],
                        help="Which model modes to evaluate.")
    args = parser.parse_args()

    from demo_v11 import MODEL_PATH as DEFAULT_MODEL_PATH  # local import to avoid circular issues

    base_model_path = args.model_path or DEFAULT_MODEL_PATH
    train_loader, val_loader, test_loader, num_classes = build_loaders(base_model_path)

    model = load_model_with_slim_checkpoint(
        Path(args.ckpt),
        model_path=base_model_path,
        num_classes=num_classes,
        num_experts=5,
        trainable_last_n_layers=args.trainable_last_n_layers,
    ).to(DEVICE)

    logger.info(f"Loaded model from {args.ckpt}; evaluating modes: {args.modes}")

    for mode in args.modes:
        evaluate(model, train_loader, mode, desc=f"Train ({mode})", num_classes=num_classes)
        evaluate(model, val_loader, mode, desc=f"Val ({mode})", num_classes=num_classes)
        evaluate(model, test_loader, mode, desc=f"Test ({mode})", num_classes=num_classes)

    logger.info("Evaluation finished.")


if __name__ == "__main__":
    main()
