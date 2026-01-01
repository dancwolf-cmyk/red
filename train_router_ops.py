import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from train_op_expert_v13 import ModularAddModelWithRouter, QwenEncoderFrozen, LABEL_VOCAB, load_checkpoint_file

CONFIG_PATH = Path("config.json")


def read_config() -> Dict[str, str]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found at {}".format(CONFIG_PATH))
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class RouterDomainDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_len):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v for k, v in enc.items()}
    enc["labels"] = labels
    return enc


def split_samples(samples: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    random.Random(seed).shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    return samples[:split], samples[split:]


def build_domain_samples(add_path: Path, sub_path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    domain_to_id = {"add": 0, "sub": 1}
    samples = []

    def load_and_tag(path: Path, domain: str):
        data = load_jsonl(path)
        for sample in data:
            op = sample.get("operation") or sample.get("op")
            if op and op not in domain_to_id:
                continue
            samples.append(
                {
                    "text": sample.get("question") or sample.get("expr"),
                    "label": domain_to_id[domain],
                }
            )

    load_and_tag(add_path, "add")
    load_and_tag(sub_path, "sub")
    if not samples:
        raise RuntimeError("No samples loaded for router training")
    return samples, domain_to_id


def remap_expert_weights(state_dict: Dict[str, torch.Tensor], src_idx: int, dst_idx: int) -> Dict[str, torch.Tensor]:
    prefix = f"experts.{src_idx}."
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            remapped[f"experts.{dst_idx}.{suffix}"] = value
    return remapped


def _drop_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in prefixes)
    }


def _select_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if any(k.startswith(prefix) for prefix in prefixes)
    }


def load_experts(
    model: nn.Module,
    add_ckpt: Path,
    sub_ckpt: Path,
    base_ckpt: Path,
) -> None:
    base_sd, _ = load_checkpoint_file(base_ckpt)
    base_sd = _drop_prefixes(base_sd, ["router."])
    model.load_state_dict(base_sd, strict=False)

    add_sd, _ = load_checkpoint_file(add_ckpt)
    expert_add = _select_prefixes(add_sd, ["experts.0."])
    model.load_state_dict(expert_add, strict=False)

    sub_sd, _ = load_checkpoint_file(sub_ckpt)
    expert_sub = _select_prefixes(sub_sd, ["experts.0."])
    remapped = remap_expert_weights(expert_sub, 0, 1)
    model.load_state_dict(remapped, strict=False)


def train_router(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_dir: Path,
    patience: int = 3,
) -> None:
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "router_best.pt"
    last_ckpt_path = save_dir / "router_last.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    no_improve_epochs = 0
    last_val_loss = float("inf")

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        best_val_loss = ckpt.get("val_loss", best_val_loss)

    if last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        model.router.load_state_dict(ckpt["router_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = min(best_val_loss, ckpt.get("best_val_loss", best_val_loss))
        no_improve_epochs = ckpt.get("no_improve_epochs", 0)
        print("[Info] Resuming router training from epoch", start_epoch)

    for epoch in range(1, epochs + 1):
        if epoch < start_epoch:
            continue
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                h = model.encoder(input_ids, attention_mask)
                h = h.to(model.router.weight.dtype)

            logits = model.router(h)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        val_loss, val_acc = evaluate_router(model, val_loader, device)
        best = val_loss < best_val_loss
        if best:
            best_val_loss = val_loss
            torch.save(
                {
                    "router_state": model.router.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        last_val_loss = val_loss

        if no_improve_epochs >= patience:
            print(f"[Info] Early stopping triggered (no improvement in val_loss for {patience} epochs)")
            break

        print(
            f"[Epoch {epoch}] loss={avg_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + (f" (best)" if best else "")
        )

    torch.save(
        {
            "router_state": model.router.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "no_improve_epochs": no_improve_epochs,
            "val_loss": last_val_loss,
        },
        last_ckpt_path,
    )


def evaluate_router(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            h = model.encoder(input_ids, attention_mask)
            h = h.to(model.router.weight.dtype)
            logits = model.router(h)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train router over add/sub experts")
    parser.add_argument("--add-data", type=Path, default=Path("train_ops_balancedplus.jsonl"))
    parser.add_argument("--sub-data", type=Path, default=Path("train_ops_balancedsub.jsonl"))
    parser.add_argument("--add-ckpt", type=Path, default=Path("checkpoints_v13_op_expert/demo_v13_op_add_expert_best.pt"))
    parser.add_argument("--sub-ckpt", type=Path, default=Path("checkpoints_v13_op_expert/demo_v13_op_sub_expert_best.pt"))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints_v13_router"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    cfg = read_config()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples, domain_to_id = build_domain_samples(args.add_data, args.sub_data)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    train_ds = RouterDomainDataset(train_samples)
    val_ds = RouterDomainDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_len),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_len),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = QwenEncoderFrozen(cfg["model_path"], trainable_last_n_layers=3).to(device)
    model = ModularAddModelWithRouter(
        encoder,
        num_classes=len(LABEL_VOCAB),
        num_experts=2,
    ).to(device)

    load_experts(model, args.add_ckpt, args.sub_ckpt, Path(cfg["base_ckpt"]))

    # Freeze all but router
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("router.")

    train_router(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
