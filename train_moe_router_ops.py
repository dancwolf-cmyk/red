import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 注意：这里改成从 demo_v13 引入 MoE 模型
from demo_v13 import ModularAddModelWithMoE
# 其余和 train_router_ops 一致，从 train_op_expert_v13 引入
from train_op_expert_v13 import QwenEncoderFrozen, LABEL_VOCAB, load_checkpoint_file

CONFIG_PATH = Path("config.json")


# -------------------------
# 通用工具
# -------------------------

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


# -------------------------
# MoE 训练用数据集
# -------------------------

class MoEDataset(Dataset):
    """
    每条样本：
      {
        "text": "What is 18 minus 9 plus 47?",
        "label": <int class idx>   # 对应 LABEL_VOCAB
      }
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_len: int):
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


def split_samples(
    samples: List[Dict],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rnd = random.Random(seed)
    rnd.shuffle(samples)
    split = int(len(samples) * (1.0 - val_ratio))
    return samples[:split], samples[split:]


def build_moe_samples(add_path: Path, sub_path: Path) -> List[Dict]:
    """
    从加法/减法 jsonl 里构造 MoE 训练样本。
    默认 jsonl 结构类似：
      {"question": "What is 18 plus 9?", "answer": 27, "operation": "add"}
    """
    label_to_id = {lbl: i for i, lbl in enumerate(LABEL_VOCAB)}
    samples: List[Dict] = []

    def load_and_convert(path: Path):
        data = load_jsonl(path)
        for sample in data:
            text = sample.get("question") or sample.get("expr")
            if not text:
                continue

            ans = sample.get("answer")
            if ans is None:
                ans = sample.get("label")
            if ans is None:
                continue

            ans_str = str(ans)
            if ans_str not in label_to_id:
                # 不在 LABEL_VOCAB 范围内的丢弃
                continue

            samples.append(
                {
                    "text": text,
                    "label": label_to_id[ans_str],
                }
            )

    load_and_convert(add_path)
    load_and_convert(sub_path)

    if not samples:
        raise RuntimeError("No valid samples for MoE training (check answer range / LABEL_VOCAB)")
    return samples


# -------------------------
# checkpoint 辅助函数（参考 train_router_ops）
# -------------------------

def remap_expert_weights(
    state_dict: Dict[str, torch.Tensor],
    src_idx: int,
    dst_idx: int,
) -> Dict[str, torch.Tensor]:
    prefix = "experts.{}.".format(src_idx)
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            remapped["experts.{}.".format(dst_idx) + suffix] = value
    return remapped


def _drop_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes: List[str],
) -> Dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in prefixes)
    }


def _select_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes: List[str],
) -> Dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if any(k.startswith(prefix) for prefix in prefixes)
    }


def load_experts_for_moe(
    model: nn.Module,
    add_ckpt: Path,
    sub_ckpt: Path,
    base_ckpt: Path,
) -> None:
    """
    1) 先加载 base（去掉旧 router）
    2) 再把 add expert 写到 experts.0
    3) 再把 sub expert 写到 experts.1
    """
    # base
    base_sd, _ = load_checkpoint_file(base_ckpt)
    base_sd = _drop_prefixes(base_sd, ["router."])
    model.load_state_dict(base_sd, strict=False)

    # add expert -> experts.0.*
    add_sd, _ = load_checkpoint_file(add_ckpt)
    expert_add = _select_prefixes(add_sd, ["experts.0."])
    model.load_state_dict(expert_add, strict=False)

    # sub expert -> experts.1.*
    sub_sd, _ = load_checkpoint_file(sub_ckpt)
    expert_sub = _select_prefixes(sub_sd, ["experts.0."])
    remapped = remap_expert_weights(expert_sub, 0, 1)
    model.load_state_dict(remapped, strict=False)


# -------------------------
# MoE 评估
# -------------------------

def evaluate_moe(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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

            # ModularAddModelWithMoE 的 forward 签名：
            # forward(input_ids, attention_mask=None, mode=None, return_gate=False)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# -------------------------
# 训练 MoE gate（传统 MoE）
# -------------------------

def train_moe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_dir: Path,
    patience: int = 3,
) -> None:
    """
    经典 MoE 训练：
      - 冻结 encoder / experts / base_head，只训练 model.router（gate）
      - 损失：CE(logits, labels)，通过 gate 反向传播，更新 router
    """
    # 只训练 router 参数
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("router.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "moe_router_best.pt"
    last_ckpt_path = save_dir / "moe_router_last.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    no_improve_epochs = 0
    last_val_loss = float("inf")

    # 若已存在 last checkpoint，则尝试断点续训
    if last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        no_improve_epochs = ckpt.get("no_improve_epochs", 0)
        last_val_loss = ckpt.get("val_loss", last_val_loss)
        print("[Info] Resuming MoE training from epoch", start_epoch)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        val_loss, val_acc = evaluate_moe(model, val_loader, device)
        best = val_loss < best_val_loss
        if best:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        last_val_loss = val_loss

        print(
            "[Epoch {}] loss={:.4f} train_acc={:.4f} val_loss={:.4f} val_acc={:.4f}{}".format(
                epoch,
                avg_loss,
                train_acc,
                val_loss,
                val_acc,
                " (best)" if best else "",
            )
        )

        if no_improve_epochs >= patience:
            print("[Info] Early stopping MoE (no improvement in val_loss for {} epochs)".format(patience))
            break

        # 保存 last checkpoint
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "no_improve_epochs": no_improve_epochs,
                "val_loss": last_val_loss,
            },
            last_ckpt_path,
        )


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Train classical MoE gate over add/sub experts")
    parser.add_argument("--add-data", type=Path, default=Path("train_ops_balancedplus.jsonl"))
    parser.add_argument("--sub-data", type=Path, default=Path("train_ops_balancedsub.jsonl"))
    parser.add_argument("--add-ckpt", type=Path, default=Path("checkpoints_v13_op_expert/demo_v13_op_add_expert_best.pt"))
    parser.add_argument("--sub-ckpt", type=Path, default=Path("checkpoints_v13_op_expert/demo_v13_op_sub_expert_best.pt"))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints_v13_moe"))
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

    # 1) 构造 MoE 训练数据（加法 + 减法混合，label 是最终答案）
    samples = build_moe_samples(args.add_data, args.sub_data)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    train_ds = MoEDataset(train_samples)
    val_ds = MoEDataset(val_samples)

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

    # 2) 建 MoE 模型，加载 base + add/sub 专家
    encoder = QwenEncoderFrozen(cfg["model_path"], trainable_last_n_layers=3).to(device)
    model = ModularAddModelWithMoE(
        encoder,
        num_classes=len(LABEL_VOCAB),
        num_experts=2,
    ).to(device)

    load_experts_for_moe(
        model,
        args.add_ckpt,
        args.sub_ckpt,
        Path(cfg["base_ckpt"]),
    )

    # 3) 训练 MoE gate
    train_moe(
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
