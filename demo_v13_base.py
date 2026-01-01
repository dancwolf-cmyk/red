# -*- coding: utf-8 -*-
"""
demo_v13_base_range2_20.py
--------------------------
目的：
  - 在现有 Task 2 混合运算数据的基础上，单独抽取 result ∈ [2, 20] 的样本，
  - 按 19 个标签 (2..20) 严格均衡地构造 train/val/test：
        每个标签在 train/val/test 中的样本数完全一致，
        标签内按照 10 : 2 : 2 的比例划分（总体约 10%:2%:2%）。
  - 只训练一个 Base 模型（QwenEncoderFrozen + 线性分类头），
    用 val 做 early stopping，最后在 test 上评估泛化。

依赖：
  - 使用与 demo_v13.py 相同的 Qwen3-0.6B 路径与缓存样本文件。
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
AMP_DTYPE = torch.bfloat16

DEFAULT_SEED = 42
BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64
MAX_EPOCHS_BASE = 30
PATIENCE_BASE = 5

MODEL_PATH = r"c:/temp/lunwen/Qwen3-0.6B"
SAVE_DIR = "checkpoints_v13_base_2_20"
LOG_FILE = Path(SAVE_DIR) / "demo_v13_base_2_20.log"
SAMPLES_CACHE_FILE = Path("checkpoints_v13") / "demo_v13_samples_mixed.json"  # 复用 v13 的缓存文件


# ========================
# 日志 & 随机种子
# ========================

def setup_logger():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("demo_v13_base_2_20")
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
# 数据加载：复用 demo_v13 的缓存样本
# ========================

def load_samples_from_cache(path: Path = SAMPLES_CACHE_FILE):
    if not path.exists():
        raise FileNotFoundError(
            "找不到缓存样本文件：{}\n请先运行原版 demo_v13.py 生成 mixed arithmetic 数据。".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 原结构：{"samples": [...], "label_vocab": [...]}
    if isinstance(data, dict) and "samples" in data:
        return data["samples"], data.get("label_vocab")
    return data, None


def build_range_2_20_balanced_splits(
    samples: List[Dict],
    train_ratio: int = 10,
    val_ratio: int = 2,
    test_ratio: int = 2,
    seed: int = DEFAULT_SEED,
):
    """
    从 mixed arithmetic 总样本中：
      1) 过滤出 domain == "range_2_20" 的样本；
      2) 按标签 y ∈ [2, 20] 分桶；
      3) 对每个标签使用相同的样本数 min_count，按 10:2:2 划分 train/val/test；
         从而保证：
           - 19 个标签在 train/val/test 的样本数完全均衡；
           - 各集合中样本互不重复。
    """
    rng = random.Random(seed)

    # 1) 只保留 2-20 区间的样本
    samples_2_20 = [s for s in samples if s.get("domain") == "range_2_20"]
    logger.info("Total samples with domain=range_2_20: {}".format(len(samples_2_20)))

    # 2) 按标签分桶
    buckets: Dict[str, List[Dict]] = {}
    for s in samples_2_20:
        ans = s["answer"]
        buckets.setdefault(ans, []).append(s)

    # 只保留标签 2..20
    labels = [str(i) for i in range(2, 21)]
    for lab in labels:
        if lab not in buckets:
            raise ValueError("标签 {} 在 range_2_20 中没有样本，请检查数据生成逻辑。".format(lab))

    # 3) 求每个标签可用的最小样本数，做严格均衡
    counts = [len(buckets[lab]) for lab in labels]
    min_count = min(counts)
    logger.info("Per-label min_count in [2,20]: {} (counts={})".format(min_count, counts))

    total_ratio = train_ratio + val_ratio + test_ratio
    per_label_train = max(1, min_count * train_ratio // total_ratio)
    per_label_val = max(1, min_count * val_ratio // total_ratio)
    per_label_test = max(1, min_count - per_label_train - per_label_val)

    logger.info(
        "Per-label split: train={} val={} test={} (sum={})".format(
            per_label_train, per_label_val, per_label_test,
            per_label_train + per_label_val + per_label_test
        )
    )

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for lab in labels:
        bucket = buckets[lab][:]
        rng.shuffle(bucket)
        if len(bucket) < per_label_train + per_label_val + per_label_test:
            raise ValueError(
                "标签 {} 样本数不足 {}，当前只有 {}，请检查。".format(
                    lab, per_label_train + per_label_val + per_label_test, len(bucket)
                )
            )
        t = bucket[:per_label_train]
        v = bucket[per_label_train: per_label_train + per_label_val]
        te = bucket[per_label_train + per_label_val: per_label_train + per_label_val + per_label_test]
        train_samples.extend(t)
        val_samples.extend(v)
        test_samples.extend(te)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    logger.info(
        "Final balanced splits (range_2_20): train={} val={} test={}".format(
            len(train_samples), len(val_samples), len(test_samples)
        )
    )

    return train_samples, val_samples, test_samples, labels


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
            "label": torch.tensor(label, dtype=torch.long),
        }


# ========================
# Qwen encoder + Base 模型
# ========================

class QwenEncoderFrozen(nn.Module):
    def __init__(self, model_path, max_len=64, trainable_last_n_layers=1):
        super().__init__()

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = AMP_DTYPE

        try:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            model_kwargs.pop("torch_dtype", None)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)

        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        for p in self.model.parameters():
            p.requires_grad = False
        self.unfreeze_last_layers(trainable_last_n_layers)

        if torch.cuda.is_available():
            try:
                self.model = self.model.to(dtype=AMP_DTYPE)
            except Exception as e:
                logger.warning("=> Failed to cast encoder to {}: {}".format(AMP_DTYPE, e))

    def unfreeze_last_layers(self, n: int):
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers

        if layers is not None:
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


class BaseClassifier(nn.Module):
    def __init__(self, encoder: QwenEncoderFrozen, num_classes: int):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size
        self.head = nn.Linear(H, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask)
        logits = self.head(h)
        return logits


# ========================
# 训练 & 评估
# ========================

def _get_accum_steps(loader: DataLoader, virtual_batch_size: int):
    unit = getattr(loader, "batch_size", 1) or 1
    return max(1, virtual_batch_size // unit)


def train_base(
    model: BaseClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = MAX_EPOCHS_BASE,
    lr: float = 5e-5,
    virtual_batch_size: int = VIRTUAL_BATCH_SIZE,
    patience: int = PATIENCE_BASE,
):
    # 只训练：head + encoder 中 requires_grad=True 的层
    params = list(model.head.parameters()) + [
        p for p in model.encoder.parameters() if p.requires_grad
    ]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    accum_steps = _get_accum_steps(train_loader, virtual_batch_size)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, max_epochs + 1):
        # ============================
        # 1) 训练一个 epoch（Train）
        # ============================
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        iters = 0
        grad_step = 0

        pbar = tqdm(
            train_loader,
            desc="[Base-2-20][Train] epoch {}/{}".format(epoch, max_epochs),
            ncols=120,
        )
        optim.zero_grad()

        for batch in pbar:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    logits = model(ids, mask)
                    loss = ce(logits, labels) / float(accum_steps)
            else:
                logits = model(ids, mask)
                loss = ce(logits, labels) / float(accum_steps)

            loss.backward()
            grad_step += 1

            if grad_step % accum_steps == 0:
                optim.step()
                optim.zero_grad()

            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

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

        avg_train_loss = total_loss / float(iters if iters > 0 else 1)
        avg_train_acc = total_acc / float(iters if iters > 0 else 1)
        logger.info(
            "=> [Base-2-20] epoch {} train_loss={:.4f} train_acc={:.4f}".format(
                epoch, avg_train_loss, avg_train_acc
            )
        )

        # ============================
        # 2) 立刻在 Val 上做一轮 eval
        # ============================
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader)
        logger.info(
            "=> [Base-2-20] epoch {} VAL loss={:.4f} acc={:.4f} macro_f1={:.4f}".format(
                epoch, val_loss, val_acc, val_macro_f1
            )
        )

        # ============================
        # 3) Early Stopping（基于 val_loss）
        # ============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(
                    "=> Early stopping at epoch {} (patience={})".format(
                        epoch, patience
                    )
                )
                break

    # 还原到最优 val epoch 的参数，并保存 ckpt
    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path = Path(SAVE_DIR) / "base_2_20_best_ep{}.pt".format(best_epoch)
        torch.save({"state_dict": best_state, "epoch": best_epoch}, ckpt_path)
        logger.info("=> Saved best base checkpoint to {}".format(ckpt_path))

    return best_epoch, best_val_loss



@torch.no_grad()
def evaluate(model: BaseClassifier, loader: DataLoader):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="[Eval-2-20]", ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logits = model(ids, mask)
                loss = ce(logits, labels)
        else:
            logits = model(ids, mask)
            loss = ce(logits, labels)

        preds = logits.argmax(dim=-1)

        bs = labels.size(0)
        total_loss += loss.item() * float(bs)
        total_correct += (preds == labels).sum().item()
        total_count += bs

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        avg_loss = total_loss / float(total_count if total_count > 0 else 1)
        avg_acc = total_correct / float(total_count if total_count > 0 else 1)
        pbar.set_postfix(loss="{:.4f}".format(avg_loss), acc="{:.4f}".format(avg_acc))

    avg_loss = total_loss / float(total_count if total_count > 0 else 1)
    avg_acc = total_correct / float(total_count if total_count > 0 else 1)

    if all_preds:
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        num_classes = int(labels_cat.max().item()) + 1
        f1s = []
        for c in range(num_classes):
            tp = ((preds_cat == c) & (labels_cat == c)).sum().item()
            fp = ((preds_cat == c) & (labels_cat != c)).sum().item()
            fn = ((preds_cat != c) & (labels_cat == c)).sum().item()
            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else (2.0 * tp) / float(denom)
            f1s.append(f1)
        macro_f1 = sum(f1s) / float(num_classes if num_classes > 0 else 1)
    else:
        macro_f1 = 0.0

    return avg_loss, avg_acc, macro_f1


# ========================
# 主流程
# ========================

def main():
    set_seed(DEFAULT_SEED)

    # 1) 加载原 demo_v13 的 samples
    samples, _ = load_samples_from_cache(SAMPLES_CACHE_FILE)
    logger.info("Loaded mixed-arithmetic samples from cache (n={})".format(len(samples)))

    # 2) 构造只含 range_2_20 的均衡 train/val/test
    train_samples, val_samples, test_samples, label_vocab_2_20 = build_range_2_20_balanced_splits(
        samples,
        train_ratio=10,
        val_ratio=2,
        test_ratio=2,
        seed=DEFAULT_SEED + 1,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def make_loader(sub_samples, shuffle, seed_offset):
        ds = QADataset(sub_samples, tokenizer, label_vocab_2_20)
        gen = torch.Generator()
        gen.manual_seed(DEFAULT_SEED + seed_offset)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)

    train_loader = make_loader(train_samples, shuffle=True, seed_offset=0)
    val_loader = make_loader(val_samples, shuffle=False, seed_offset=1)
    test_loader = make_loader(test_samples, shuffle=False, seed_offset=2)

    logger.info(
        "Final dataset sizes (range_2_20): train={} val={} test={}".format(
            len(train_samples), len(val_samples), len(test_samples)
        )
    )
    logger.info("Label vocab (2-20): {}".format(label_vocab_2_20))

    # 3) 构建 encoder + base 模型
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = BaseClassifier(encoder, num_classes=len(label_vocab_2_20)).to(DEVICE)

    # 4) 训练 base（只在 2-20 上）
    best_epoch, best_val_loss = train_base(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=MAX_EPOCHS_BASE,
        lr=5e-5,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        patience=PATIENCE_BASE,
    )
    logger.info("Best base epoch={} best_val_loss={:.4f}".format(best_epoch, best_val_loss))

    # 5) 在 test 上评估最终泛化
    test_loss, test_acc, test_macro_f1 = evaluate(model, test_loader)
    logger.info(
        "[Final Test] Base-2-20 loss={:.4f} acc={:.4f} macro_f1={:.4f}".format(
            test_loss, test_acc, test_macro_f1
        )
    )


if __name__ == "__main__":
    main()
