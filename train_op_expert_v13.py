# -*- coding: utf-8 -*-
"""
train_op_expert_v13.py
----------------------
在 v13 的 base 上，单独训练一个“运算类型专家”（加法 or 减法）：

  - encoder + base_head：从 demo_v13 的 base ckpt 载入并全部冻结
  - expert：1 个 MLP expert（结构和 v13 一致），只训练这一块参数
  - 输出空间仍然是 answer ∈ [2..99] 的分类，保持和 v11/v13 一致

数据格式（你自己生成）：
  JSONL，每行至少包含:
    {
      "question": "12 plus 7",
      "answer": 19
    }
  如果你习惯用 "expr" 字段，也支持，会自动映射成 question。

通过修改：
  - DATA_PATH：指定“加法数据”或“减法数据”
  - RUN_TAG：改成 "add_expert" / "sub_expert"
来区分不同专家。

依赖：
  - 本目录下有 config.json，提供 model_path 和 base_ckpt
"""

import json
import math
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging

# =============== 日志 ===============
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =============== 脚本内可变配置（每个脚本各不相同） ===============

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 这里用 bfloat16，如果显卡不支持，会在 QwenEncoderFrozen 里自动 fallback
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 64
MAX_EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 50
MAX_LEN = 32
VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 3
SEED = 2025

# 你针对加法 / 减法生成的样本路径
DATA_PATH = "./train_ops_balancedplus.jsonl"   # 改成你的文件，比如 train_sub_2_99.jsonl

# 保存目录和当前 run 的名字
SAVE_DIR = Path("./checkpoints_v13_op_expert")
RUN_TAG = "add_expert"                 # "sub_expert" 等


# =============== 从 config.json 读取机器相关路径 ===============

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError("config.json not found at {}".format(CONFIG_PATH))

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

MODEL_PATH = cfg["model_path"]
BASE_CKPT = cfg["base_ckpt"]

SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger.info("MODEL_PATH = %s", MODEL_PATH)
logger.info("BASE_CKPT = %s", BASE_CKPT)


# =============== 数据集定义：简化版 QADataset（和 v13 label 逻辑兼容） ===============

LABEL_VOCAB = [str(i) for i in range(2, 101)]  # 2..99，保持和 v11/v13 一致


class SimpleQADataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_len: int = 32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_id = {a: i for i, a in enumerate(LABEL_VOCAB)}
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # question 字段兼容 expr 命名
        q = s.get("question")
        if q is None:
            q = s.get("expr")
        if q is None:
            raise KeyError("Sample missing 'question' or 'expr' field: {}".format(s))
        s["question"] = q

        ans = s["answer"]
        if not isinstance(ans, str):
            ans = str(ans)
            s["answer"] = ans

        if ans not in self.label_to_id:
            raise KeyError("Answer {} not in LABEL_VOCAB".format(ans))
        label = self.label_to_id[ans]

        enc = self.tokenizer(
            q,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
        }


def load_samples_from_jsonl(path: str) -> List[Dict]:
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


def split_train_val(samples: List[Dict], val_ratio: float, seed: int = 42):
    n = len(samples)
    n_val = max(1, int(n * val_ratio))
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = set(perm[:n_val])
    train = [samples[i] for i in range(n) if i not in val_idx]
    val = [samples[i] for i in range(n) if i in val_idx]
    return train, val


def load_checkpoint_file(path: Path):
    """
    Load checkpoint supporting both plain state_dict and wrapped dicts with metadata.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("val_loss")
    return obj, None


# =============== Qwen encoder & RED 模型（结构保持和 v13 一致） ===============

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
            # 兼容老版本 transformers 没有 torch_dtype 的情况
            model_kwargs.pop("torch_dtype", None)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)

        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        # 先全部冻结，再解锁最后 n 层（和 demo_v13 一样）:contentReference[oaicite:1]{index=1}
        for p in self.model.parameters():
            p.requires_grad = False
        self.unfreeze_last_layers(trainable_last_n_layers)

        if torch.cuda.is_available():
            try:
                self.model = self.model.to(dtype=AMP_DTYPE)
            except Exception as e:
                logger.warning("=> Failed to cast encoder to %s: %s", AMP_DTYPE, e)

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
        lengths = attention_mask.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(h.size(0), device=h.device)
        pooled = h[batch_idx, last_idx]
        return pooled


class ModularAddModelWithRouter(nn.Module):
    """
    RED 结构：base_head + MLP experts + router（这里我们只用 num_experts=1）
    结构保持和 demo_v13 一致，确保可以从 base ckpt 加载。:contentReference[oaicite:2]{index=2}
    """
    def __init__(self, encoder, num_classes, num_experts=5):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size

        # 强一点的 base head（v13 升级版）
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
        target_dtype = self.base_head[0].weight.dtype
        if h.dtype != target_dtype:
            h = h.to(target_dtype)
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


# =============== 训练 & 验证 ===============

def make_loader(dataset, batch_size, shuffle, seed_offset=0):
    gen = torch.Generator()
    gen.manual_seed(SEED + seed_offset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_one_epoch(model, loader, optimizer, criterion, accum_steps=1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    grad_step = 0

    pbar = tqdm(loader, desc="[Train]", ncols=120)
    optimizer.zero_grad()

    for i, batch in enumerate(pbar, start=1):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(ids, mask, mode="all_fixed")  # base + expert residual
        loss = criterion(logits, labels) / float(accum_steps)
        loss.backward()
        grad_step += 1

        if MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                MAX_GRAD_NORM,
            )

        if grad_step >= accum_steps:
            optimizer.step()
            optimizer.zero_grad()
            grad_step = 0

        total_loss += loss.item() * labels.size(0) * float(accum_steps)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        if i % LOG_INTERVAL == 0:
            avg_loss = total_loss / max(total_count, 1)
            acc = total_correct / max(total_count, 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, mode="all_fixed"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="[Val]", ncols=120)
    for batch in pbar:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(ids, mask, mode=mode)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main():
    # 1) 加载数据
    all_samples = load_samples_from_jsonl(DATA_PATH)
    train_samples, val_samples = split_train_val(all_samples, VAL_RATIO, seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    train_ds = SimpleQADataset(train_samples, tokenizer, max_len=MAX_LEN)
    val_ds = SimpleQADataset(val_samples, tokenizer, max_len=MAX_LEN)

    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True, seed_offset=0)
    val_loader = make_loader(val_ds, BATCH_SIZE, shuffle=False, seed_offset=1)

    # 2) 构建模型：num_experts=1，即一个“运算类型专家”
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=3).to(DEVICE)
    model = ModularAddModelWithRouter(
        encoder,
        num_classes=len(LABEL_VOCAB),
        num_experts=1,
    ).to(DEVICE)

    best_ckpt_path = SAVE_DIR / f"demo_v13_op_{RUN_TAG}_best.pt"
    last_ckpt_path = SAVE_DIR / f"demo_v13_op_{RUN_TAG}_last.pt"

    resume_path = None
    resumed_val_loss = None
    if last_ckpt_path.exists():
        resume_path = last_ckpt_path
    elif best_ckpt_path.exists():
        resume_path = best_ckpt_path

    if resume_path is not None:
        state_dict, resumed_val_loss = load_checkpoint_file(resume_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed weights from %s. missing=%d, unexpected=%d", resume_path, len(missing), len(unexpected))
    else:
        ckpt_obj = torch.load(BASE_CKPT, map_location="cpu")
        state_dict = ckpt_obj.get("state_dict", ckpt_obj)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded base ckpt. missing=%d, unexpected=%d", len(missing), len(unexpected))

    # 4) 冻结 encoder + base_head，只训练 expert[0] 及 router_scale（你也可以把 router 都关掉）
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.base_head.parameters():
        p.requires_grad = False
    # 如果你只想训练 expert，不想动 router_scale，也可以把下面这行注释掉
    # model.router_scale.requires_grad = False

    # 当前结构下只有一个 expert[0]，默认 requires_grad=True
    for i, exp in enumerate(model.experts):
        if i == 0:
            for p in exp.parameters():
                p.requires_grad = True
        else:
            for p in exp.parameters():
                p.requires_grad = False

    # 5) 优化器只看 requires_grad=True 的参数
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    if best_ckpt_path.exists():
        if resume_path == best_ckpt_path and resumed_val_loss is not None:
            best_val_loss = resumed_val_loss
        else:
            _, stored_val = load_checkpoint_file(best_ckpt_path)
            if stored_val is not None:
                best_val_loss = stored_val
    no_improve_epochs = 0
    last_val_loss = float("inf")

    logger.info("Start training op expert (%s), total epochs=%d", RUN_TAG, MAX_EPOCHS)
    for epoch in range(1, MAX_EPOCHS + 1):
        logger.info("Epoch %d", epoch)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            accum_steps=max(1, VIRTUAL_BATCH_SIZE // BATCH_SIZE),
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, mode="all_fixed")
        logger.info(
            "[Epoch %d] train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
            epoch, train_loss, train_acc, val_loss, val_acc,
        )

        last_val_loss = val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_ckpt_path.exists():
                best_ckpt_path.unlink()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            logger.info("=> New best model saved to %s", best_ckpt_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping triggered (val loss not improved for %d epochs)", EARLY_STOP_PATIENCE)
                break

    # 保存一份 last
    torch.save(
        {
            "state_dict": model.state_dict(),
            "val_loss": last_val_loss,
        },
        last_ckpt_path,
    )
    logger.info("=> Last model saved to %s", last_ckpt_path)


if __name__ == "__main__":
    main()
