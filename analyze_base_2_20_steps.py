# -*- coding: utf-8 -*-
"""
analyze_base_2_20_steps.py
--------------------------
功能：
  - 读取 demo_v13 的缓存样本 demo_v13_samples_mixed.json
  - 构造 2-20 区间的均衡 train/val/test（逻辑与 base_2_20 训练版一致）
  - 加载 checkpoints_v13_base_2_20/base_2_20_best_ep29.pt
  - 在 val / test 上评估，并统计：
        - overall acc
        - 2 步 / 3 步 / 4 步表达式的错误数与错误率
"""

from pathlib import Path
import json
import logging
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

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

# 这里沿用 demo_v13 的模型和样本缓存路径
MODEL_PATH = r"c:/temp/lunwen/Qwen3-0.6B"   # 如果你本地路径不同，改一下这一行
SAMPLES_CACHE_FILE = Path("checkpoints_v13") / "demo_v13_samples_mixed.json"

# base_2_20 的 checkpoint 目录
BASE_2_20_DIR = Path("checkpoints_v13_base_2_20")
BASE_2_20_CKPT = BASE_2_20_DIR / "base_2_20_best_ep29.pt"

LOG_FILE = BASE_2_20_DIR / "analyze_base_2_20_steps.log"

LABEL_VOCAB_2_20 = [str(i) for i in range(2, 21)]  # "2"..."20"


# ========================
# 日志 & 随机种子
# ========================

def setup_logger():
    BASE_2_20_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analyze_base_2_20")
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
# Qwen encoder（与 demo_v13 保持一致）
# ========================

class QwenEncoderFrozen(nn.Module):
    def __init__(self, model_path, max_len=64, trainable_last_n_layers=1):
        super().__init__()

        model_kwargs = {
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            # 注意：torch_dtype 已经在新版本 transformers 里被标注 deprecated，
            # 但这里仍然兼容使用；控制台会有 warning，可以忽略。
            model_kwargs["torch_dtype"] = AMP_DTYPE

        try:
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            # 兼容旧版本 transformers
            model_kwargs.pop("torch_dtype", None)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)

        self.hidden_size = self.model.config.hidden_size
        self.max_len = max_len

        # 先全部冻结，再解锁最后 n 层
        for p in self.model.parameters():
            p.requires_grad = False
        self.unfreeze_last_layers(trainable_last_n_layers)

        # 统一 dtype
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
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ========================
# Base 2-20 分类模型
# ========================

class Base220Classifier(nn.Module):
    """
    假设 base_2_20 训练时的结构就是：
      encoder(Qwen) + dropout + Linear(hidden_size -> 19 labels)
    如果你原来训练脚本里有不同名字（比如 self.fc / self.classifier），
    也没关系，后面会用“按维度匹配”的方式自动找分类层参数。
    """
    def __init__(self, encoder: QwenEncoderFrozen, num_classes: int):
        super().__init__()
        self.encoder = encoder
        H = encoder.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(H, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids, attention_mask)
        h = self.dropout(h)
        # encoder 可能输出 bfloat16，这里统一转成 float32 再进分类头，避免 dtype 冲突
        logits = self.classifier(h.float())
        return logits


# ========================
# 数据集 & DataLoader
# ========================

class Range220Dataset(Dataset):
    """
    专门给 2-20 区间用的数据集：
      - text: 题目
      - label: 2-20 映射到 0-18
      - steps: 2步/3步/4步（根据 question 解析）
    """
    def __init__(self, samples: List[Dict], tokenizer, label_vocab: List[str], max_len: int = 32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_id = {a: i for i, a in enumerate(label_vocab)}
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def count_steps_from_question(q: str) -> int:
        """
        根据 question 解析 2步/3步/4步：
        题目格式示例：
          "What is 12 plus 3?"
          "What is 45 minus 17?"
          "What is 12 plus 8 minus 3?"
        简单做法：
          - 去掉 "What is" 和问号
          - 按空格切分，只统计数字 token 的个数 => 步数
        """
        q = q.strip()
        lower = q.lower()
        if lower.startswith("what is "):
            expr = q[8:].strip()
        else:
            # 防御性处理
            if q.endswith("?"):
                expr = q[:-1].strip()
            else:
                expr = q

        if expr.endswith("?"):
            expr = expr[:-1].strip()

        tokens = expr.split()
        numbers = [t for t in tokens if t.lstrip("-").isdigit()]
        steps = len(numbers)
        if steps not in (2, 3, 4):
            steps = -1  # 预防脏数据
        return steps

    def __getitem__(self, idx):
        s = self.samples[idx]
        question = s["question"]
        # answer 在缓存里一般是 int，这里统一转成 str 再查 vocab
        ans_str = str(s["answer"])
        label = self.label_to_id[ans_str]

        enc = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        steps = self.count_steps_from_question(question)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "steps": torch.tensor(steps, dtype=torch.long),
            "question": question,
            "answer": int(ans_str),
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    steps = torch.stack([b["steps"] for b in batch], dim=0)
    # question / answer 用列表返回，方便后续需要 debug
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "steps": steps,
        "questions": questions,
        "answers": answers,
    }


def make_loader(dataset: Dataset, batch_size: int = BATCH_SIZE, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


# ========================
# 构造 2-20 区间的均衡 train/val/test
# ========================

def load_cached_samples(cache_path: Path) -> List[Dict]:
    assert cache_path.exists(), "样本缓存不存在：{}".format(cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # demo_v13_samples_mixed.json 的结构一般是：
    # {"samples": [...], "label_vocab": [...]}
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    else:
        samples = data

    logger.info("=> loaded cached samples from {} (n={})".format(cache_path, len(samples)))
    return samples


def build_range_2_20_balanced_splits(
        samples: List[Dict],
        seed: int = DEFAULT_SEED + 1
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    与 base_2_20 训练脚本的逻辑保持一致：
      - 只保留 domain == "range_2_20"
      - 按标签(2..20)分桶
      - 每个标签取同样数量 min_count
      - 再按 train/val/test 拆分
    base_2_20 日志中显示：
      min_count = 802 -> train=572, val=114, test=116
    这里当 min_count >= 802 时直接使用这三个数，
    否则按 70% / 15% / 15% 比例缩放。
    """
    range_samples = [s for s in samples if s.get("domain") == "range_2_20"]
    logger.info("=> range_2_20 samples: {}".format(len(range_samples)))

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for s in range_samples:
        ans = str(s["answer"])
        if ans in LABEL_VOCAB_2_20:
            buckets[ans].append(s)

    counts = {a: len(buckets[a]) for a in LABEL_VOCAB_2_20}
    logger.info("=> per-label counts (2-20): {}".format(counts))

    min_count = min(counts.values())
    logger.info("=> min_count among 2-20 = {}".format(min_count))

    if min_count >= 802:
        train_per, val_per, test_per = 572, 114, 116
    else:
        train_per = int(min_count * 0.7)
        val_per = int(min_count * 0.15)
        test_per = min_count - train_per - val_per

    logger.info(
        "=> per-label split: train={} val={} test={} (sum={})".format(
            train_per, val_per, test_per, train_per + val_per + test_per
        )
    )

    rand = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []

    for ans in LABEL_VOCAB_2_20:
        bucket = buckets[ans]
        rand.shuffle(bucket)
        train_samples.extend(bucket[:train_per])
        val_samples.extend(bucket[train_per:train_per + val_per])
        test_samples.extend(bucket[train_per + val_per:train_per + val_per + test_per])

    logger.info("=> final sizes: train={} val={} test={}".format(
        len(train_samples), len(val_samples), len(test_samples))
    )
    return train_samples, val_samples, test_samples


# ========================
# 加载 checkpoint（带分类层参数自动匹配）
# ========================

def load_base_2_20_checkpoint(model: Base220Classifier, ckpt_path: Path):
    assert ckpt_path.exists(), "checkpoint 不存在: {}".format(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu")

    # 兼容两种情况：
    # 1) {"state_dict": ..., "epoch": ...}
    # 2) 直接就是 state_dict
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    # 先宽松加载一遍（encoder 那部分参数可以直接对上）
    incompat = model.load_state_dict(state, strict=False)
    missing = getattr(incompat, "missing_keys", [])
    unexpected = getattr(incompat, "unexpected_keys", [])
    logger.info(
        "=> load_state_dict(strict=False) done. missing={} unexpected={}".format(
            len(missing), len(unexpected)
        )
    )

    # 然后按“形状匹配”的方式，把原来 BaseClassifier.head 的权重拷到 classifier 上
    num_labels = model.classifier.out_features
    hidden_size = model.classifier.in_features

    candidate_keys = []
    for k, v in state.items():
        # 只处理 Tensor，防止再对 dict / 其它对象调用 dim()
        if isinstance(v, torch.Tensor) and v.dim() == 2:
            if v.size(0) == num_labels and v.size(1) == hidden_size:
                candidate_keys.append(k)

    if not candidate_keys:
        logger.warning(
            "=> 没有在 checkpoint 里找到形状为 ({}, {}) 的线性层权重，"
            "可能无法自动恢复分类头。".format(num_labels, hidden_size)
        )
        return

    head_w_key = candidate_keys[0]
    head_b_key = head_w_key.replace("weight", "bias")

    with torch.no_grad():
        logger.info(
            "=> manually loading classifier from keys: {}, {}".format(
                head_w_key, head_b_key
            )
        )
        model.classifier.weight.copy_(state[head_w_key])
        if head_b_key in state and isinstance(state[head_b_key], torch.Tensor):
            model.classifier.bias.copy_(state[head_b_key])
        else:
            logger.warning(
                "=> bias key {} not found or not Tensor; keep current bias.".format(
                    head_b_key
                )
            )



# ========================
# 评估 & 2步/3步/4步错误分布
# ========================

def analyze_split(
        split_name: str,
        model: Base220Classifier,
        data_loader: DataLoader
):
    model.eval()
    total = 0
    correct = 0

    step_total = Counter()   # {2: n, 3: n, 4: n, -1: n}
    step_errors = Counter()  # {2: err_n, ...}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"{split_name}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            steps = batch["steps"].cpu().tolist()

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            preds_cpu = preds.cpu().tolist()
            labels_cpu = labels.cpu().tolist()

            for y, p, st in zip(labels_cpu, preds_cpu, steps):
                total += 1
                step_total[st] += 1
                if y == p:
                    correct += 1
                else:
                    step_errors[st] += 1

    acc = correct / max(total, 1)
    logger.info("=> [{}] overall acc = {:.4f} ({}/{})".format(
        split_name, acc, correct, total)
    )

    for st in sorted(step_total.keys()):
        tot = step_total[st]
        err = step_errors.get(st, 0)
        err_rate = err / max(tot, 1)
        logger.info("=> [{}] steps={} : total={} errors={} err_rate={:.4f}".format(
            split_name, st, tot, err, err_rate)
        )


# ========================
# main
# ========================

def main():
    set_seed(DEFAULT_SEED)
    logger.info("=> DEVICE = {}".format(DEVICE))

    # 1) 读取缓存样本
    samples = load_cached_samples(SAMPLES_CACHE_FILE)

    # 2) 构造 2-20 区间的 train/val/test
    train_samples, val_samples, test_samples = build_range_2_20_balanced_splits(samples)

    # 3) tokenizer & Dataset & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    val_dataset = Range220Dataset(val_samples, tokenizer, LABEL_VOCAB_2_20, max_len=32)
    test_dataset = Range220Dataset(test_samples, tokenizer, LABEL_VOCAB_2_20, max_len=32)

    val_loader = make_loader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4) 构建模型 + 加载 checkpoint
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = Base220Classifier(encoder, num_classes=len(LABEL_VOCAB_2_20)).to(DEVICE)

    load_base_2_20_checkpoint(model, BASE_2_20_CKPT)

    # 5) 分析 val / test 的错误分布
    logger.info("=== Analyze VAL split (2-20, 按 2/3/4 步统计) ===")
    analyze_split("Val-2_20", model, val_loader)

    logger.info("=== Analyze TEST split (2-20, 按 2/3/4 步统计) ===")
    analyze_split("Test-2_20", model, test_loader)


if __name__ == "__main__":
    main()
