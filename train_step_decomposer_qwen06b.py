# -*- coding: utf-8 -*-
"""
train_step_decomposer_qwen06b.py
--------------------------------
用 Qwen3-0.6B 微调“拆步器”：
  输入: expr (如 "12 minus 5 plus 3")
  输出: program (如:
        Step 1: 12 + 5 = x1
        Step 2: x1 - 3 = x2
        Answer: x2)

说明:
  - 使用 causal LM 方式训练 (AutoModelForCausalLM)
  - 构造 prompt + program 拼接的输入
  - 只在 program 部分计算 loss，prompt 部分 label = -100
  - 支持虚拟 batch (gradient_accumulation)
  - 显存设定：bfloat16 + 只解冻最后 3 层 + lm_head + gradient checkpointing
  - 已加入断点续训：每 100 次参数更新保存一次 ckpt，可从中恢复
"""

import json
import math
import random
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm


# ===================== 配置区域 =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"e:/dev/lunwen/qwen3-0.6b"  # 你的 Qwen3-0.6B 本地路径
DATA_PATH = r"./mixed_arith_step_decomposer_80k.jsonl"  # 上一步生成的数据
SAVE_DIR = Path("./checkpoints_step_decomposer")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_INTERVAL = 100

MAX_SEQ_len = 256

BATCH_SIZE = 1             # 实际 batch
VIRTUAL_BATCH_SIZE = 64     # 虚拟 batch (gradient_accumulation)
LR = 5e-5
MAX_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 使用 bfloat16（和你之前训练专家的习惯一致），不再配合 GradScaler
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else None

# 随机种子（为了 DataLoader 打乱顺序在重启时尽量可复现）
SEED = 2025

# 断点续训相关
SAVE_EVERY_UPDATES = 40  # 每 100 次 optimizer.step() 保存一次 ckpt
CKPT_PATH = SAVE_DIR / "step_decomposer_ckpt.pt"


# ===================== 数据集定义 =====================

class StepDecomposerDataset(Dataset):
    """
    读取 JSONL:
      {
        "id": ...,
        "expr": "12 minus 5 plus 3",
        "program": "Step 1: ...\nStep 2: ...\nAnswer: x2",
        ...
      }

    训练目标:
      Input: prompt + program
      Label: 在 program 部分的 token 计算 cross-entropy，prompt 部分为 -100
    """

    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer, max_len: int = 256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.samples: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

        print("[Dataset] Loaded", len(self.samples), "samples from", jsonl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        obj = self.samples[idx]
        expr = obj["expr"]
        program = obj["program"]

        # prompt 尽量保持稳定
        prompt = (
            "Decompose the following arithmetic expression into step-by-step computation "
            "using variables x1, x2, ...\n"
            "Expression: " + expr + "\n\n"
            "Program:\n"
        )

        # 拼接: [prompt][program]
        full_text = prompt + program

        # 为了只在 program 上算 loss，记录 program 在 full_text 中的起始字符位置
        program_start_idx = len(prompt)

        return {
            "full_text": full_text,
            "program_start_idx": program_start_idx,
        }


# ===================== Collate 函数 =====================

def collate_fn(batch, tokenizer, max_len: int):
    """
    将一批 sample 转成模型可用的 input_ids / attention_mask / labels

    处理逻辑:
      - 先对 full_text 做一次 batch encode
      - 然后根据 program_start_idx，在 labels 中把 prompt 部分设为 -100
    """
    full_texts = [item["full_text"] for item in batch]
    program_start_idxs = [item["program_start_idx"] for item in batch]

    enc = tokenizer(
        full_texts,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]          # (B, T)
    attention_mask = enc["attention_mask"]

    # 构造 labels
    labels = input_ids.clone()

    labels_list = labels.tolist()
    for i, (full_text, prog_start_char_idx) in enumerate(zip(full_texts, program_start_idxs)):
        # 截取 prompt 部分的原始文本（按字符）
        prompt_text = full_text[:prog_start_char_idx]
        # 对 prompt_text 单独编码，得到 token 的长度
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )["input_ids"][0]  # (L_prompt,)

        prompt_token_len = prompt_ids.size(0)

        # 在当前样本中，将前 prompt_token_len 个位置标签置为 -100
        cur_len = int(attention_mask[i].sum().item())  # 有效 token 数
        cut = min(prompt_token_len, cur_len)
        for t in range(cut):
            labels_list[i][t] = -100

    labels = torch.tensor(labels_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ===================== 主训练逻辑 =====================

def main():
    # 固定随机种子，保证 DataLoader 打乱在重启后可复现（至少顺序一致，便于跳过 step）
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("[Info] Loading tokenizer and model from", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 保证有 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型，直接用 bfloat16 参数
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=AMP_DTYPE if AMP_DTYPE is not None else None,
    )
    model.to(DEVICE)

    # ====== 只解冻最后 3 层 + lm_head，其他全部冻结 ======
    for p in model.parameters():
        p.requires_grad = False

    # 找到 transformer 的 block 列表（不同实现名字不一样）
    blocks = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = model.transformer.blocks

    if blocks is None:
        print("[Warn] Cannot find transformer blocks (layers/blocks). "
              "Only lm_head will be trainable. You may need to inspect model structure.")
    else:
        n_blocks = len(blocks)
        n_unfreeze = min(3, n_blocks)
        print("[Info] Total transformer blocks:", n_blocks, "| Unfreeze last:", n_unfreeze)
        for block in blocks[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

    # 解冻 lm_head
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
        print("[Info] lm_head is trainable.")
    else:
        print("[Warn] model has no lm_head attribute, check architecture.")

    # gradient checkpointing + 关闭 cache 节省显存
    model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    dataset = StepDecomposerDataset(DATA_PATH, tokenizer, max_len=MAX_SEQ_len)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, MAX_SEQ_len),
    )

    # 优化器 & Scheduler：只会对 requires_grad=True 的参数建立 state
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

    num_update_steps_per_epoch = math.ceil(len(train_loader) * BATCH_SIZE / VIRTUAL_BATCH_SIZE)
    t_total = int(num_update_steps_per_epoch * MAX_EPOCHS)
    warmup_steps = int(t_total * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # ====== 断点续训：尝试加载 ckpt ======
    start_epoch = 1
    global_step = 0          # 以 optimizer.step() 次数为单位
    resume_step_in_epoch = 0 # 上一次 epoch 内已完成的 dataloader step 数

    if CKPT_PATH.exists():
        print("[Info] Found checkpoint at", CKPT_PATH, ", loading...")
        # 1. 先全部加载到 CPU，不要直接 map 到 GPU
        ckpt = torch.load(CKPT_PATH, map_location="cpu")

        # 2. 再把 state_dict 拷到已经在 GPU 上的 model / optimizer 里
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 1)
        global_step = ckpt.get("global_step", 0)
        resume_step_in_epoch = ckpt.get("step_in_epoch", 0)

        # 3. 释放 ckpt 占用的内存，并清一下 CUDA cache
        del ckpt
        torch.cuda.empty_cache()

        print(
            "[Info] Resumed from epoch", start_epoch,
            "global_step", global_step,
            "step_in_epoch", resume_step_in_epoch,
        )

    else:
        print("[Info] No checkpoint found, start from scratch.")

    model.train()

    best_loss = float("inf")

    print("[Info] Start training, total updates =", t_total)
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_loss_sum = 0.0
        epoch_tokens = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{MAX_EPOCHS}]")
        # 注意：不要在这里 zero_grad，会在循环里按 virtual batch 控制

        for step, batch in enumerate(pbar, start=1):
            # 如果是从中间恢复，且还在恢复起始 epoch，就跳过之前已经完成的 step
            if epoch == start_epoch and resume_step_in_epoch and step <= resume_step_in_epoch:
                continue

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # 有效 token 数（仅统计非 -100）
            valid_mask = labels.ne(-100)
            tokens_this_batch = valid_mask.sum().item()
            if tokens_this_batch == 0:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # 按虚拟 batch 缩放
            loss = loss * (BATCH_SIZE / VIRTUAL_BATCH_SIZE)
            loss.backward()

            epoch_loss_sum += loss.item() * tokens_this_batch
            epoch_tokens += tokens_this_batch

            # 每累计到 virtual batch 一次，更新一次参数
            if (step * BATCH_SIZE) % VIRTUAL_BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # ====== 每 SAVE_EVERY_UPDATES 次更新，保存一次 ckpt ======
                if global_step % SAVE_EVERY_UPDATES == 0:
                    ckpt_obj = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_in_epoch": step,
                    }
                    torch.save(ckpt_obj, CKPT_PATH)
                    # 这里打印一下方便确认
                    print(
                        "\n[Checkpoint] Saved at",
                        "epoch", epoch,
                        "step", step,
                        "global_step", global_step,
                    )

                if global_step % LOG_INTERVAL == 0 and epoch_tokens > 0:
                    avg_loss = epoch_loss_sum / max(epoch_tokens, 1)
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                    })

        # 一个 epoch 结束后，做简单的平均 loss
        if epoch_tokens > 0:
            epoch_avg_loss = epoch_loss_sum / epoch_tokens
        else:
            epoch_avg_loss = float("inf")

        print(f"[Epoch {epoch}] average loss = {epoch_avg_loss:.6f}")
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            save_path = SAVE_DIR / f"step_decomposer_best_ep{epoch}.pt"
            torch.save(model.state_dict(), save_path)
            print("[Info] New best model saved to", save_path)

        # 同时存一份 last（仅模型，用来快速 inference）
        last_path = SAVE_DIR / "step_decomposer_last.pt"
        torch.save(model.state_dict(), last_path)
        print("[Info] Last model saved to", last_path)

        # 一轮结束后，如果是从中间恢复的那一轮，后续 epoch 就不用再跳步了
        resume_step_in_epoch = 0


if __name__ == "__main__":
    main()
