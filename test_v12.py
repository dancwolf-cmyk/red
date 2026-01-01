# -*- coding: utf-8 -*-
"""
compare_v12_base_vs_exp0.py
---------------------------
在 Task2 (demo_v12) 中，对比：
  1) base-only (mode="base")
  2) base + expert_0 (mode="all_fixed", expert_mask=[1,0,0,0,0])
在完整 16000 条 range_2_20 样本上的 ACC。

使用方法：
  1. 确保当前目录下有 demo_v12.py
  2. v12 已经训练并生成：
       - checkpoints_v12/demo_v12_base_best_*.pt
       - checkpoints_v12/demo_v12_expert_0_best_*.pt
       - checkpoints_v12/demo_v12_samples_mixed.json
  3. 运行：
       python compare_v12_base_vs_exp0.py
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 从 demo_v12 导入现成组件
from demo_v12 import (
    QwenEncoderFrozen,
    ModularAddModelWithRouter,
    QADataset,
    load_or_create_samples,
    DOMAIN_ORDER,
    MODEL_PATH,
    DEVICE,
    BATCH_SIZE,
    DEFAULT_SEED,
    SAVE_DIR,
    load_best_checkpoint,
)


def make_loader(dataset, shuffle: bool = False, seed_offset: int = 0):
    gen = torch.Generator()
    gen.manual_seed(DEFAULT_SEED + seed_offset)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen)


@torch.no_grad()
def compute_acc(model, loader, mode: str, expert_mask: torch.Tensor = None):
    model.eval()
    total = 0
    correct = 0

    if expert_mask is not None:
        expert_mask = expert_mask.to(DEVICE)

    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(
            ids,
            mask,
            mode=mode,
            expert_mask=expert_mask
        )
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / float(total if total > 0 else 1)


def main():
    print("=> Loading cached samples (v12)...")
    samples, label_vocab = load_or_create_samples()  # 会从 demo_v12_samples_mixed.json 读

    # 1) 取 domain = "range_2_20" 的全部样本（应该是 16000 条）
    domain_name = DOMAIN_ORDER[0]  # "range_2_20"
    range0_samples = [s for s in samples if s["domain"] == domain_name]
    print("Domain =", domain_name, "samples =", len(range0_samples))

    # 2) 构建 Dataset & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ds_range0 = QADataset(range0_samples, tokenizer, label_vocab)
    loader_range0 = make_loader(ds_range0, shuffle=False, seed_offset=100)

    # 3) 构建 RED 模型：encoder + ModularAddModelWithRouter
    print("=> Building model...")
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=1).to(DEVICE)
    model = ModularAddModelWithRouter(
        encoder,
        num_classes=len(label_vocab),
        num_experts=5,
    ).to(DEVICE)

    # 4) 加载 base & expert_0 的 best checkpoint
    print("=> Loading best base checkpoint...")
    load_best_checkpoint("base", model, strict=False)

    print("=> Loading best expert_0 checkpoint...")
    load_best_checkpoint("expert_0", model, strict=False)

    # 5) 计算 base-only ACC
    print("\n=> Evaluating base-only on full range_2_20 (16000 samples)...")
    acc_base = compute_acc(model, loader_range0, mode="base")
    print("[RESULT] ACC (base-only) on domain range_2_20 =", acc_base)

    # 6) 计算 base + expert_0 ACC
    print("\n=> Evaluating base + expert_0 on full range_2_20 (16000 samples)...")
    expert_mask = torch.zeros(model.num_experts)
    expert_mask[0] = 1.0  # 只启用 expert_0

    acc_mix = compute_acc(model, loader_range0, mode="all_fixed", expert_mask=expert_mask)
    print("[RESULT] ACC (base + expert_0) on domain range_2_20 =", acc_mix)

    print("\nDone.")


if __name__ == "__main__":
    main()
