#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert the v12 base-stage checkpoint so it can be re-used in demo_v13.
Usage:
  python convert_v12_base_to_v13.py \
      --input checkpoints_v12/demo_v12_base_best_ep1.pt \
      --output checkpoints_v13/demo_v13_base_converted.pt
      python convert_v12_base_to_v13.py \
    --stage expert_2 \
    --input checkpoints_v12/demo_v12_expert_2_best_epX.pt \
    --output checkpoints_v13/demo_v13_expert_2_converted.pt

        python convert_v12_base_to_v13.py    --stage expert_4   --input checkpoints_v12/demo_v12_expert_4_best_ep9.pt     --output checkpoints_v13/demo_v12_expert_4_best_ep28.pt
"""

import argparse
import torch

from demo_v13 import ModularAddModelWithRouter, QwenEncoderFrozen, _collect_stage_state_dict, MODEL_PATH


STAGE_CHOICES = ["base"] + [f"expert_{i}" for i in range(5)]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a v12 stage checkpoint for demo_v13.")
    parser.add_argument("--stage", "-s", choices=STAGE_CHOICES, default="base",
                        help="Which stage checkpoint is being converted.")
    parser.add_argument("--input", "-i", required=True, help="Existing v12 stage checkpoint.")
    parser.add_argument("--output", "-o", required=True, help="Path where the converted ckpt will be written.")
    parser.add_argument("--trainable-layers", "-n", type=int, default=1,
                        help="Number of encoder layers that are trainable in demo_v13.")
    return parser.parse_args()


def set_stage_requires(model, stage):
    for param in model.parameters():
        param.requires_grad = False
    if stage == "base":
        for param in model.base_head.parameters():
            param.requires_grad = True
        # encoder already unfreezes last layers via constructor
    elif stage.startswith("expert_"):
        idx = int(stage.split("_")[-1])
        for param in model.experts[idx].parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def main():
    args = parse_args()
    encoder = QwenEncoderFrozen(MODEL_PATH, trainable_last_n_layers=args.trainable_layers)
    model = ModularAddModelWithRouter(encoder, num_classes=99)
    set_stage_requires(model, args.stage)
    ckpt = torch.load(args.input, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    filtered = _collect_stage_state_dict(model)
    torch.save({"state_dict": filtered, "stage": f"{args.stage}_converted"}, args.output)
    print(f"Converted v12 {args.stage} checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
