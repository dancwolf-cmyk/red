"""
Shrink demo_v11 checkpoints by keeping only the trainable pieces.

What is kept by default:
- The last N transformer layers of the Qwen encoder (matches trainable_last_n_layers).
- The encoder final norm (also trainable in demo_v11).
- The heads: base_head, experts, router, router_scale.

Usage examples
-------------
# 1) Shrink a big checkpoint to only the useful parameters (keep last 1 layer)
py shrink_checkpoint.py --in checkpoints/demo_v11_base_best_ep10.pt

# 2) Shrink and cast kept tensors to fp16 to save more space
py shrink_checkpoint.py --in checkpoints/demo_v11_base_best_ep10.pt --dtype float16

# 3) After shrinking, rebuild a model and load the slim weights
py shrink_checkpoint.py --in checkpoints/demo_v11_base_best_ep10.pt --check-load
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import torch

# Reuse the model definitions from demo_v11 so we can reload the slim checkpoint.
from demo_v11 import (
    MODEL_PATH as DEFAULT_MODEL_PATH,
    ModularAddModelWithRouter,
    QwenEncoderFrozen,
)


# Support both encoder.model.layers.* and encoder.model.model.layers.* naming variants.
LAYER_RE_LIST = [
    re.compile(r"^encoder\.model\.model\.layers\.(\d+)\."),
    re.compile(r"^encoder\.model\.layers\.(\d+)\."),
]
NORM_PREFIXES: Tuple[str, ...] = (
    "encoder.model.model.norm",
    "encoder.model.norm",
)
HEAD_PREFIXES: Tuple[str, ...] = ("base_head", "experts", "router", "router_scale")


def _collect_layer_ids(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    ids: Set[int] = set()
    for k in state_dict:
        for pat in LAYER_RE_LIST:
            m = pat.match(k)
            if m:
                ids.add(int(m.group(1)))
                break
    return sorted(ids)


def _should_keep_key(key: str, keep_layers: Set[int]) -> bool:
    for pat in LAYER_RE_LIST:
        m = pat.match(key)
        if m and int(m.group(1)) in keep_layers:
            return True
    if key.startswith(HEAD_PREFIXES):
        return True
    if any(key.startswith(p) for p in NORM_PREFIXES):
        return True
    return False


def _filter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    last_n_layers: int,
    target_dtype: torch.dtype | None = None,
) -> Tuple[Dict[str, torch.Tensor], Set[int], List[int]]:
    all_layers = _collect_layer_ids(state_dict)
    keep_layers = set(all_layers[-last_n_layers:] if last_n_layers > 0 else [])
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not _should_keep_key(k, keep_layers):
            continue
        filtered[k] = v.to(target_dtype) if target_dtype is not None else v
    return filtered, keep_layers, all_layers


def _count_params(tensors: Iterable[torch.Tensor]) -> Tuple[int, int]:
    total_elems = 0
    total_bytes = 0
    for t in tensors:
        total_elems += t.numel()
        total_bytes += t.numel() * t.element_size()
    return total_elems, total_bytes


def save_slim_checkpoint(
    in_path: Path,
    out_path: Path,
    last_n_layers: int = 1,
    dtype: str | None = None,
) -> Dict:
    raw = torch.load(in_path, map_location="cpu")
    state_dict: Dict[str, torch.Tensor] = raw["state_dict"] if "state_dict" in raw else raw
    target_dtype = {"float16": torch.float16, "float32": torch.float32}.get(dtype)

    filtered, keep_layers, all_layers = _filter_state_dict(
        state_dict, last_n_layers=last_n_layers, target_dtype=target_dtype
    )
    if last_n_layers > 0 and not keep_layers:
        raise RuntimeError(
            f"No encoder layers matched known patterns; cannot keep last {last_n_layers} layers. "
            f"Check state_dict keys and LAYER_RE_LIST."
        )
    elems_before, bytes_before = _count_params(state_dict.values())
    elems_after, bytes_after = _count_params(filtered.values())

    payload = {
        "state_dict": filtered,
        "meta": {
            "source": str(in_path),
            "stage": raw.get("stage"),
            "epoch": raw.get("epoch"),
            "all_layers": all_layers,
            "kept_layers": sorted(keep_layers),
            "trainable_last_n_layers": last_n_layers,
            "dtype": str(target_dtype) if target_dtype is not None else "original",
            "num_params_before": elems_before,
            "num_params_after": elems_after,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload


def load_model_with_slim_checkpoint(
    slim_path: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    num_classes: int = 99,
    num_experts: int = 5,
    trainable_last_n_layers: int = 1,
):
    """
    Rebuild the model and load the slim checkpoint (frozen parts come from base model).
    """
    ckpt = torch.load(slim_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    encoder = QwenEncoderFrozen(
        model_path=model_path,
        trainable_last_n_layers=trainable_last_n_layers,
    )
    model = ModularAddModelWithRouter(
        encoder=encoder,
        num_classes=num_classes,
        num_experts=num_experts,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded slim checkpoint from {slim_path}")
    print(f"Missing keys (expected for frozen layers): {missing}")
    print(f"Unexpected keys: {unexpected}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Shrink demo_v11 checkpoints.")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to original .pt")
    parser.add_argument(
        "--out", dest="output_path", default=None, help="Path to slim .pt (default: add .slim.pt)"
    )
    parser.add_argument(
        "--trainable-last-n-layers",
        type=int,
        default=1,
        help="How many encoder layers to keep (matches your finetuned layers).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default=None,
        help="Optional: cast kept tensors to this dtype to shrink further.",
    )
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="Rebuild the model and load the slim checkpoint to verify it works.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path for reload check.")
    parser.add_argument("--num-classes", type=int, default=99, help="num_classes for the heads.")
    parser.add_argument("--num-experts", type=int, default=5, help="Number of experts in the router model.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = (
        Path(args.output_path) if args.output_path else input_path.with_suffix(input_path.suffix + ".slim.pt")
    )

    payload = save_slim_checkpoint(
        input_path,
        output_path,
        last_n_layers=args.trainable_last_n_layers,
        dtype=args.dtype,
    )
    meta = payload["meta"]
    print(f"Slim checkpoint saved to: {output_path}")
    print(
        f"Params: {meta['num_params_after']} (was {meta['num_params_before']}), "
        f"bytes: {meta['bytes_after']} (was {meta['bytes_before']})"
    )
    print(f"Kept encoder layers: {meta['kept_layers']}, dtype: {meta['dtype']}")

    if args.check_load:
        load_model_with_slim_checkpoint(
            output_path,
            model_path=args.model_path,
            num_classes=args.num_classes,
            num_experts=args.num_experts,
            trainable_last_n_layers=args.trainable_last_n_layers,
        )


if __name__ == "__main__":
    main()
