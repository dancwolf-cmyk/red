import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_op_expert_v13 import (
    LABEL_VOCAB,
    ModularAddModelWithRouter,
    QwenEncoderFrozen,
    load_checkpoint_file,
)

CONFIG_PATH = Path("config.json")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

logger = logging.getLogger("step_exec")


def setup_logger(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    for hd in list(logger.handlers):
        logger.removeHandler(hd)
    logger.addHandler(handler)


def read_config() -> Dict[str, str]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} is missing.")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate_after_answer(text: str) -> str:
    match = re.search(r"Answer:\s*x\d+", text)
    if not match:
        return text
    end = match.end()
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:end]


def build_prompt(expr: str) -> str:
    return (
        "Decompose the following arithmetic expression into step-by-step computation "
        "using variables x1, x2, ...\n"
        f"Expression: {expr}\n\n"
        "Program:\n"
    )


def generate_batch_programs(
    samples_batch: List[Dict],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    max_len: int,
    max_new_tokens: int,
) -> List[str]:
    prompts = [build_prompt(sample["expr"]) for sample in samples_batch]
    enc = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    prompt_lens = attention_mask.sum(dim=1)
    programs = []
    for i, length in enumerate(prompt_lens):
        start = int(length.item())
        gen_ids = outputs[i, start:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        programs.append(truncate_after_answer(decoded.strip()))
    return programs


def chunked(sequence: List[Dict], size: int):
    for i in range(0, len(sequence), size):
        yield sequence[i : i + size]


STEP_RE = re.compile(r"Step\s+\d+:\s+(.+?)\s+([+-])\s+(.+?)\s+=\s+(x\d+)")


def resolve_token(token: str, vars_map: Dict[str, int]) -> Optional[int]:
    token = token.strip()
    if not token:
        return None
    if token.startswith("x"):
        return vars_map.get(token)
    try:
        return int(token)
    except ValueError:
        return None


def router_step_question(lhs: int, rhs: int, op: str) -> str:
    if op == "+":
        return f"What is {lhs} plus {rhs}?"
    return f"What is {lhs} minus {rhs}?"


def router_predict(
    lhs: int,
    rhs: int,
    op: str,
    tokenizer: AutoTokenizer,
    model: ModularAddModelWithRouter,
    device: torch.device,
    max_len: int,
) -> Tuple[Optional[int], List[str]]:
    logs = []
    question = router_step_question(lhs, rhs, op)
    enc = tokenizer(
        question,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, mode="router")
    pred_idx = torch.argmax(logits, dim=-1).item()
    value = int(LABEL_VOCAB[pred_idx])
    logs.append(f"Router Q: {question} => {value} (idx={pred_idx})")
    return value, logs


def execute_program(
    program: str,
    router_fn: Callable[[int, int, str], Tuple[Optional[int], List[str]]],
) -> Tuple[Optional[int], List[str]]:
    vars_map: Dict[str, int] = {}
    last_var: Optional[str] = None
    logs: List[str] = []
    for line in program.splitlines():
        line = line.strip()
        if not line.startswith("Step"):
            continue
        match = STEP_RE.match(line)
        if not match:
            logs.append(f"Unmatched step line: {line}")
            continue
        lhs_token, op, rhs_token, out_var = match.groups()
        lhs_val = resolve_token(lhs_token, vars_map)
        rhs_val = resolve_token(rhs_token, vars_map)
        if lhs_val is None or rhs_val is None:
            logs.append(f"Cannot resolve operands: {lhs_token}, {rhs_token}")
            return None, logs
        result, step_logs = router_fn(lhs_val, rhs_val, op)
        if result is None:
            logs.extend(step_logs)
            return None, logs
        vars_map[out_var] = result
        logs.append(f"{line} => {result}")
        logs.extend(step_logs)
        last_var = out_var
    if last_var is None:
        logs.append("No steps parsed")
        return None, logs
    value = vars_map.get(last_var)
    logs.append(f"{last_var} = {value}")
    return value, logs


def load_jsonl(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def split_dataset(
    samples: List[Dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    random.Random(seed).shuffle(samples)
    total = len(samples)
    val_n = int(total * val_ratio)
    test_n = int(total * test_ratio)
    if val_n + test_n >= total:
        raise ValueError("val_ratio + test_ratio must be < 1")
    val_samples = samples[:val_n]
    test_samples = samples[val_n : val_n + test_n]
    return val_samples, test_samples


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


def build_router_model(
    cfg: Dict[str, str],
    device: torch.device,
    add_ckpt: Path,
    sub_ckpt: Path,
    base_ckpt: Path,
    router_ckpt: Path,
) -> ModularAddModelWithRouter:
    encoder = QwenEncoderFrozen(cfg["model_path"], trainable_last_n_layers=3).to(device)
    model = ModularAddModelWithRouter(
        encoder,
        num_classes=len(LABEL_VOCAB),
        num_experts=2,
    ).to(device)

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

    router_obj = torch.load(router_ckpt, map_location="cpu")
    router_state = router_obj.get("router_state", router_obj)
    model.router.load_state_dict(router_state)

    model.eval()
    return model


def find_best_checkpoint(folder: Path) -> Path:
    candidates = sorted(folder.glob("step_decomposer_best_ep*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint under {folder}")
    return candidates[-1]


def normalize_prediction(value: Optional[int]) -> int:
    return value if value is not None else -1


def compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def evaluate_split(name: str, samples: List[Dict], predictions: List[Optional[int]]) -> Dict:
    y_true = [sample["answer"] for sample in samples]
    y_pred = [normalize_prediction(p) for p in predictions]
    total = len(samples)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    accuracy = correct / total if total else 0.0
    f1 = compute_macro_f1(y_true, y_pred)
    print(f"\n=== {name.upper()} set ({total} samples) ===")
    print(f"Overall  | Accuracy={accuracy:.4f}  Macro-F1={f1:.4f}")
    by_steps: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for sample, pred in zip(samples, y_pred):
        by_steps[sample["num_steps"]].append((sample["answer"], pred))
    step_stats = {}
    for steps in sorted(by_steps):
        true_vals, pred_vals = zip(*by_steps[steps])
        acc = sum(1 for gt, pr in zip(true_vals, pred_vals) if gt == pr) / len(true_vals)
        f1_steps = compute_macro_f1(list(true_vals), list(pred_vals))
        print(f"{steps}-step | Accuracy={acc:.4f}  Macro-F1={f1_steps:.4f}")
        step_stats[steps] = {"accuracy": acc, "f1": f1_steps}
    return {"accuracy": accuracy, "f1": f1, "step_stats": step_stats}


def run_split(
    split_name: str,
    samples_list: List[Dict],
    tokenizer: AutoTokenizer,
    causal_model: AutoModelForCausalLM,
    router_tokenizer: AutoTokenizer,
    router_model: ModularAddModelWithRouter,
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_new: int,
    router_max_len: int,
) -> Dict:
    predictions: List[Optional[int]] = []
    iterator = tqdm(total=len(samples_list), desc=f"{split_name} inference")
    for batch_samples in chunked(samples_list, batch_size):
        programs = generate_batch_programs(batch_samples, tokenizer, causal_model, device, max_length, max_new)
        for sample, program in zip(batch_samples, programs):
            router_fn = lambda lhs, rhs, op: router_predict(lhs, rhs, op, router_tokenizer, router_model, device, router_max_len)
            value, exec_logs = execute_program(program, router_fn)
            predictions.append(value)
            if value is None or value != sample["answer"]:
                logger.info(f"[{split_name.upper()}] id={sample.get('id')} expr={sample['expr']} steps={sample['num_steps']} reference={sample['answer']} prediction={value}")
                logger.info(f"Generated program:\n{program}")
                logger.info("Execution trace:")
                for log_line in exec_logs:
                    logger.info(log_line)
        iterator.update(len(batch_samples))
    iterator.close()
    stats = evaluate_split(split_name, samples_list, predictions)
    logger.info(f"{split_name} overall accuracy={stats['accuracy']:.4f} f1={stats['f1']:.4f}")
    for steps, stats_map in stats["step_stats"].items():
        logger.info(f"{split_name} {steps}-step accuracy={stats_map['accuracy']:.4f} f1={stats_map['f1']:.4f}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Execute step decomposer programs and score predictions.")
    parser.add_argument("--data", type=Path, default=Path("mixed_arith_step_decomposer_80k2.jsonl"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints_step_decomposer"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-file", type=Path, default=Path("execute_step.log"))
    parser.add_argument("--add-ckpt", type=Path, default=Path("checkpoints_v13_op_expert/demo_v13_op_add_expert_best.pt"))
    parser.add_argument("--sub-ckpt", type=Path, default=Path("checkpoints_v13_op_expert/demo_v13_op_sub_expert_best.pt"))
    parser.add_argument("--router-ckpt", type=Path, default=Path("checkpoints_v13_router/router_best.pt"))
    parser.add_argument("--router-max-len", type=int, default=32)
    args = parser.parse_args()

    cfg = read_config()
    setup_logger(args.log_file)
    samples = load_jsonl(args.data)
    val_samples, test_samples = split_dataset(samples, args.val_ratio, args.test_ratio, args.seed)
    if args.limit:
        val_samples = val_samples[: args.limit]
        test_samples = test_samples[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    router_tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    router_tokenizer.padding_side = "left"
    if router_tokenizer.pad_token is None:
        router_tokenizer.pad_token = router_tokenizer.eos_token

    checkpoint_path = find_best_checkpoint(args.ckpt_dir)
    print(f"[Info] Loading step decomposer checkpoint from {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=AMP_DTYPE,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    router_model = build_router_model(
        cfg,
        device,
        args.add_ckpt,
        args.sub_ckpt,
        Path(cfg["base_ckpt"]),
        args.router_ckpt,
    )

    run_split(
        "val",
        val_samples,
        tokenizer,
        model,
        router_tokenizer,
        router_model,
        device,
        args.batch_size,
        args.max_length,
        args.max_new,
        args.router_max_len,
    )
    run_split(
        "test",
        test_samples,
        tokenizer,
        model,
        router_tokenizer,
        router_model,
        device,
        args.batch_size,
        args.max_length,
        args.max_new,
        args.router_max_len,
    )


if __name__ == "__main__":
    main()
