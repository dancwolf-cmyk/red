# -*- coding: utf-8 -*-
"""
execute_step_decomposer_moe.py
------------------------------
在 step decomposer + add/sub 专家 + MoE gate 的框架下做测试：

- Step decomposer: 把表达式拆成 Step 1, Step 2, ...
- 每一步调用 MoE 模型：
    * 输入："What is A plus B?" / "What is A minus B?"
    * 模型：ModularAddModelWithMoE
    * mode="moe"，使用训练好的 gate 混合 add/sub 专家
- 按样本整体计算：
    * Overall accuracy / Macro-F1
    * 按 num_steps 统计分段指标
"""

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

from demo_v13 import ModularAddModelWithMoE
from train_op_expert_v13 import QwenEncoderFrozen, LABEL_VOCAB

CONFIG_PATH = Path("config.json")
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

eval_logger = logging.getLogger("step_exec_moe")
error_logger = logging.getLogger("step_exec_moe.error")


# ========================
# 基础工具
# ========================

def setup_logger(eval_log_path: Path, error_log_path: Path) -> None:
    eval_logger.setLevel(logging.INFO)
    error_logger.setLevel(logging.INFO)

    eval_handler = logging.FileHandler(eval_log_path, encoding="utf-8")
    eval_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    for hd in list(eval_logger.handlers):
        eval_logger.removeHandler(hd)
    eval_logger.addHandler(eval_handler)
    eval_logger.propagate = False

    err_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    err_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    for hd in list(error_logger.handlers):
        error_logger.removeHandler(hd)
    error_logger.addHandler(err_handler)
    error_logger.propagate = False


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
        "Expression: " + expr + "\n\n"
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
        yield sequence[i: i + size]


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


# ========================
# MoE 推理部分
# ========================

def moe_step_question(lhs: int, rhs: int, op: str) -> str:
    if op == "+":
        return "What is " + str(lhs) + " plus " + str(rhs) + "?"
    return "What is " + str(lhs) + " minus " + str(rhs) + "?"


def moe_predict(
    lhs: int,
    rhs: int,
    op: str,
    tokenizer: AutoTokenizer,
    model: ModularAddModelWithMoE,
    device: torch.device,
    max_len: int,
) -> Tuple[Optional[int], List[str]]:
    """
    使用传统 MoE：
      - 问题文本："What is A plus/minus B?"
      - ModularAddModelWithMoE(mode="moe") 输出 logits
      - gate 根据隐层自动决定 add/sub 的混合
    """
    logs: List[str] = []
    question = moe_step_question(lhs, rhs, op)

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
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="moe",
        )

    pred_idx = torch.argmax(logits, dim=-1).item()
    value = int(LABEL_VOCAB[pred_idx])
    logs.append("MoE Q: " + question + " => " + str(value) + " (idx=" + str(pred_idx) + ")")
    return value, logs


def execute_program(
    program: str,
    step_fn: Callable[[int, int, str], Tuple[Optional[int], List[str]]],
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
            logs.append("Unmatched step line: " + line)
            continue

        lhs_token, op, rhs_token, out_var = match.groups()
        lhs_val = resolve_token(lhs_token, vars_map)
        rhs_val = resolve_token(rhs_token, vars_map)
        if lhs_val is None or rhs_val is None:
            logs.append("Cannot resolve operands: " + lhs_token + ", " + rhs_token)
            return None, logs

        result, step_logs = step_fn(lhs_val, rhs_val, op)
        if result is None:
            logs.extend(step_logs)
            return None, logs

        vars_map[out_var] = result
        logs.append(line + " => " + str(result))
        logs.extend(step_logs)
        last_var = out_var

    if last_var is None:
        logs.append("No steps parsed")
        return None, logs

    value = vars_map.get(last_var)
    logs.append(last_var + " = " + str(value))
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
    rnd = random.Random(seed)
    rnd.shuffle(samples)
    total = len(samples)
    val_n = int(total * val_ratio)
    test_n = int(total * test_ratio)
    if val_n + test_n >= total:
        raise ValueError("val_ratio + test_ratio must be < 1")
    val_samples = samples[:val_n]
    test_samples = samples[val_n: val_n + test_n]
    return val_samples, test_samples


def find_best_step_ckpt(folder: Path) -> Path:
    candidates = sorted(folder.glob("step_decomposer_best_ep*.pt"))
    if not candidates:
        raise FileNotFoundError("No best checkpoint under " + str(folder))
    return candidates[-1]


def normalize_prediction(value: Optional[int]) -> int:
    return value if value is not None else -1


def compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores: List[float] = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / float(len(scores))


def evaluate_split(name: str, samples: List[Dict], predictions: List[Optional[int]]) -> Dict:
    y_true = [sample["answer"] for sample in samples]
    y_pred = [normalize_prediction(p) for p in predictions]
    total = len(samples)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / float(total) if total else 0.0
    f1 = compute_macro_f1(y_true, y_pred)

    print("\n=== " + name.upper() + " set (" + str(total) + " samples) ===")
    print("Overall  | Accuracy={:.4f}  Macro-F1={:.4f}".format(accuracy, f1))

    by_steps: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for sample, pred in zip(samples, y_pred):
        by_steps[sample["num_steps"]].append((sample["answer"], pred))

    step_stats: Dict[int, Dict[str, float]] = {}
    for steps in sorted(by_steps.keys()):
        true_vals, pred_vals = zip(*by_steps[steps])
        acc = sum(1 for gt, pr in zip(true_vals, pred_vals) if gt == pr) / float(len(true_vals))
        f1_steps = compute_macro_f1(list(true_vals), list(pred_vals))
        print("{}-step | Accuracy={:.4f}  Macro-F1={:.4f}".format(steps, acc, f1_steps))
        step_stats[steps] = {"accuracy": acc, "f1": f1_steps}

    return {"accuracy": accuracy, "f1": f1, "step_stats": step_stats}


def run_split(
    split_name: str,
    samples_list: List[Dict],
    tokenizer: AutoTokenizer,
    causal_model: AutoModelForCausalLM,
    moe_tokenizer: AutoTokenizer,
    moe_model: ModularAddModelWithMoE,
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_new: int,
    moe_max_len: int,
) -> Dict:
    predictions: List[Optional[int]] = []
    iterator = tqdm(total=len(samples_list), desc=split_name + " inference (MoE)")

    for batch_samples in chunked(samples_list, batch_size):
        programs = generate_batch_programs(
            batch_samples,
            tokenizer,
            causal_model,
            device,
            max_length,
            max_new,
        )

        for sample, program in zip(batch_samples, programs):
            step_fn = lambda lhs, rhs, op: moe_predict(
                lhs,
                rhs,
                op,
                moe_tokenizer,
                moe_model,
                device,
                moe_max_len,
            )
            value, exec_logs = execute_program(program, step_fn)
            predictions.append(value)

            # ✅ 每条样本单独判断并写错误日志（与 batch_size 无关）
            if value is None or value != sample["answer"]:
                error_logger.info(
                    "[{}] id={} expr={} steps={} reference={} prediction={}".format(
                        split_name.upper(),
                        sample.get("id"),
                        sample["expr"],
                        sample["num_steps"],
                        sample["answer"],
                        value,
                    )
                )
                error_logger.info("Generated program:\n" + program)
                error_logger.info("Execution trace:")
                for log_line in exec_logs:
                    error_logger.info(log_line)

        iterator.update(len(batch_samples))

    iterator.close()
    stats = evaluate_split(split_name, samples_list, predictions)

    eval_logger.info(
        "{} overall accuracy={:.4f} f1={:.4f}".format(
            split_name,
            stats["accuracy"],
            stats["f1"],
        )
    )
    for steps, stats_map in stats["step_stats"].items():
        eval_logger.info(
            "{} {}-step accuracy={:.4f} f1={:.4f}".format(
                split_name,
                steps,
                stats_map["accuracy"],
                stats_map["f1"],
            )
        )
    return stats


# ========================
# 构建 MoE 模型（加载 ckpt）
# ========================

def build_moe_model(
    cfg: Dict[str, str],
    moe_ckpt: Path,
    device: torch.device,
) -> ModularAddModelWithMoE:
    """
    构建 ModularAddModelWithMoE 并加载训练好的传统 MoE ckpt：
      - ckpt 建议来自 train_moe_router_ops.py 保存的 moe_router_best.pt
      - 里面包含 encoder + experts + router 的完整 model_state
    """
    encoder = QwenEncoderFrozen(cfg["model_path"], trainable_last_n_layers=3).to(device)
    model = ModularAddModelWithMoE(
        encoder,
        num_classes=len(LABEL_VOCAB),
        num_experts=2,
    ).to(device)

    ckpt = torch.load(moe_ckpt, map_location="cpu")
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    # 统一 dtype，避免 hidden(bf16) 和 router(float) 冲突
    encoder_dtype = next(encoder.parameters()).dtype
    model = model.to(device=device, dtype=encoder_dtype)
    model.eval()
    return model


# ========================
# main
# ========================

def main():
    parser = argparse.ArgumentParser(description="Execute step decomposer programs (traditional MoE experts) and score predictions.")
    parser.add_argument("--data", type=Path, default=Path("mixed_arith_step_decomposer_80k2.jsonl"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints_step_decomposer"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-file", type=Path, default=Path("execute_step_moe.log"))
    parser.add_argument("--error-log", type=Path, default=Path("execute_step_moe_error.log"))
    parser.add_argument("--moe-ckpt", type=Path, default=Path("checkpoints_v13_moe/moe_router_best.pt"))
    parser.add_argument("--moe-max-len", type=int, default=32)
    args = parser.parse_args()

    cfg = read_config()
    setup_logger(args.log_file, args.error_log)

    samples = load_jsonl(args.data)
    val_samples, test_samples = split_dataset(samples, args.val_ratio, args.test_ratio, args.seed)
    if args.limit:
        val_samples = val_samples[: args.limit]
        test_samples = test_samples[: args.limit]

    # step decomposer tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MoE tokenizer（同一个 Qwen）
    moe_tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    moe_tokenizer.padding_side = "left"
    if moe_tokenizer.pad_token is None:
        moe_tokenizer.pad_token = moe_tokenizer.eos_token

    # 加载 step decomposer checkpoint
    checkpoint_path = find_best_step_ckpt(args.ckpt_dir)
    print("[Info] Loading step decomposer checkpoint from " + str(checkpoint_path))
    causal_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=AMP_DTYPE,
    )
    causal_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    causal_model.to(device)
    causal_model.eval()

    # 构建 MoE 模型（传统 gate）
    moe_model = build_moe_model(cfg, args.moe_ckpt, device)

    # # 评价
    # run_split(
    #     "val",
    #     val_samples,
    #     tokenizer,
    #     causal_model,
    #     moe_tokenizer,
    #     moe_model,
    #     device,
    #     args.batch_size,
    #     args.max_length,
    #     args.max_new,
    #     args.moe_max_len,
    # )
    run_split(
        "test",
        test_samples,
        tokenizer,
        causal_model,
        moe_tokenizer,
        moe_model,
        device,
        args.batch_size,
        args.max_length,
        args.max_new,
        args.moe_max_len,
    )


if __name__ == "__main__":
    main()
