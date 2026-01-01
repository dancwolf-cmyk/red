import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

TOTAL_DEFAULT = 20000
ANSWER_MIN_DEFAULT = 2
ANSWER_MAX_DEFAULT = 100
OUTPUT_DEFAULT = "train_ops_balancedplus.jsonl"


def gen_add(answer: int) -> Tuple[str, str]:
    a = random.randint(0, answer)
    b = answer - a
    question = f"What is {a} plus {b}?"
    expr = f"{a} plus {b}"
    return question, expr


def gen_sub(answer: int) -> Tuple[str, str]:
    minuend = answer + random.randint(0, 50)
    subtrahend = minuend - answer
    question = f"What is {minuend} minus {subtrahend}?"
    expr = f"{minuend} minus {subtrahend}"
    return question, expr


def gen_mul(answer: int) -> Tuple[str, str]:
    max_factor = min(20, answer) if answer > 0 else 1
    for _ in range(50):
        factor = random.randint(1, max_factor)
        if factor != 0 and answer % factor == 0:
            other = answer // factor
            question = f"What is {other} times {factor}?"
            expr = f"{other} times {factor}"
            return question, expr
    question = f"What is {answer} times 1?"
    expr = f"{answer} times 1"
    return question, expr


def gen_div(answer: int) -> Tuple[str, str]:
    divisor = random.randint(1, 12)
    dividend = answer * divisor
    question = f"What is {dividend} divided by {divisor}?"
    expr = f"{dividend} divided by {divisor}"
    return question, expr


OP_FUNCTIONS: Dict[str, Callable[[int], Tuple[str, str]]] = {
    "add": gen_add,
    "sub": gen_sub,
    "mul": gen_mul,
    "div": gen_div,
}


def parse_operations(raw: str) -> List[str]:
    ops = [op.strip().lower() for op in raw.split(",") if op.strip()]
    for op in ops:
        if op not in OP_FUNCTIONS:
            raise ValueError(f"Unsupported operation: {op}")
    if not ops:
        raise ValueError("At least one operation must be specified")
    return ops


def build_sample(answer: int, operations: List[str]) -> dict:
    op = random.choice(operations)
    question, expr = OP_FUNCTIONS[op](answer)
    return {
        "question": question,
        "expr": expr,
        "answer": answer,
        "operation": op,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced QA samples over multiple operators.")
    parser.add_argument("--total", type=int, default=TOTAL_DEFAULT)
    parser.add_argument("--min-answer", type=int, default=ANSWER_MIN_DEFAULT)
    parser.add_argument("--max-answer", type=int, default=ANSWER_MAX_DEFAULT)
    parser.add_argument("--operations", type=str, default="add,sub,mul,div")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_answer < args.min_answer:
        raise ValueError("--max-answer must be >= --min-answer")

    operations = parse_operations(args.operations)
    labels = list(range(args.min_answer, args.max_answer + 1))
    label_count = len(labels)
    base_count = args.total // label_count
    remainder = args.total % label_count

    random.seed(args.seed)
    samples: List[dict] = []
    for i, ans in enumerate(labels):
        count = base_count + (1 if i < remainder else 0)
        for _ in range(count):
            samples.append(build_sample(ans, operations))

    random.shuffle(samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
