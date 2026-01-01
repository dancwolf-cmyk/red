# -*- coding: utf-8 -*-
"""
生成“拆步器”训练数据（不含中间数值的步骤）

Task 2: 混合加减，2~4 步表达式，结果限制在 [2,100] 并映射到 5 个区间：
    [2, 20], [21, 40], [41, 60], [61, 80], [81, 100]

目标：
  - 总样本数：80,000
  - 每个区间：16,000
  - 每个区间内 2/3/4 步比例：
        2-step: 50%
        3-step: 30%
        4-step: 20%

每条样本包含字段：
  - id        : 全局编号
  - expr      : 原始表达式文本（如 "12 minus 5 plus 9"）
  - answer    : 最终结果（int，仅供后续使用，拆步器可以不用）
  - label     : 0..4
  - domain    : 区间名（如 "range_2_20"）
  - num_steps : 2 / 3 / 4
  - program   : 步骤程序（仅结构，不含中间数值），例：
        Step 1: 12 + 5 = x1
        Step 2: x1 + 3 = x2
        Answer: x2
"""

import json
import random
from typing import List, Tuple, Dict

# 五个数值区间
RANGES: List[Tuple[int, int]] = [
    (2, 20),
    (21, 40),
    (41, 60),
    (61, 80),
    (81, 100),
]

RANGE_NAMES: List[str] = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]

TOTAL_SAMPLES = 80000
SAMPLES_PER_CLASS = TOTAL_SAMPLES // len(RANGES)  # 16000

# 2/3/4 步比例
RATIO_2 = 0.2
RATIO_3 = 0.4
RATIO_4 = 0.4

# 操作数采样范围（可以按需调整）
OPERAND_MIN = 1
OPERAND_MAX = 50  # 不要太大，避免太多表达式结果落在区间外


def eval_expression(operands: List[int], operators: List[str]) -> int:
    """
    从左到右计算加减表达式的结果。
    operands: [a, b, c, ...]
    operators: ['+', '-', ...]，长度 = len(operands) - 1
    """
    assert len(operands) == len(operators) + 1
    value = operands[0]
    for op, x in zip(operators, operands[1:]):
        if op == "+":
            value = value + x
        elif op == "-":
            value = value - x
        else:
            raise ValueError("Unknown operator: " + str(op))
    return value


def build_structural_program(operands: List[int], operators: List[str]) -> str:
    """
    构造“仅结构”的拆步程序（不包含任何中间数值）。

    例如：
      operands = [12, 5, 3]
      operators = ['+', '-']

    返回：
      Step 1: 12 + 5 = x1
      Step 2: x1 - 3 = x2
      Answer: x2
    """
    assert len(operands) == len(operators) + 1

    steps: List[str] = []
    num_steps = len(operators)

    # 第一步：用第一个操作数做左侧
    lhs = str(operands[0])
    for i, (op, x) in enumerate(zip(operators, operands[1:]), start=1):
        rhs = str(x)
        out_var = "x" + str(i)
        # 使用符号 + / -，更简洁
        step_str = "Step " + str(i) + ": " + lhs + " " + op + " " + rhs + " = " + out_var
        steps.append(step_str)
        # 下一步的左侧变成当前的 out_var
        lhs = out_var

    # 最终答案只是最后一个变量
    final_var = "x" + str(num_steps)
    steps.append("Answer: " + final_var)

    return "\n".join(steps)


def build_expr_text(operands: List[int], operators: List[str]) -> str:
    """
    构造自然语言形式的表达式：
        12 minus 5 plus 9
    """
    assert len(operands) == len(operators) + 1

    tokens: List[str] = [str(operands[0])]
    for op, x in zip(operators, operands[1:]):
        if op == "+":
            tokens.append("plus")
        else:
            tokens.append("minus")
        tokens.append(str(x))
    return " ".join(tokens)


def sample_expression_for_range(
    low: int,
    high: int,
    num_steps: int,
    max_trials: int = 1000,
) -> Tuple[List[int], List[str], int]:
    """
    为指定区间 [low, high] 采样一个 num_steps 步的表达式（加减混合），
    要求每个中间结果都落在 [2, 100] 范围内，最终值也落在 [low, high]。

    返回：
      operands, operators, value

    若尝试 max_trials 次仍失败，则抛异常。
    """
    for _ in range(max_trials):
        operands: List[int] = []
        operators: List[str] = []
        cur = random.randint(2, 50)
        operands.append(cur)
        valid = True
        for step_idx in range(num_steps):
            # determine target range for this step result
            if step_idx == num_steps - 1:
                target_min, target_max = 2, 100
            else:
                target_min, target_max = 2, 50
            candidates = []
            # addition candidate if we can pick operand in [2,50] keeping within bounds
            max_add = min(50, target_max - cur)
            if max_add >= 2:
                candidates.append("+")
            # subtraction candidate
            max_sub = min(50, cur - target_min)
            if max_sub >= 2:
                candidates.append("-")
            if not candidates:
                valid = False
                break

            op = random.choice(candidates)
            if op == "+":
                operand = random.randint(2, max_add)
                new_cur = cur + operand
            else:
                operand = random.randint(2, max_sub)
                new_cur = cur - operand

            if not (target_min <= new_cur <= target_max):
                valid = False
                break

            operators.append(op)
            operands.append(operand)
            cur = new_cur

        if not valid:
            continue

        if low <= cur <= high:
            return operands, operators, cur

    raise RuntimeError(
        "Failed to sample expression for range [" + str(low) + ", " + str(high) +
        "] with steps " + str(num_steps)
    )


def build_step_decomposer_dataset(
    output_jsonl: str,
    seed: int = 42,
) -> None:
    """
    生成完整数据集并写入 JSONL 文件。
    """
    random.seed(seed)

    samples: List[Dict] = []
    global_id = 0

    for class_idx, (rng, rng_name) in enumerate(zip(RANGES, RANGE_NAMES)):
        low, high = rng

        n_total = SAMPLES_PER_CLASS
        n_2 = int(n_total * RATIO_2)
        n_3 = int(n_total * RATIO_3)
        n_4 = n_total - n_2 - n_3  # 剩余给 4 步，避免浮点误差

        print(
            "[Info] Building class", rng_name,
            "total", n_total,
            "2-step", n_2,
            "3-step", n_3,
            "4-step", n_4,
        )

        # 2 步
        for _ in range(n_2):
            operands, operators, value = sample_expression_for_range(low, high, num_steps=2)
            expr_text = build_expr_text(operands, operators)
            program_text = build_structural_program(operands, operators)

            sample = {
                "id": global_id,
                "expr": expr_text,
                "answer": value,
                "label": class_idx,
                "domain": rng_name,
                "num_steps": 2,
                "program": program_text,
            }
            samples.append(sample)
            global_id = global_id + 1

        # 3 步
        for _ in range(n_3):
            operands, operators, value = sample_expression_for_range(low, high, num_steps=3)
            expr_text = build_expr_text(operands, operators)
            program_text = build_structural_program(operands, operators)

            sample = {
                "id": global_id,
                "expr": expr_text,
                "answer": value,
                "label": class_idx,
                "domain": rng_name,
                "num_steps": 3,
                "program": program_text,
            }
            samples.append(sample)
            global_id = global_id + 1

        # 4 步
        for _ in range(n_4):
            operands, operators, value = sample_expression_for_range(low, high, num_steps=4)
            expr_text = build_expr_text(operands, operators)
            program_text = build_structural_program(operands, operators)

            sample = {
                "id": global_id,
                "expr": expr_text,
                "answer": value,
                "label": class_idx,
                "domain": rng_name,
                "num_steps": 4,
                "program": program_text,
            }
            samples.append(sample)
            global_id = global_id + 1

    print("[Info] Total samples:", len(samples))

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("[Done] Saved to", output_jsonl)


if __name__ == "__main__":
    # 可以按需修改输出文件名和随机种子
    build_step_decomposer_dataset("mixed_arith_step_decomposer_80k2.jsonl", seed=2025)
