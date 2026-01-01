# -*- coding: utf-8 -*-
"""
generate_balanced_mixed_samples.py
----------------------------------

生成 balanced 混合算术样本：
- 总样本数 = 80,000
- 标签范围 2~100，共 99 个标签
- 每个标签均衡分布
- 每个标签内部：
    2-step 50%
    3-step 30%
    4-step 20%
- 输出与 demo_v13 格式一致
"""

import json
import random
from pathlib import Path

SAVE_PATH = Path("samples_80k_balanced.json")
TOTAL_SAMPLES = 160000

# 5 个区间，用于 domain 字段
RANGES = {
    "range_2_20": list(range(2, 21)),
    "range_21_40": list(range(21, 41)),
    "range_41_60": list(range(41, 61)),
    "range_61_80": list(range(61, 81)),
    "range_81_100": list(range(81, 101)),
}

ALL_LABELS = []
for rs in RANGES.values():
    ALL_LABELS.extend(rs)

NUM_LABELS = len(ALL_LABELS)  # 99
PER_LABEL = TOTAL_SAMPLES // NUM_LABELS  # 80000/99 ≈ 808

# step 配比
STEP2_RATIO = 0.50
STEP3_RATIO = 0.30
STEP4_RATIO = 0.20

def gen_2step_expression(ans):
    """
    ans = a (+|-) b
    生成一个随机 2 步表达式并保证结果正确。
    """
    op = random.choice(["+", "-"])

    if op == "+":
        a = random.randint(0, ans)
        b = ans - a
    else:  # a - b = ans
        a = random.randint(ans, ans + 50)
        b = a - ans

    return f"What is {a} {op} {b}?", ans


def gen_3step_expression(ans):
    """
    ans = a (+|-) b (+|-) c
    两层运算
    """
    op1 = random.choice(["+", "-"])
    op2 = random.choice(["+", "-"])

    # 先构造中间结果 x = a op1 b
    # 让 a,b 范围大点，增加随机性
    a = random.randint(0, ans + 50)
    if op1 == "+":
        b = random.randint(0, a + 30)
        x = a + b
    else:
        b = random.randint(0, a)
        x = a - b

    # x op2 c = ans → 求 c
    if op2 == "+":
        c = ans - x
    else:
        c = x - ans

    # c 要尽量合理
    if abs(c) > 200:
        c = random.randint(0, 100)

    return f"What is {a} {op1} {b} {op2} {c}?", ans


def gen_4step_expression(ans):
    """
    ans = a (+|-) b (+|-) c (+|-) d
    """
    op1 = random.choice(["+", "-"])
    op2 = random.choice(["+", "-"])
    op3 = random.choice(["+", "-"])

    # 先构造中间 result1 = a op1 b
    a = random.randint(0, ans + 80)
    if op1 == "+":
        b = random.randint(0, a + 50)
        r1 = a + b
    else:
        b = random.randint(0, a)
        r1 = a - b

    # result2 = r1 op2 c
    if op2 == "+":
        c = random.randint(0, 80)
        r2 = r1 + c
    else:
        c = random.randint(0, r1 if r1 > 0 else 10)
        r2 = r1 - c

    # 最后一层： r2 op3 d = ans
    if op3 == "+":
        d = ans - r2
    else:
        d = r2 - ans

    # d 合理化
    if abs(d) > 200:
        d = random.randint(0, 100)

    return f"What is {a} {op1} {b} {op2} {c} {op3} {d}?", ans


def domain_of_label(label):
    """
    返回 domain 例如 'range_41_60'
    """
    for dom, values in RANGES.items():
        if label in values:
            return dom
    raise ValueError("标签不属于任何范围: {}".format(label))


def generate_all():
    samples = []
    random.seed(42)

    for label in ALL_LABELS:
        dom = domain_of_label(label)

        # 本标签 808 条
        n2 = int(PER_LABEL * STEP2_RATIO)
        n3 = int(PER_LABEL * STEP3_RATIO)
        n4 = PER_LABEL - n2 - n3

        # 2-step
        for _ in range(n2):
            q, ans = gen_2step_expression(label)
            samples.append({"question": q, "answer": ans, "domain": dom})

        # 3-step
        for _ in range(n3):
            q, ans = gen_3step_expression(label)
            samples.append({"question": q, "answer": ans, "domain": dom})

        # 4-step
        for _ in range(n4):
            q, ans = gen_4step_expression(label)
            samples.append({"question": q, "answer": ans, "domain": dom})

    # 目前数量 808*99 = 79992，需要补到 80000
    while len(samples) < TOTAL_SAMPLES:
        label = random.choice(ALL_LABELS)
        dom = domain_of_label(label)
        q, ans = gen_2step_expression(label)
        samples.append({"question": q, "answer": ans, "domain": dom})

    random.shuffle(samples)
    return samples


def main():
    samples = generate_all()
    label_vocab = [str(x) for x in ALL_LABELS]

    obj = {
        "samples": samples,
        "label_vocab": label_vocab,
    }

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved to: {SAVE_PATH} (n={len(samples)})")


if __name__ == "__main__":
    main()
