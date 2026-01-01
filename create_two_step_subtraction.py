import json
import random
from typing import List, Tuple

TOTAL_SAMPLES = 20000
RANGES: List[Tuple[int, int]] = [
    (2, 20),
    (21, 40),
    (41, 60),
    (61, 80),
    (81, 100),
]
RANGE_NAMES = [
    "range_2_20",
    "range_21_40",
    "range_41_60",
    "range_61_80",
    "range_81_100",
]

SAMPLES_PER_DOMAIN = TOTAL_SAMPLES // len(RANGES)
OUTPUT_PATH = "two_step_subtraction_20k.jsonl"


def build_two_step_program(a: int, b: int) -> str:
    """
    Describe the two-step decomposition for the subtraction.
    """
    return f"Step 1: {a} + {b} = x1\nAnswer: x1"


def sample_pair(answer: int) -> Tuple[int, int]:
    """
    Keep the sample within reasonable operand ranges.
    """
    minuend = answer + random.randint(0, 50)
    subtrahend = minuend - answer
    return minuend, subtrahend


def main(seed: int = 2025) -> None:
    random.seed(seed)
    samples = []
    global_id = 0

    for label, (rng, rng_name) in enumerate(zip(RANGES, RANGE_NAMES)):
        low, high = rng
        for _ in range(SAMPLES_PER_DOMAIN):
            answer = random.randint(low, high)
            minuend, subtrahend = sample_pair(answer)
            question = f"What is {minuend} minus {subtrahend}?"
            program = build_two_step_program(minuend, subtrahend)
            samples.append(
                {
                    "id": global_id,
                    "question": question,
                    "answer": answer,
                    "label": label,
                    "domain": rng_name,
                    "num_steps": 2,
                    "program": program,
                }
            )
            global_id += 1

    random.shuffle(samples)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} two-step subtraction samples in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
