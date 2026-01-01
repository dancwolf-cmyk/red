import json
import random
from pathlib import Path

TOTAL_SAMPLES = 20000
ANSWER_MIN = 2
ANSWER_MAX = 99
LABELS = list(range(ANSWER_MIN, ANSWER_MAX + 1))
OUTPUT_PATH = Path("train_add_2_99.jsonl")


def build_sample(answer: int) -> dict:
    """Construct a single addition question whose sum is `answer`."""
    a = random.randint(0, answer)
    b = answer - a
    return {
        "question": f"What is {a} plus {b}?",
        "answer": answer,
    }


def main(seed: int = 42) -> None:
    random.seed(seed)

    base_count = TOTAL_SAMPLES // len(LABELS)
    remainder = TOTAL_SAMPLES % len(LABELS)

    samples = []
    for i, ans in enumerate(LABELS):
        count = base_count + (1 if i < remainder else 0)
        for _ in range(count):
            samples.append(build_sample(ans))

    random.shuffle(samples)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} addition samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
