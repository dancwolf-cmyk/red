import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = Path("e:/dev/lunwen/qwen3-0.6b")
DATA_PATH = Path("./mixed_arith_step_decomposer_80k2.jsonl")
CHECKPOINT_DIR = Path("./checkpoints_step_decomposer")
BEST_PATTERN = "step_decomposer_best_ep*.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2025


def build_prompt(expr: str) -> str:
    return (
        "Decompose the following arithmetic expression into step-by-step computation "
        "using variables x1, x2, ...\n"
        "Expression: "
        + expr
        + "\n\n"
        "Program:\n"
    )


def load_dataset(path: Path) -> list:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
    return samples


def select_samples(samples: list, counts: dict, seed: int) -> list:
    random.seed(seed)
    selected = []
    for steps, c in counts.items():
        subset = [s for s in samples if s.get("num_steps") == steps]
        if len(subset) < c:
            raise ValueError(f"Not enough {steps}-step samples (need {c}, got {len(subset)})")
        selected.extend(random.sample(subset, c))
    return selected


def find_best_checkpoint(dir_path: Path) -> Path:
    candidates = list(dir_path.glob(BEST_PATTERN))
    if not candidates:
        raise FileNotFoundError("No best checkpoint found in {}".format(dir_path))

    def epoch_key(path: Path) -> int:
        m = re.search(r"ep(\d+)", path.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=epoch_key)


def truncate_after_answer(text: str) -> str:
    match = re.search(r"Answer:\s*x\d+", text)
    if not match:
        return text
    end = match.end()
    remainder = end
    # keep trailing newline if present
    if end < len(text) and text[end] == "\n":
        remainder = end + 1
    return text[:remainder]


def main():
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"{CHECKPOINT_DIR} does not exist")

    samples = load_dataset(DATA_PATH)
    counts = {2: 2, 3: 4, 4: 4}
    selected = select_samples(samples, counts, SEED)

    checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
    print("[Info] Loading best checkpoint", checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    for idx, sample in enumerate(selected, start=1):
        expr = sample["expr"]
        gold = sample["program"]
        prompt = build_prompt(expr)

        tokenized = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = tokenized["input_ids"].to(DEVICE)
        attention_mask = tokenized["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        gen_ids = outputs[0][input_ids.size(1) :]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = truncate_after_answer(pred).strip()

        print(f"\n--- Sample {idx} ---")
        print("Expression:", expr)
        print("Num steps:", sample.get("num_steps"))
        print("Gold program:")
        print(gold)
        print("Generated program:")
        print(pred)


if __name__ == "__main__":
    main()
