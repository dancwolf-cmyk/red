# Residual Expert Decomposition (RED)
Robust Post-hoc Recomposition under Routing Noise

This repository contains research code and reproducible scripts for a **modular reasoning** framework: a single shared backbone with multiple lightweight experts, trained on disjoint sub-domains, and a router for post-hoc recomposition. The codebase includes:

- A controlled **MNIST label-pair / 5-domain** modular benchmark
- **Natural-language arithmetic** (addition / mixed operations) experiments
- A **step decomposer** (program synthesis style) for mixed arithmetic

> Checkpoint policy: this repository aims to keep **code + data + lightweight artifacts** in git. Large weights can be distributed via **GitHub Releases** or **Git LFS**. All experiments are designed to be reproducible by retraining.

---

## Repository layout

Key scripts / assets in the repo root:

- `demo_mnist_v11_label5.py` — MNIST modular benchmark (5 fixed domains, RED vs MoE vs baselines)
- `demo_v11.py`, `demo_v13.py`, `demo_v11_moe.py`, etc. — arithmetic experiments (variants)
- `train_op_expert_v13.py`, `train_router_ops.py`, `train_moe_router_ops.py` — expert/router training for mixed ops
- `train_step_decomposer_qwen06b.py` — train a step-by-step decomposer for mixed arithmetic
- `test_step_decomposer.py` — load best decomposer checkpoint and run a few samples
- `create_add_2_99_jsonl.py`, `create_balanced_ops_jsonl.py`, `create_two_step_subtraction.py` — dataset generators
- `eval_slim.py`, `shrink_checkpoint.py` — evaluate / shrink checkpoints for easier sharing

Datasets (JSONL) typically live in the repo root (examples):

- `mixed_arith_step_decomposer_80k*.jsonl`
- `train_ops_balanced*.jsonl`
- `train_add_2_99.jsonl`

Artifact directories (may be large; often ignored by default):

- `checkpoints/` — arithmetic experiment logs & checkpoints
- `ckpt_mnist_v11_label5/` — MNIST benchmark artifacts (logs, metrics JSON, splits)
- `checkpoints_*` — alternative runs/variants

---

## Environment

### Python

- Python 3.11

### Install dependencies

```bash
pip install -r requirements.txt
```

### GPU / CUDA

Install a PyTorch build that matches your CUDA environment.

To record versions for reproducibility:

```bash
python -c "import torch; print('torch', torch.__version__); print('torch_cuda_build', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

nvidia-smi
```

---

## External model dependency (Qwen backbone)

Some experiments use a Hugging Face causal LM backbone (e.g., **Qwen3-0.6B**). Model weights are **not** included in this repository.

Typical options:

1. Download the model from Hugging Face to a local folder (recommended for offline runs).
2. Point scripts to that local folder path.

Important:

- `test_step_decomposer.py` uses hard-coded paths by default (e.g., `MODEL_PATH`, `DATA_PATH`, `CHECKPOINT_DIR`). Edit these constants to match your environment before running.

---

## Quick start

### 1) MNIST modular benchmark (5 domains)

Run RED and MoE training/evaluation on MNIST with a fixed 5-domain mapping (label-pair domains):

```bash
python demo_mnist_v11_label5.py \
  --run all \
  --device cuda \
  --data_dir ./data_mnist \
  --save_dir ./ckpt_mnist_v11_label5
```

Common options:

- `--run red|moe|all`
- `--device cpu|cuda`
- `--batch_size`, `--virtual_batch_size`
- `--max_epochs_*`, `--patience`

To see all arguments:

```bash
python demo_mnist_v11_label5.py --help
```

Outputs are written under `--save_dir`:

- Logs: `*.log`
- Cached splits: `splits.json` (if used)
- Metrics JSON: `*results*.json` (overall + per-domain)
- Checkpoints: `*_best_*.pt` (if enabled)

---

### 2) Natural-language arithmetic (addition / mixed ops)

There are multiple experiment variants in `demo_v*.py`. The typical workflow is:

1. Generate or load JSONL datasets
2. Train base head (optional, depending on variant)
3. Train domain experts
4. Train router (or MoE gating)
5. Evaluate overall + per-domain metrics

Start by checking the variant you want:

```bash
python demo_v13.py --help
# or
python demo_v11.py --help
```

If you want to (re)generate datasets:

```bash
python create_add_2_99_jsonl.py --help
python create_balanced_ops_jsonl.py --help
python create_two_step_subtraction.py --help
```

---

### 3) Step decomposer (mixed arithmetic program generation)

Train:

```bash
python train_step_decomposer_qwen06b.py --help
```

Test (after editing paths inside `test_step_decomposer.py`):

```bash
python test_step_decomposer.py
```

Expected files:

- Dataset: `mixed_arith_step_decomposer_80k2.jsonl` (or similar)
- Checkpoints: `./checkpoints_step_decomposer/step_decomposer_best_ep*.pt`

---

## Checkpoint policy (recommended)

### What to commit

- Code (`.py`)
- Datasets (or dataset generators)
- Small result artifacts (`.json`, `.csv`)
- Logs (`*.log`) if not too large
- Optionally: **best** / **slim** checkpoints if they are small enough

### What NOT to commit

- Very large `.pt` files without Git LFS
- Frequent training artifacts such as `*_last_*.pt`

Suggested `.gitignore` patterns:

```gitignore
# Python cache
__pycache__/
*.pyc

# Large training artifacts
*_last_*.pt
checkpoints/
checkpoints_*/
ckpt_*/

# If you want to keep only "best" checkpoints, use this instead of ignoring all:
# checkpoints/**/*_last_*.pt
# checkpoints/**/*_ep*.pt
# !checkpoints/**/*_best*.pt
# !checkpoints/**/*.slim*.pt
```

### Shrinking checkpoints

If you need to share weights without LFS, try the included utilities:

```bash
python shrink_checkpoint.py --help
python eval_slim.py --help
```

---

## Reproducibility notes

- Most scripts set a fixed seed and write logs + JSON artifacts.
- For best reproducibility: run on a single GPU, keep the same PyTorch/CUDA versions, and avoid changing tokenizer/model files.

---

## Citation

If you use this code in academic work, please cite the accompanying manuscript. A `CITATION.cff` file can be added to the repository root.

---

## License

MIT License (see `LICENSE`).

---

## Contact / Issues

If you encounter issues, include:

1. The command you ran
2. The relevant log file
3. Your OS, Python, PyTorch, CUDA build, and GPU model(s)

Support email: 15876031@qq.com
