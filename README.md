# Residual Expert Decomposition (RED)
Robust Post-hoc Recomposition under Routing Noise

This repository contains research code and reproducible scripts for a **modular reasoning** framework:
a single shared backbone with multiple lightweight experts, trained on disjoint sub-domains, and a router
for post-hoc recomposition. The codebase includes:
- a controlled **MNIST label-pair / 5-domain** modular benchmark,
- **natural-language arithmetic** (addition / mixed operations) experiments,
- a **step decomposer** (program synthesis style) for mixed arithmetic.

> Note on checkpoints: large `.pt` files can be difficult to push to GitHub. This repo is intended to
> keep **code + data + minimal artifacts** in Git, and publish large weights via **Git LFS** or **GitHub Releases**.
> In particular, do not commit `*_last_*.pt` unless you deliberately use LFS.

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
- `mixed_arith_step_decomposer_80k*.jsonl`, `train_ops_balanced*.jsonl`, `train_add_2_99.jsonl` — datasets (JSONL)

Directories:
- `checkpoints/` — arithmetic experiment logs & checkpoints (may be large)
- `ckpt_mnist_v11_label5/` — MNIST benchmark artifacts (logs, metrics JSON, splits, etc.)
- other `checkpoints_*` folders — alternative runs/variants

---

## Environment

### Python
Recommended:
- Python 3.10+ (tested with 3.11)

### Install dependencies
```bash
pip install -r requirements.txt
