# 4. Experiments



This section presents a controlled empirical evaluation of the proposed modular reasoning framework.

The model uses a single shared backbone together with multiple lightweight expert heads, each trained

exclusively on a disjoint numerical sub-domain. This setup allows clear measurement of three core

modular properties:



1. **Local specialization** — Each expert should excel within its designated numerical range.  

2. **Global recomposition** — A router must integrate experts into a coherent global predictor.  

3. **Structural robustness** — The system should remain stable when experts are added, removed, or replaced.



We evaluate the following research questions:



- **RQ1 — Performance:** Does the residual expert structure outperform global models, uniform ensembles, and MoE-style gating?  

- **RQ2 — Stability:** Do experts maintain consistent behavior across ranges, and does routing preserve balanced accuracy?  

- **RQ3 — Modularity:** Can experts trained independently be recombined without retraining the backbone?  

- **RQ4 — Efficiency:** Does modularization reduce computation cost while improving interpretability?



Two arithmetic tasks are used for evaluation:

(1) two-number addition, and (2) mixed addition–subtraction.



---



## 4.1 Experimental Overview



We evaluate three modular configurations:



- Linear expert heads  

- Nonlinear MLP experts  

- Residual Experts (RED) with a shared Qwen backbone  



Although arithmetic tasks appear simple, they expose essential modular behaviors such as specialization,

interference, and recomposition.



---



### Task Formulation



Each input is a natural-language arithmetic query:



> "What is 37 plus 12?"



Operands follow:



$$

a \in [1, 50]

$$



$$

b \in [1, 50]

$$



$$

s = a + b \in [2, 100]

$$



To induce expert specialization, the output space is partitioned into five disjoint numerical ranges:



$$

R_1: 2 \le s \le 20

$$



$$

R_2: 21 \le s \le 40

$$



$$

R_3: 41 \le s \le 60

$$



$$

R_4: 61 \le s \le 80

$$



$$

R_5: 81 \le s \le 100

$$



Each range corresponds to a dedicated expert head.



---



### Shared Encoder



All experts use a shared Qwen-0.6B encoder.  

Only the top two transformer blocks are trainable.



Let:



$$

h(x) \text{ = shared backbone representation}

$$



Each expert learns:



$$

E_i(h(x)) \rightarrow \hat{y}_i

$$



Only samples whose true label lies in range \(R_i\) are used to train expert \(E_i\).



---



### Router Training



After all experts converge, a lightweight router is trained to assign weights:



$$

\alpha(x) = \text{softmax}(W_r h(x))

$$



For the RED model, the final prediction is:



$$

\hat{y}(x) = h_{\text{base}}(x) + \sum_{i=1}^{5} \alpha_i(x)\,\Delta h_i(x)

$$



where \(\Delta h_i\) is the residual correction from expert \(i\).



---



### Why Arithmetic?



Arithmetic is an ideal modular testbed because:



- Label ranges define **explicit domain boundaries**  

- Errors are **fully interpretable**  

- No annotation noise exists  

- Difficulty increases across ranges  



This structure matches the empirical standards used in *Knowledge-Based Systems* research.



---



## 4.2 Dataset and Task Details



We study two synthetic tasks:

(1) addition, and  

(2) mixed addition–subtraction.  

Synthetic generation provides exact labels, controlled difficulty, and clean domain boundaries.



---



### 4.2.1 Task 1 — Two-Number Addition

Each query is:

"What is a plus b?"

Operands follow the same formulation defined in Section 4.1.

Output Space Partition

(Identical to ranges in Section 4.1; omitted here for brevity.)

Dataset Sizes



- 20,000 training  

- 2,500 validation  

- 2,500 test  

Split ratio: **8 : 1 : 1**



---



### 4.2.2 Task 2 — Mixed Arithmetic

Example expressions:

"What is 45 minus 17?"
"What is 12 plus 8 minus 3?"

Grammar supports one or two operators. Operand constraints follow Section 4.1.

The same five output ranges R₁–R₅ from Task 1 are reused.

Dataset sizes mirror Task 1.



- 20,000 training  

- 2,500 validation  

- 2,500 test  



---



### 4.2.3 Motivation for Synthetic Tasks

The motivations for synthetic arithmetic tasks follow directly from Section 4.1 (clean boundaries, exact labels, interpretability).



### 4.2.4 Summary

Both tasks share:

The common Qwen-0.6B backbone

Five experts (per Section 4.1)

A trainable router for recomposition

Balanced 8:1:1 data splits

This avoids external noise and focuses evaluation on specialization and routing behavior. specialization, compatibility, and routing behavior without external noise.
## 4.3 Baselines

To contextualize the effectiveness of the proposed Residual Expert Decomposition (RED), we compare it against four representative baseline systems widely used in modular reasoning, Mixture-of-Experts, and ensemble learning. These baselines span the spectrum from non-modular global models to routing-driven expert systems.

---

### 4.3.1 Baseline 1 — Global Head (Single Expert)

A single classifier head is trained on all samples using the same shared backbone as RED.  
This baseline contains:

- no modular decomposition,  
- no expert specialization,  
- no routing mechanism.

It answers the question:

> *Is a single global model sufficient to learn heterogeneous numerical reasoning patterns?*

---

### 4.3.2 Baseline 2 — Oracle Per-Range Experts *(Upper bound)*

Five expert heads are trained separately, each using only samples from its own numerical sub-domain.  
At inference, the correct expert is selected using ground-truth label ranges:

$$
p(y \mid x) = \mathrm{softmax}(E_i(h(x))), \quad i = \mathrm{trueRange}(x)
$$

This is not feasible in practice (true ranges are unknown), but it represents the **upper bound** of expert specialization.

---

### 4.3.3 Baseline 3 — Uniform Expert Averaging *(Parameter-free ensemble)*

All expert logits are averaged with equal weights:

$$
p(y \mid x) = \frac{1}{5} \sum_{i=1}^5 \mathrm{softmax}(E_i(h(x))).
$$

This tests whether simple ensembling (without routing or residual correction) is sufficient.

---

### 4.3.4 Baseline 4 — Softmax Gating (Classical MoE)

A softmax router computes mixture weights:

$$
\\alpha(x) = \mathrm{softmax}(W_r h_{\mathrm{base}}(x)).
$$

Final prediction:

$$
p(y \mid x) = \sum_{i=1}^5 \\alpha_i(x) \, E_i(h(x)).
$$

This is the strongest structural competitor to RED, but it lacks residual correction and is more sensitive to routing errors.

---

### 4.3.5 Summary

| Baseline | Routing | Modularity | Residual Correction | Purpose |
|----------|---------|------------|----------------------|---------|
| Global Head | ✗ | ✗ | ✗ | Non-modular reference |
| Oracle Experts | ✓ (perfect) | ✓ | ✗ | Upper-bound specialization |
| Uniform Avg | ✗ | ✓ | ✗ | Parameter-free ensemble |
| Softmax Gating | ✓ | ✓ | ✗ | Strong MoE competitor |
| **RED (Ours)** | ✓ | ✓ | ✓ | Full specialist integration |
## 4.4 Our Method: Residual Expert Decomposition (RED)

Residual Expert Decomposition (RED) is designed to integrate independently trained numerical experts 
through a shared semantic backbone and a residual correction mechanism. 
Unlike classical Mixture-of-Experts (MoE), which directly interpolates expert logits, 
RED decomposes each prediction into:

1. **Backbone semantic signal**  
2. **Expert-specific residual adjustments**  
3. **Router-determined expert mixing**

This structure allows experts to remain independent while still contributing cooperatively at inference time.

---

### 4.4.1 Residual Expert Structure (Core Idea)

Given a shared backbone representation:

$$
h_{\text{base}}(x) \in \mathbb{R}^d,
$$

each expert \(E_i\) produces a residual correction rather than a full prediction.  
The residual mapping is:

$$
\Delta h_i(x)
= V_i \, \sigma( U_i \, h_{\text{base}}(x) ),
$$

where:
$$
 (U_i \in \mathbb{R}^{d \times r})  
 (V_i \in \mathbb{R}^{r \times C})  
 (r \ll d) (low-rank bottleneck)
$$
Thus each expert learns **how** to adjust the backbone, not to replace it.

---

### 4.4.2 Router for Expert Mixing

A lightweight router predicts expert mixing weights:

$$
\alpha(x) = \mathrm{softmax}(W_r h_{\text{base}}(x)),
$$
$$
where (\alpha(x) \in \mathbb{R}^5).
$$
The router is trained *after* all experts converge, ensuring experts remain independent modules.

---

### 4.4.3 Residual Decomposition (RED Forward Pass)

RED combines backbone semantics with expert corrections:

$$
\hat{h}(x)
= h_{\text{base}}(x)
+ \sum_{i=1}^{5} \alpha_i(x)\, \Delta h_i(x).
$$

The final logits are computed using a shared classifier:

$$
p(y \mid x) = \mathrm{softmax}(W_c \hat{h}(x)).
$$

Key properties:

- Backbone provides domain-general semantics  
- Experts add range-specific corrections  
- Router controls mixture strength  
- Residual formulation prevents destructive expert interference  

---

### 4.4.4 Why RED Improves Modular Stability

**(A) Additive, not substitutive expert influence**  
Experts do not overwrite the backbone representation; they refine it.  
This preserves global semantics and prevents drift across ranges.

**(B) Robustness to routing errors**  
Even if the router imperfectly assigns weights, residual corrections remain small and stable, avoiding MoE-style collapse.

**(C) Specialized but cooperative experts**  
Each \($\Delta h_i$\) focuses only on the numerical dimensions relevant to its range, 
while residual fusion allows experts to help each other on boundary samples.

---

### 4.4.5 Summary of Advantages

| Property | Classical MoE | Uniform Avg | RED (Ours) |
|---------|---------------|-------------|------------|
| Expert interaction | direct interpolation | equal mixing | residual correction |
| Robustness to routing errors | low | medium | **high** |
| Plug-and-play modularity | partial | yes | **fully supported** |
| Backbone preservation | partial | full | **full** |
| Per-range performance | unstable | weak | **balanced + strong** |

RED provides a principled mechanism for integrating independently trained experts with robustness and interpretability, 
making it suitable for modular reasoning systems.
## 4.5 Training Setup

This section describes the unified training pipeline used for all models and baselines. Unless otherwise noted, the same data, backbone configuration, optimization settings, and hardware are applied to the global head, oracle experts, uniform averaging, MoE gating, and RED.

---

### 4.5.1 Backbone Configuration

All systems use **Qwen3-0.6B** as the shared encoder.

- Frozen layers: bottom 28 transformer blocks  
- Trainable layers: top 2 transformer blocks  
- Precision: mixed-precision FP16  
- Pooling: final hidden states are mean-pooled to obtain a fixed-length vector representation.

This setup preserves general linguistic and arithmetic capability while allowing limited task-specific adaptation.

---

### 4.5.2 Expert Training (Stage 1)

Each expert head is trained **only** on samples whose correct answer lies in its assigned numerical range **R₁–R₅**, as defined in Section 4.1. The backbone parameters are shared across experts and remain fixed except for the top two layers.

For linear experts we use the mapping:

$$
h_i(x) = W_i \, h_{\text{base}}(x) + b_i.
$$

For RED, the expert definition follows the residual form in Section 4.4 but is trained on the same per-range subsets.

All experts share the same optimization settings:

| Component       | Value                            |
|-----------------|----------------------------------|
| Optimizer       | AdamW                           |
| Learning rate   | $1\times 10^{-4}$               |
| Epochs          | 8                                |
| Loss            | Cross-entropython                    |
| Batch size      | 16 (accumulated to 128)          |
| Early stopping  | Not used (experts converge fast) |

---

### 4.5.3 Router Training (Stage 2)

Once expert heads have converged, we train a router that predicts mixture weights over the five experts:

$$
\alpha(x) = \mathrm{softmax}(W_r h_{\text{base}}(x)).
$$

The router is implemented as a two-layer MLP with hidden size 256 and output dimension 5.

It is optimized with cross-entropython loss over the ground-truth range labels defined in Section 4.2, using the same AdamW optimizer and learning rate as the experts. No additional regularization is required to reach stable validation accuracy around 0.96–0.97.

---

### 4.5.4 RED Training (Residual Integration, Stage 3)

For RED, we introduce low-rank residual mappings:

$$
\Delta h_i(x) = V_i \, \sigma(U_i h_{\text{base}}(x)),
$$

with

$$
U_i \in \mathbb{R}^{r \times d}, \quad V_i \in \mathbb{R}^{d \times r}, \quad r = d/4.
$$

Only the residual parameters and router are updated in this stage; the backbone remains fixed.

The final representation is

$$
\hat{h}(x) = h_{\text{base}}(x) + \sum_{i=1}^5 \alpha_i(x) \, \Delta h_i(x),
$$

and logits are obtained via a shared classifier:

$$
p(y \mid x) = \mathrm{softmax}(W_c \hat{h}(x)).
$$

This stage adds less than 1% extra parameters relative to the backbone.

---

### 4.5.5 Data Generation and Splits

For each task (addition and mixed arithmetic), we reuse the synthetic datasets defined in Section 4.2. Concretely, we generate:

- 20,000 training examples  
- 2,500 validation examples  
- 2,500 test examples  

for a total of 25,000 instances per task, following an **8:1:1** split. Labels are integers in $[2, 100]$, giving 99 classes.

All datasets are noise-free and reproducible using fixed random seeds.

---

### 4.5.6 Hardware and Runtime

All experiments are run on a single **NVIDIA RTX 3060 Ti (8 GB VRAM)** using pythonTorch with FP16. Measured runtimes are:

- Backbone pre-adaptation + global head: ≈ 1.5 h  
- Per-range experts (all five): ≈ 35 min total  
- Router + RED residuals: ≈ 1 h  

Total end-to-end training time is around **3–4 hours**. Peak VRAM usage remains below 6.5 GB for all configurations, so RED fits comfortably on a consumer-grade GPU.

---

### 4.5.7 Summary

| Stage                     | Trainable components                     | Purpose                                   |
|---------------------------|------------------------------------------|-------------------------------------------|
| Stage 1 – Base            | Top-2 backbone layers + global head      | Learn global arithmetic semantics         |
| Stage 2 – Experts         | Five range-specific expert heads         | Capture local numerical patterns          |
| Stage 3 – Router          | Router MLP                               | Learn expert assignment over ranges       |
| Stage 4 – RED integration | Residual modules + router                | Resolve cross-expert incompatibilities    |

This standardized pipeline ensures that all models are directly comparable and that observed gains for RED stem from its structural design rather than unfair training advantages.
python shrink_checkpoint.py --in checkpoints/demo_v11_base_best_ep10.pt --trainable-last-n-layers 1
python shrink_checkpoint.py --in checkpoints/demo_v11_expert_0_best_ep60.pt --trainable-last-n-layers 1
python shrink_checkpoint.py --in checkpoints/demo_v11_expert_1_best_ep60.pt --trainable-last-n-layers 1
python shrink_checkpoint.py --in checkpoints/demo_v11_expert_2_best_ep60.pt --trainable-last-n-layers 1
python shrink_checkpoint.py --in checkpoints/demo_v11_expert_3_best_ep60.pt --trainable-last-n-layers 1
python shrink_checkpoint.py --in checkpoints/demo_v11_expert_4_best_ep60.pt --trainable-last-n-layers 1
python shrink_checkpoint.py --in checkpoints/demo_v11_router_best_ep8.pt --trainable-last-n-layers 1
