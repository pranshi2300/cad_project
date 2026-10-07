# Literature Review & Comparative Analysis
## Adaptive Security & Resource Risk Gateway vs. Related Work

> *Use this for professor meetings and future paper writing.*

---

## 1. Paper Summaries

### SHIELD — arXiv:2601.19174
Three-stage defense pipeline: (1) semantic similarity vs. known sponge vectors, (2) KMP substring matching, (3) full LLM reasoning agent. Self-healing loop: when an attack slips through, auxiliary agents update the knowledgebase and refine the defense LLM's prompt.

**Limitations explicitly stated in paper:**
- Stage 3 invokes a **full LLM per query** — high latency at scale (the paper itself acknowledges this).
- Evaluation on single target LLM (Llama2-7B); no multi-service generalization.
- No graduated execution tiers; no compute-cost model.
- No adversarial injection coverage — exhaustion attacks only.
- Training-free (no empirical ML model).

---

### Mélange — arXiv:2404.14527
GPU allocation framework formulated as a **cost-aware bin packing ILP**. Routes requests to cheapest GPU type (H100 for large/latency-sensitive, L4 for small/loose-SLO). Saves 9–77% deployment cost.

**What it does NOT do:**
- Zero security awareness. Sponge attacks would be routed to cheap GPUs — still burning compute.
- No real-time adaptation; ILP solved offline.

**Paper's own future work:** spot instance integration, dynamic reallocation, prefill/decode phase disaggregation.

---

### Jiang et al. — arXiv:2502.00722
MILP-based heterogeneous GPU scheduler. Maps compute-vs-memory-bound requests to appropriate GPU types.

**Gaps:** Security-agnostic. Static pre-deployment plan. No risk scoring.

---

### BOute — arXiv:2602.10729
Co-optimizes **model heterogeneity** (routes simpler queries to smaller models) AND **GPU heterogeneity** using Multi-Objective Bayesian Optimization. 15–61% cost reduction.

**Critical gap:** Routes by complexity/quality — a sponge prompt that appears "semantically complex" gets routed to a *more expensive* large-GPU model. No adversarial safety signal.

---

### SGLang / RadixAttention — arXiv:2312.07104
RadixAttention: radix-tree KV-cache sharing across requests with shared prefixes. Up to 6.4× throughput improvement.

**Unexplored vulnerability:** Shared prefix pools can propagate adversarial cached context to benign subsequent requests. Our gateway closes this gap.

---

### US Patent 12,437,058 — Amazon Technologies
Binary classifier detecting indirect prompt injection in **agentic tool-call results**. Terminates session on detection.

**Limitations:** Binary only, injection only, agentic context only. No resource-exhaustion coverage. No graduated tiers. No adaptive retraining.

---

## 2. Comparison Table

| Dimension | **Our Gateway** | SHIELD | Mélange/BOute | SGLang | US Pat. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Injection detection | ✅ | ❌ | ❌ | ❌ | ✅ (partial) |
| Exhaustion/sponge detection | ✅ | ✅ | ❌ | ❌ | ❌ |
| Unified single risk score | ✅ | ❌ | ❌ | ❌ | ❌ |
| Graduated tiers (Full/Sandbox/Quarantine) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Sub-5ms, pre-GPU interception | ✅ | ❌ | ❌ | ❌ | ❌ |
| Empirical ML on real datasets | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dense semantic embeddings | ✅ | ✅ (Stage 1) | ❌ | ❌ | ❌ |
| GPU cost-aware routing | ✅ | ❌ | ✅ | ❌ | ❌ |
| Security + cost co-design | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 3. Our Novel Contributions

**Novel #1: Security-Aware GPU Cost Co-Routing**
No existing paper combines a security risk score with GPU cost-tier routing. Mélange/BOute route by cost only; SHIELD/PD3F detect attacks but have no cost-tier output. We are the first to use `Risk = 1 − P(benign)` to simultaneously prevent security violations AND save GPU compute in a unified decision.

**Novel #2: Unified Multi-Vector Risk in One Classifier**
SHIELD handles exhaustion. US 12,437,058 handles injection. PD3F handles DoS. Our single XGBoost model captures all three threat vectors simultaneously with one graduated continuous score — compound threat handling (e.g., a jailbreak embedded in a sponge request) is inherently covered.

**Novel #3: Pre-GPU Interception (Zero Defense Compute Cost)**
SHIELD's Stage 3 and PD3F still run GPU inference to detect attacks. Our gateway intercepts on a CPU proxy — the defense itself costs $0.00 GPU compute. This matters especially at scale.

**Novel #4: KV-Cache Adversarial Contamination Prevention**
SGLang's RadixAttention shares prefixes for efficiency but never validates prefix safety. Our gateway's pre-interception of high-risk requests prevents malicious prefixes from entering the shared cache — a new attack vector we implicitly close that no other paper addresses.

---

## 4. New Ideas to Incorporate (From Literature Gaps)

**From SHIELD's acknowledged limitation:**
- Replace SHIELD's Stage 3 (full LLM invocation) with our XGBoost classifier as a drop-in. Demonstrate latency improvement while maintaining accuracy. This directly advances SHIELD's own stated future work.

**From Mélange/Jiang/BOute:**
- Add **security-conditioned ILP constraint**: high-risk requests are never routed to GPUs with large shared KV-cache pools (prevents cache eviction of benign prefixes). This is a direct co-design contribution that neither security nor cost papers have attempted.
- **Spot Instance Safety Routing:** Low-risk requests → preemptible spot GPUs. Sandbox-tier → on-demand capped. Quarantine → dropped. Mélange explicitly lists spot integration as future work; we can build it with a security condition.

**From SGLang:**
- Formally model and quantify KV-cache contamination risk from unchecked prefix reuse. Show empirically how our gateway prevents it. First paper to formally address this threat vector.

---

## 5. GPU Testing Requirements — For Lab Assistant

Give this to your lab assistant verbatim:

---

### What I Need from the GPU Lab

**Hardware (any of these):**
- NVIDIA A100 (80GB or 40GB) — preferred
- NVIDIA V100, A40, RTX 3090/4090
- For heterogeneous test: 2 different GPU types (e.g., A100 + RTX 3080)

**Software to set up on the machine:**
```bash
# CUDA 12.1+
# Ubuntu 22.04

pip install vllm
pip install sentence-transformers joblib xgboost tiktoken flask
# Download llama3.2:3b via Ollama OR use vLLM with any HF model
```

**Tests to run (I'll give you exact scripts — you just need to launch them and share outputs):**

| Test | What You'll Run | Metrics to Capture |
| :--- | :--- | :--- |
| T1 – Baseline GPU Load | vLLM serving benign requests | `nvidia-smi`: VRAM used (MB), GPU compute %, tokens/sec |
| T2 – Sponge Attack (No Gateway) | 50 concurrent 100k-word requests → vLLM direct | VRAM spike, OOM crash?, benign user throughput collapse |
| T3 – Sponge Attack (With Gateway) | Same 50 requests → our gateway → vLLM | VRAM flat (attacks blocked), benign throughput unchanged |
| T4 – Gateway Feature Latency | 1,000 requests through gateway only (no LLM) | p50/p95/p99 latency in ms — must be under 10ms |
| T5 – KV-Cache Contamination | vLLM RadixAttention + benign prefix + sponge prefix | Cache hit rate with vs. without gateway pre-filtering |
| T6 – Heterogeneous GPU Routing | Gateway routes FULL-tier → A100, SANDBOX → smaller GPU | Cost per token on each tier vs. single GPU baseline |

**Logging command for all tests:**
```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory \
           --format=csv --loop-ms=500 > gpu_log.csv
```

**Just provide me:**
1. The `gpu_log.csv` from each test
2. Any OOM / crash logs from vLLM
3. The vLLM serving logs showing tokens/sec per test
