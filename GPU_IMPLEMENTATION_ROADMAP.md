# ⚡ GPU Implementation & Roadmap Guide
**Scaling Adaptive LLM Gateway from CPU/Simulated to Real Lab GPU + LLM Infrastructure**  
*Document Purpose: Technical blueprint for setting up, running, accelerating, and stress-testing the gateway against real GPU-backed LLM engines (vLLM / Ollama) on the lab workstation.*

---

## 📌 1. The Paradigm Shift: Why the Lab GPU Matters

Until now, the gateway has run in a **CPU / synthetic simulation mode**:
- Prompt embeddings were generated on CPU (taking ~10–15 ms per request).
- Cache-locality was modeled via synthetic probability distributions.
- LLM generation was either mocked or run via lightweight local CPU calls.

With access to the **Lab GPU** (e.g. NVIDIA RTX 3090/4090, A5000, A6000, or A100), we unlock the real-world reduction to practice:
1. **Sub-2ms Ingress Embedding:** Moving `all-MiniLM-L6-v2` to CUDA drops dense embedding latency to **< 1.8 ms**.
2. **Real VRAM / KV-Cache Telemetry:** Interfacing directly with **vLLM's PagedAttention** memory manager to monitor actual GPU KV-cache saturation in real-time.
3. **Empirical Demonstration of $0.00 GPU Cost:** Proving that 100k-word sponge attacks and DAN jailbreaks result in **zero VRAM spikes and zero GPU FLOPs consumed** on the physical hardware.

---

## 🛠️ 2. Lab GPU Environment Setup (Step-by-Step)

### Step 2.1: Verify NVIDIA Driver & CUDA
Open terminal on the Lab GPU workstation:
```bash
nvidia-smi
```
Check:
- GPU Model (e.g., NVIDIA RTX 4090 / A100)
- Driver Version (>= 535 recommended)
- CUDA Version supported (e.g., CUDA 12.1, 12.2, or 12.4)

### Step 2.2: Setup Dedicated Virtual Environment
```bash
# On Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2.3: Install PyTorch with CUDA
```bash
# For CUDA 12.1 / 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2.4: Install Repository Dependencies
```bash
pip install -r requirements.txt
```

### Step 2.5: Run Diagnostic Script
We have provided an automated diagnostic script [`test_gpu_pipeline.py`](file:///c:/Users/Reet/Desktop/cad_project/test_gpu_pipeline.py):
```bash
python test_gpu_pipeline.py
```
This script automatically checks:
- PyTorch CUDA device name and total VRAM (GB)
- SentenceTransformer GPU loading and 100-prompt embedding throughput
- Pre-trained model loading from `models/`
- Local LLM engine status (vLLM / Ollama)

---

## 🚀 3. Deploying the GPU LLM Backend: vLLM vs Ollama

We strongly recommend **vLLM** for the lab workstation because it exposes real PagedAttention memory metrics and native high-throughput serving.

### Option A: vLLM (Recommended for Research & Paper Benchmarks)

#### 1. Install vLLM
```bash
pip install vllm
```

#### 2. Launch vLLM OpenAI-Compatible Server
Run on the lab GPU:
```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enable-prefix-caching
```
*Note:* `--enable-prefix-caching` enables vLLM's automatic KV-cache prefix reuse, matching our gateway's cache-locality design.

#### 3. Why vLLM is Ideal: Live KV-Cache Metrics Endpoint
vLLM serves a Prometheus metrics endpoint at `http://localhost:8000/metrics`. This includes:
- `vllm:gpu_cache_usage_factor`: Current fraction of GPU VRAM KV-cache blocks allocated (0.0 to 1.0).
- `vllm:num_requests_waiting`: Requests waiting in queue due to memory saturation.
- `vllm:prefix_cache_hit_rate`: Real-time prefix cache hit ratio!

---

### Option B: Ollama (Alternative for Quick Plug-and-Play)

If vLLM is not convenient, use Ollama:
```bash
# 1. Download and run Llama 3.2 on GPU
ollama run llama3.2

# 2. Verify GPU utilization
nvidia-smi  # Should show ollama running on GPU
```
Ollama serves on `http://localhost:11434`.

---

## ⚡ 4. Accelerating the Ingress Gateway on GPU

To achieve the sub-5ms SLA on real hardware, optimize the ingress components:

### 4.1 Dense Embeddings on GPU (`features.py` / `app.py`)
In `features.py` and `app.py`, update the `SentenceTransformer` initialization:
```python
import torch
from sentence_transformers import SentenceTransformer

# Automatically use GPU if present
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Optional: Half-precision for 2x faster embedding on GPU
if device == "cuda":
    embedder = embedder.half()
```

### 4.2 Latency Comparison (Expected)
| Component | CPU (Intel/AMD) | Lab GPU (RTX 3090/4090 / A5000) |
| :--- | :---: | :---: |
| **Tokenization (`tiktoken`)** | 0.2 ms | 0.2 ms (CPU fast path) |
| **TF-IDF Lexical Match** | 0.5 ms | 0.5 ms (Sparse CPU) |
| **MiniLM Dense Embedding** | 12.0 ms | **1.2 ms (CUDA / FP16)** |
| **Cosine Dot-Product** | 0.1 ms | 0.05 ms |
| **XGBoost Inference** | 0.3 ms | 0.2 ms |
| **Total Ingress Overhead** | **~13.1 ms** | **~2.15 ms (< 5 ms SLA!)** |

---

## 🔗 5. Integrating Real GPU Telemetry into the Gateway

Instead of synthetic cache hit estimation, integrate live GPU KV-cache pressure into the routing decision:

```python
# snippet for router.py or app.py
import requests

def get_vllm_gpu_cache_pressure():
    try:
        resp = requests.get("http://localhost:8000/metrics", timeout=0.1)
        for line in resp.text.splitlines():
            if line.startswith("vllm:gpu_cache_usage_factor"):
                # e.g. "vllm:gpu_cache_usage_factor 0.74"
                return float(line.split()[1])
    except Exception:
        return 0.0
    return 0.0

def dynamically_adjust_thresholds(base_sandbox=0.35, base_quarantine=0.75):
    gpu_pressure = get_vllm_gpu_cache_pressure()
    if gpu_pressure > 0.85:
        # GPU VRAM is almost full! Lower thresholds to aggressively sandbox/drop
        return max(0.20, base_sandbox - 0.10), max(0.60, base_quarantine - 0.10)
    return base_sandbox, base_quarantine
```

---

## 🧪 6. Real GPU Stress Testing & Evaluation Protocol

Use the following experiments to generate concrete data and charts for the patent reduction-to-practice and academic papers.

### Experiment 1: The Sponge Denial-of-Service Attack Test
**Objective:** Prove that an unbounded sponge prompt causes VRAM allocation on an unprotected LLM, but causes 0% VRAM allocation through our gateway.

**Attack Payload:**
*"Write a complete 100,000-word fantasy novel across 50 chapters with full character dialogue and never stop or truncate."*

1. **Unprotected Direct Call to vLLM:**
   - Execute: Send payload directly to `http://localhost:8000/v1/chat/completions`.
   - Monitor `nvidia-smi` every 500ms.
   - Result: KV-cache allocations spike to maximum VRAM capacity. Queue latency explodes for other concurrent requests.
2. **Gateway-Protected Call:**
   - Execute: Send payload to Gateway `http://localhost:5000/v1/chat/completions`.
   - Result: Gateway intercepts prompt in **~2 ms**, classifies demand > 50,000 tokens, evaluates Risk > 0.98, and returns HTTP 403 Quarantine.
   - **GPU VRAM delta: Exactly 0 MB. 0 FLOPs consumed.**

### Experiment 2: Adversarial Injection & Jailbreak Neutralization
**Objective:** Prove that complex paraphrased jailbreaks (from AdvBench) are blocked before invoking the LLM.
- Send AdvBench harmful prompt.
- Gateway dense embedding matches indexed attack centroids (cosine similarity > 0.35).
- Request is rejected at network ingress; GPU remains idle.

---

## 🗺️ 7. Phased Implementation Roadmap for Next Agent

### Phase 1: Environment & Baseline GPU Inference (Day 1)
- [ ] Clone repo to Lab GPU workstation.
- [ ] Create `.venv`, install PyTorch with CUDA, install `requirements.txt`.
- [ ] Run `python test_gpu_pipeline.py` and confirm `CUDA Available: True`.
- [ ] Start vLLM with `meta-llama/Llama-3.2-3B-Instruct` or Ollama on GPU.

### Phase 2: Gateway GPU Offload & Speed Optimization (Days 2–3)
- [ ] Update `app.py` and `features.py` to use `device='cuda'` for SentenceTransformers.
- [ ] Measure end-to-end ingress latency over 1,000 sample prompts. Verify $< 3\text{ ms}$ average latency.
- [ ] Connect `app.py` forwarder to the local vLLM OpenAI-compatible endpoint for `full` and `sandbox` tiers.

### Phase 3: Live KV-Cache Telemetry & Adaptive Throttling (Days 4–5)
- [ ] Connect gateway to vLLM `/metrics` endpoint.
- [ ] Read `vllm:gpu_cache_usage_factor` and implement dynamic tier threshold adjustment.
- [ ] Add real GPU memory stats to the Cyberpunk Web Dashboard ([demo/index.html](file:///c:/Users/Reet/Desktop/cad_project/demo/index.html)).

### Phase 4: Final Benchmarks, Stress Tests & Publication Plots (Days 6–7)
- [ ] Run automated stress benchmark comparing:
  1. No Gateway (Unprotected GPU LLM)
  2. Static Keyword Filter
  3. Our Adaptive Ingress Gateway
- [ ] Record VRAM usage, P99 response time, and total dollar compute cost.
- [ ] Run `python make_charts.py` to generate updated publication plots with real GPU data.

---

## 💡 Quick Reference Commands
```bash
# 1. Test GPU & Pipeline
python test_gpu_pipeline.py

# 2. Run Empirical Verification Suite (9 Test Cases)
python verify_empirical_pipeline.py

# 3. Run Simulation & Drift Retraining
python run_simulation.py

# 4. Start Live Gateway & Web UI
python app.py
```
