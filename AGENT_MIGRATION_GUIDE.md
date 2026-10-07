# 🚀 Coding Agent Handover & Migration Guide
**Adaptive LLM Security & Resource Risk Gateway**  
*Document Purpose: Seamless handoff dossier for the next AI coding assistant (Cursor, Claude Code, Windsurf, Copilot, ChatGPT, Aider) and complete local-to-lab PC migration instructions.*

---

## 📌 Executive Summary & Context for Incoming Agent

Welcome! You are taking over development of the **Adaptive LLM Security & Resource Risk Gateway** (also referenced as the CAD Project / Invention Disclosure).

### What This System Is:
A high-throughput, low-latency (**< 5 ms**) network ingress proxy gateway positioned directly in front of LLM inference engines (vLLM, Ollama, TensorRT-LLM, or OpenAI-compatible backends).

### The Core Problem It Solves:
1. **Denial-of-Wallet (DoW) & Compute Wastage:** Modern LLMs handle safety alignments post-prompt. An attacker sending a 50,000-token sponge prompt or complex jailbreak forces the GPU cluster to allocate KV-cache pages and run billions of FLOPs just to output *"I cannot assist with this request."* The organization pays full GPU inference costs for attacker traffic.
2. **KV-Cache Starvation & OOM Crashes:** Massive prompt lengths or unbounded generation requests exhaust GPU VRAM pages, causing Out-Of-Memory (OOM) faults and latency spikes for concurrent users.
3. **Keyword Filter Fragility:** Simple regex and keyword blocklists fail against paraphrased, roleplay, or zero-day semantic jailbreaks (e.g., DAN, Developer Mode).

### How This System Neutralizes Threats:
Before any GPU memory or inference thread is touched, our gateway extracts lexical, semantic, and resource-demand features and evaluates a trained multi-class XGBoost model to classify requests into graduated execution tiers:
- **Full Tier ($\text{Risk} < 0.35$):** Benign queries routed to primary LLM with default context.
- **Sandbox Tier ($0.35 \le \text{Risk} < 0.75$):** Borderline queries capped at 256 output tokens and 5.0-second timeout.
- **Quarantine Tier ($\text{Risk} \ge 0.75$):** Immediate HTTP 403 block at the network layer (**$0.00 GPU cost**).

---

## 🏗️ Architectural Topology & Flow

```
                          [ Client / API / Web Application ]
                                          │
                                          ▼  (POST /v1/chat/completions)
                      ┌───────────────────────────────────────┐
                      │     Ingress Gateway (Flask / Proxy)   │
                      │               (< 5 ms)                │
                      └───────────────────┬───────────────────┘
                                          │
             ┌────────────────────────────┴────────────────────────────┐
             ▼                                                         ▼
     [ Lexical & Demand Engine ]                              [ Dense Semantic Engine ]
     • tiktoken BPE token count                               • sentence-transformers
     • Repetition ratio (bigram loop)                         • all-MiniLM-L6-v2 (384-d)
     • Predicted output demand heuristic                      • Cosine similarity against
     • TF-IDF n-gram jailbreak match                            adversarial centroids
     • Session repeat count                                   • Baseline floor filtering
             └────────────────────────────┬────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Multi-Class XGBoost Risk Engine    │
                      │  P(benign), P(injection), P(exhaust)  │
                      │          Risk = 1 - P(benign)         │
                      └───────────────────┬───────────────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
  [ Risk < 0.35 ]                [ 0.35 ≤ Risk < 0.75 ]             [ Risk ≥ 0.75 ]
    FULL TIER                        SANDBOX TIER                   QUARANTINE TIER
         │                                │                                │
         ▼                                ▼                                ▼
• Unconstrained Context          • Max 256 Output Tokens          • HTTP 403 Gateway Drop
• Standard LLM Generation        • 5.0s Timeout Isolation         • $0.00 GPU FLOPs
• Primary Serving Node           • Memory Runaway Protection      • Zero VRAM Allocated
         │                                │                                │
         └────────────────┬───────────────┘                                │
                          ▼                                                ▼
              ┌────────────────────────┐                        ┌────────────────────┐
              │ GPU Inference Backend  │                        │  Diagnostic JSON   │
              │  (vLLM / Ollama Node)  │                        │   Returned to User │
              └────────────────────────┘                        └────────────────────┘
```

---

## 📐 Mathematical Formulation & Feature Vectors

The system evaluates incoming requests using a 6-dimensional feature vector `FEATURE_COLUMNS`:

$$\mathbf{x} = \begin{bmatrix} f_{\text{tokens}}, & f_{\text{jb\_sim}}, & f_{\text{sem\_sim}}, & f_{\text{rep}}, & f_{\text{pred\_out}}, & f_{\text{session}} \end{bmatrix}^T$$

| Feature Name | Description | Source / Extraction Method |
| :--- | :--- | :--- |
| `prompt_length_tokens` | Prompt token count | `tiktoken.get_encoding("cl100k_base")` |
| `jailbreak_similarity` | Lexical n-gram jailbreak match | Scikit-learn TF-IDF vectorizer against known jailbreak signatures |
| `semantic_similarity` | Dense semantic cosine similarity | `all-MiniLM-L6-v2` dense vector dot product against pre-indexed attack centroids |
| `repetition_ratio` | Bigram repetition loop frequency | $\frac{\sum (count(b_i) - 1)}{\max(N, 1)}$ over all bigrams |
| `predicted_output_tokens` | Resource demand heuristic | Base tokens + regex demand multipliers (e.g. "write 100,000 words") |
| `session_repeat_count` | Rapid-burst repetition count | Session tracker for low-and-slow denial-of-service detection |

### Risk Score:
$$\text{Risk} = 1.0 - P(\text{benign}) = P(\text{injection}) + P(\text{exhaustion})$$

### Cost Savings Model:
$$\text{Cost} = (N_{\text{prompt}} \cdot P_{\text{in}} \cdot (1 - 0.8 \cdot P_{\text{hit}})) + (N_{\text{completion}} \cdot P_{\text{out}})$$
- For Quarantine: $\text{Cost} = \$0.00$.
- For Sandbox: $N_{\text{completion}} \le 256$.

---

## 🗂️ Project Directory Structure & Codebase Map

```
cad_project/
├── AGENT_MIGRATION_GUIDE.md         # [THIS FILE] Master handover for next agent
├── GPU_IMPLEMENTATION_ROADMAP.md    # Guide for deploying to Lab GPU & vLLM/Ollama
├── PROJECT_CONTEXT_FULL.md          # Comprehensive patent/system technical context
├── LITERATURE_REVIEW.md             # Survey of existing defense literature & gaps
├── PROFESSOR_PRESENTATION.md        # Academic presentation talking points & slides
├── RESULTS_SUMMARY.md               # Empirical evaluation summary & benchmark data
├── README.md                        # Project landing readme & quickstart
├── Invention_Disclosure_Updated.docx# Official patent invention disclosure document
├── cloud_basepaper.pdf              # Reference baseline paper
│
├── app.py                           # Live Flask API gateway with SSE streaming
├── router.py                        # Tier assignment logic & cost estimation
├── features.py                      # Feature extractor & cache-locality estimator
├── classifier.py                    # XGBoost multi-class wrapper & feedback retrain
├── baselines.py                     # Baseline policies (no protection, regex filter)
├── data_gen.py                      # Synthetic traffic generator for simulations
├── run_simulation.py                # Reduction-to-practice simulation (Sections 10)
├── train_on_datasets.py             # Empirical trainer on AdvBench + Alpaca + DoS
├── verify_empirical_pipeline.py     # 9-case benchmark validation suite
├── verify_pipeline.py               # Short test runner entry point
├── make_charts.py                   # Matplotlib chart generator for publications
├── test_gpu_pipeline.py             # Turnkey GPU diagnostic & benchmark script
├── start_demo.bat                   # 1-click Windows launcher for interactive demo
├── requirements.txt                 # Canonical dependencies (CPU & CUDA GPU)
├── requirements_demo.txt            # Minimal demo dependencies
├── .gitignore                       # Git exclusion rules
│
├── data/                            # Training & benchmark datasets
│   ├── advbench_harmful_behaviors.csv # AdvBench attack prompts (520+ harmful goals)
│   └── alpaca_sample.json          # Stanford Alpaca instruction dataset (~22MB)
│
├── demo/                            # Interactive Web Dashboard
│   └── index.html                  # Cyberpunk dark-mode live UI with SSE meters
│
├── models/                          # Serialized trained models & embeddings
│   ├── classifier.joblib           # Trained XGBoost model (98.41% accuracy)
│   ├── tfidf.joblib                # Trained TF-IDF vectorizer
│   ├── ref_embeddings.npy          # 384-d normalized attack centroids
│   ├── reference_jailbreaks.json   # Raw text of reference jailbreak patterns
│   └── model_metadata.json         # Training metrics, parameters, & schema
│
└── results/                         # Generated experimental results & charts
    ├── classifier_report_initial.json
    ├── classifier_report_after_retrain.json
    ├── comparison_charts.png       # 4-panel publication-ready comparison graph
    ├── summary_metrics.csv         # Simulation metrics table across Batches A/B/C
    ├── per_request_batchA.csv
    ├── per_request_batchB_pre_retrain.csv
    └── per_request_batchC_post_retrain.csv
```

---

## 📊 Current Verification & Benchmark Status

### 1. Empirical Training (`train_on_datasets.py`)
- **Dataset Size:** 4,264 real samples (994 AdvBench/Jailbreak, 2,700 Alpaca/LMSYS Benign, 570 DoS/Sponge).
- **Model:** XGBoost Multi-Class (`multi:softprob`).
- **Test Accuracy:** **98.41%**.

### 2. Empirical Verification (`verify_empirical_pipeline.py`)
Run: `python verify_empirical_pipeline.py`  
Status: **9/9 PASS (100%)**
- Benign Code Algorithm -> `FULL` tier (Risk 0.110)
- Benign ML Explanation -> `FULL` tier (Risk 0.168)
- Benign Transformer Theory -> `FULL` tier (Risk 0.142)
- AdvBench Direct Attack -> `QUARANTINE` tier (Risk 0.984)
- Paraphrased DAN Jailbreak -> `QUARANTINE` tier (Risk 0.963)
- System Prompt Override -> `QUARANTINE` tier (Risk 0.997)
- 100k Novel Sponge Demand -> `QUARANTINE` tier (Risk 0.985)
- Repetition Flood DoS -> `QUARANTINE` tier (Risk 0.998)
- Low-and-Slow Burst DoS -> `QUARANTINE` tier (Risk 0.999)

### 3. Simulation & Concept Drift (`run_simulation.py`)
Run: `python run_simulation.py`  
Status: **PASS (Exit Code 0)**
- Cost savings vs Unprotected baseline: **56.3%**.
- Adaptive feedback retrain on Batch B drift: Recovers detection from 51.5% to **97.5%**.

---

## 💻 Step-by-Step: Migrating Files to the Lab PC

Follow these exact steps to migrate the codebase from the local laptop to the lab GPU workstation.

### Step 1: Clean Local Directory (Already Done!)
- Redundant root artifacts (`comparison_charts.png`, `summary_metrics.csv`) have been removed from root and are clean in `results/`.
- `data_gen.py` has been updated with `semantic_similarity` to align with `FEATURE_COLUMNS`.
- `.gitignore` is prepared to exclude virtual environments, cache, and logs.

### Step 2: Choose Your Transfer Method

#### Method A: Git Repository (Recommended)
From the local terminal in `c:\Users\Reet\Desktop\cad_project`:
```powershell
# 1. Initialize git and commit files
git init
git add .
git commit -m "feat: complete adaptive llm gateway prototype with empirical models"

# 2. Push to your GitHub/GitLab private repo
git remote add origin https://github.com/YOUR_USERNAME/cad_llm_gateway.git
git branch -M main
git push -u origin main
```
On the Lab GPU workstation (Linux or Windows):
```bash
git clone https://github.com/YOUR_USERNAME/cad_llm_gateway.git
cd cad_llm_gateway
```

#### Method B: Direct Network Transfer (SCP / SFTP / Rsync)
If the Lab PC is accessible via SSH:
```powershell
# From local PowerShell (zip excluding .venv)
tar --exclude='.venv' --exclude='__pycache__' -czvf cad_project.tar.gz .

# Transfer via scp
scp cad_project.tar.gz lab_user@lab_ip_or_hostname:~/cad_project.tar.gz
```
On the Lab PC:
```bash
mkdir -p ~/cad_project && cd ~/cad_project
tar -xzvf ~/cad_project.tar.gz
```

#### Method C: USB Flash Drive or Google Drive / OneDrive
- Compress the `cad_project` folder **WITHOUT** the `.venv` folder.
- Copy `cad_project.zip` to your flash drive or cloud drive.
- Extract into the lab PC workspace.

---

## ⚡ Lab GPU Setup Instructions for Incoming Agent

Once the repository is on the Lab GPU PC:

### 1. Create a Fresh Virtual Environment
```bash
# On Linux Lab PC:
python3 -m venv .venv
source .venv/bin/activate

# On Windows Lab PC:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA Acceleration
```bash
# For CUDA 12.1/12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Repository Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Hardware Diagnostic Script
```bash
python test_gpu_pipeline.py
```
This script will verify:
- CUDA availability and VRAM size.
- MiniLM embedding acceleration on GPU.
- Artifact loading integrity.
- Ollama or vLLM server connectivity.

---

## 🎯 Next Tasks & Priorities for the Incoming Agent

Here is the exact task roadmap for you (the incoming agent):

1. **Verify GPU Acceleration on Gateway Ingress (`features.py` / `app.py`)**:
   - Update `features.py` or `verify_empirical_pipeline.py` to initialize `SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")`.
   - Benchmark ingress latency on the lab GPU to achieve $< 2.0\text{ ms}$ embedding latency.

2. **Connect Gateway to Real GPU Inference Engine (vLLM)**:
   - Launch vLLM on the lab GPU:
     ```bash
     vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000 --gpu-memory-utilization 0.85
     ```
   - Connect `app.py` or a dedicated forwarder to route full/sandbox queries to `http://localhost:8000/v1/chat/completions`.

3. **Integrate Real PagedAttention KV-Cache Metrics**:
   - Query vLLM's live telemetry endpoint `http://localhost:8000/metrics`.
   - Read `vllm:gpu_cache_usage_factor` and feed real GPU memory pressure into `router.py`.
   - When GPU VRAM cache usage exceeds 85%, dynamically tighten the Quarantine/Sandbox thresholds to protect the GPU from starvation.

4. **Run Real VRAM Exhaustion Stress Test**:
   - Submit a 100,000-token sponge attack directly to vLLM (observe memory spike/OOM).
   - Submit the same sponge attack through our gateway (observe immediate 403 block with 0% VRAM spike).
   - Capture `nvidia-smi` logs and generate comparison plots for the paper/patent reduction-to-practice.

---

## 📜 Key Contact Points & Reference Files

- Patent Text & Claims: [PROJECT_CONTEXT_FULL.md](file:///c:/Users/Reet/Desktop/cad_project/PROJECT_CONTEXT_FULL.md) & [Invention_Disclosure_Updated.docx](file:///c:/Users/Reet/Desktop/cad_project/Invention_Disclosure_Updated.docx)
- Academic Paper Literature: [LITERATURE_REVIEW.md](file:///c:/Users/Reet/Desktop/cad_project/LITERATURE_REVIEW.md) & [cloud_basepaper.pdf](file:///c:/Users/Reet/Desktop/cad_project/cloud_basepaper.pdf)
- Experimental Results: [RESULTS_SUMMARY.md](file:///c:/Users/Reet/Desktop/cad_project/RESULTS_SUMMARY.md) & [summary_metrics.csv](file:///c:/Users/Reet/Desktop/cad_project/results/summary_metrics.csv)
- GPU Roadmap & Setup: [GPU_IMPLEMENTATION_ROADMAP.md](file:///c:/Users/Reet/Desktop/cad_project/GPU_IMPLEMENTATION_ROADMAP.md)
