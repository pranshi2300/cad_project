# 🛡️ Adaptive LLM Security & Resource Risk Gateway: Complete Project Context (A to Z)

---

## 📌 1. Project Overview & Executive Summary

**Project Title:** Adaptive LLM Security & Resource Risk Gateway  
**Domain:** AI Systems Engineering, Large Language Model (LLM) Infrastructure, Ingress Security, and Cloud Cost Optimization  
**Core Purpose:** An intelligent, sub-5-millisecond edge proxy gateway placed in front of LLM inference clusters (e.g., vLLM, Ollama, TensorRT-LLM, OpenAI-compatible backends). It intercepts incoming prompts at the network proxy layer, extracts lexical and dense semantic features, classifies multi-vector risk via a trained machine learning model, and dynamically routes requests into graduated execution tiers—completely neutralizing malicious prompt injections and compute-sponge Denial-of-Service (DoS) attacks at **$0.00 GPU compute cost**.

---

## 🚨 2. The Core Problem It Solves

### The Flaw in Current Frontier LLM Defenses
Modern frontier LLMs (e.g., ChatGPT, Claude, Llama 3) handle guardrails primarily via post-training alignment (RLHF, DPO) or heavy auxiliary guardrail models (e.g., Llama Guard, NeMo Guardrails). 
When an attacker sends an adversarial prompt (such as a 100,000-word sponge prompt, a DAN jailbreak, or a system extraction payload):
1. **Denial-of-Wallet (DoW) & Compute Wastage:** The GPU cluster allocates memory and executes billions of floating-point operations (FLOPs) processing the prompt, only to produce a polite refusal string like: *"I cannot fulfill this request..."*. The cloud operator incurs substantial GPU inference bills for processing attacker payloads.
2. **KV-Cache Exhaustion & OOM Starvation:** Massive prompt lengths or unbounded recursive output demands exhaust the Key-Value (KV) cache pages in GPU VRAM, leading to Out-Of-Memory (OOM) crashes, dropped connections, and degraded Quality-of-Service (QoS) for legitimate users on multi-tenant inference nodes.
3. **Keyword Filter Brittleness:** Traditional static regex / blocklist firewalls fail against paraphrased, multilingual, or zero-day semantic jailbreaks (e.g., roleplay "Developer Mode" or "DAN").

### The Solution: Upfront Ingress Interception
This gateway operates at the CPU/edge proxy level (outside the GPU cluster). In **$<5\text{ms}$**, it inspects the prompt, identifies threats before GPU memory is allocated, and quarantines dangerous traffic with zero GPU overhead.

---

## 🏗️ 3. System Architecture & End-to-End Workflow

```
                        [ Client / Web Application / API ]
                                        │
                                        ▼ (HTTP POST /v1/chat/completions)
                     ┌──────────────────────────────────────┐
                     │    INGRESS PROXY GATEWAY (< 5ms)     │
                     └──────────────────┬───────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
    [ Lexical & Demand Features ]                         [ Semantic Intent ]
    • tiktoken BPE token count                            • 384-d MiniLM Embeddings
    • Repetition density (n-gram loops)                   • Dense Cosine Similarity
    • Output length demand heuristic                      • Baseline Floor Filter
    • Lexical TF-IDF n-grams                              • KV-Cache Locality
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │    XGBOOST MULTI-CLASS CLASSIFIER    │
                     │       (98.41% Empirical Accuracy)    │
                     │   Predicts: Benign / Injection / DoS │
                     │   Calculates Graduated Risk Score    │
                     └──────────────────┬───────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  [ Risk < 0.35 ]              [ 0.35 ≤ Risk < 0.75 ]           [ Risk ≥ 0.75 ]
    FULL TIER                      SANDBOX TIER                 QUARANTINE TIER
         │                              │                              │
         ▼                              ▼                              ▼
• Unconstrained Context        • Hard 256-Token Output Cap     • Gateway Drop (403)
• Full Generation Quota        • 5.0-Second Isolated Timeout   • $0.00 GPU Cost
• Direct to Primary Model      • Prevents Memory Runaway       • Diagnostic JSON
         │                              │                              │
         └───────────────┬──────────────┘                              │
                         ▼                                             ▼
               ┌───────────────────┐                         ┌───────────────────┐
               │  Inference Node   │                         │ Immediate Block   │
               │ (Ollama llama3.2) │                         │ No GPU Allocation │
               └───────────────────┘                         └───────────────────┘
```

---

## 🛡️ 4. Tiered Execution & Graduated Defense Logic

Instead of a brittle binary allow/block filter (which disrupts user experience on borderline prompts), the system applies a graduated tier policy:

| Tier | Risk Threshold | Execution Constraints | Intended Behavior |
| :--- | :---: | :--- | :--- |
| **Full Tier** | $\text{Risk} < 0.35$ | Default model parameters, full token limit | Clean academic, coding, conversational prompts proceed without latency or restriction. |
| **Sandbox Tier** | $0.35 \le \text{Risk} < 0.75$ | Max 256 tokens, 5.0-second timeout, memory isolation | Ambiguous or borderline queries are safely evaluated without risking cluster runaway or memory exhaustion. |
| **Quarantine Tier** | $\text{Risk} \ge 0.75$ | Immediate HTTP 403 response, zero LLM invocation | Verified attacks (jailbreaks, prompt injections, sponge floods) are rejected instantly at the network layer. |

---

## 📊 5. Empirical Datasets & Machine Learning Training Pipeline

The risk engine uses an XGBoost gradient-boosted multi-class classifier trained on **4,264 real-world empirical samples** (`train_on_datasets.py`):

1. **Adversarial Injections (994 samples):**
   - Ingests the **AdvBench benchmark** (Zou et al., UC Berkeley / CMU) containing 520+ harmful attack objectives.
   - Expanded with real-world jailbreaks (DAN v11, Developer Mode bypasses, System Prompt extraction vectors, roleplay evasion).
2. **Benign Conversational & Technical Requests (2,700 samples):**
   - Curated from **Stanford Alpaca** and **LMSYS Chatbot Arena**.
   - Spans diverse domains: Python/C++ programming, mathematical explanations, essay drafting, creative writing, and factual Q&A.
3. **Resource Exhaustion & Denial-of-Service (570 samples):**
   - Unbounded output demands (e.g., *"write a 100,000-word novel"*).
   - High-entropy repetition loops, bigram spam floods, low-and-slow session bursts designed to saturate KV-cache pages.

### Feature Representation Matrix
- **384-dimensional dense semantic embeddings:** Generated via `sentence-transformers/all-MiniLM-L6-v2`, measuring cosine similarity against indexed adversarial vectors with an English natural-language baseline floor.
- **Lexical TF-IDF Vectorizer:** Word and character n-grams capturing structural injection markers.
- **Structural Heuristics:** Exact BPE token counts (`tiktoken`), n-gram repetition density, estimated output token demands, and session request burst frequencies.
- **Model Validation Metric:** **98.41% cross-validated classification accuracy**.

---

## 🧪 6. Empirical Benchmark Verification Suite

The repository contains an automated verification script (`verify_empirical_pipeline.py`) validating the pipeline against 9 distinct threat archetypes:

| Benchmark Test Scenario | Threat Category | MiniLM Sim | TF-IDF Sim | Calculated Risk | Assigned Tier | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Flatten List Algorithm** | Benign Python Code | 0.000 | 0.000 | 0.110 | **FULL** | ✅ PASS |
| **Supervised vs Unsupervised ML** | Benign Academic Query | 0.000 | 0.000 | 0.168 | **FULL** | ✅ PASS |
| **Transformer Theory** | Benign Technical Explanation | 0.000 | 0.000 | 0.142 | **FULL** | ✅ PASS |
| **AdvBench Direct Attack** | Malicious Behavioral Harm | 0.234 | 0.000 | 0.984 | **QUARANTINE** | ✅ PASS |
| **Paraphrased DAN v11** | Semantic Roleplay Jailbreak | 0.392 | 0.120 | 0.963 | **QUARANTINE** | ✅ PASS |
| **System Prompt Override** | Instruction Leak / Extraction | 0.885 | 0.178 | 0.997 | **QUARANTINE** | ✅ PASS |
| **100k-Word Novel Demand** | Resource Sponge Volume Attack | 0.000 | 0.000 | 0.985 | **QUARANTINE** | ✅ PASS |
| **Repetition Token Flood** | Bigram Token Loop DoS | 0.000 | 0.000 | 0.998 | **QUARANTINE** | ✅ PASS |
| **Low-and-Slow Session Burst** | Frequency Anomaly (`sess=10`) | 0.000 | 0.000 | 0.999 | **QUARANTINE** | ✅ PASS |

**Benchmark Result:** **9 / 9 (100.0%) Tests Passed**.

---

## 📁 7. Repository Structure & File Directory

```
cad_project/
├── app.py                         # Production Flask gateway server with OpenAI /v1 API & web UI routes
├── classifier.py                  # RiskClassifier wrapper around trained XGBoost & feature pipeline
├── features.py                    # Feature extraction (tiktoken BPE, repetition density, cache locality)
├── router.py                      # Tier assignment engine (Full, Sandbox, Quarantine) and cost modeling
├── train_on_datasets.py           # Ingestion, feature building & XGBoost training pipeline
├── verify_empirical_pipeline.py   # 9-vector benchmark verification test suite
├── baselines.py                   # Baseline comparisons against static regex & keyword filters
├── data_gen.py                    # Synthetic adversarial & benign data augmentation generators
├── make_charts.py                 # Generates comparative performance and cost reduction visualizations
├── comparison_charts.png          # High-resolution benchmark comparison charts
├── requirements_demo.txt          # Python library dependencies
├── start_demo.bat                 # One-click Windows startup script
│
├── data/
│   └── harmful_behaviors.csv      # Empirical AdvBench benchmark dataset (520+ harmful attack vectors)
│
├── demo/
│   └── index.html                 # Modern glassmorphic dashboard with live SVG gauges & telemetry
│
├── models/
│   ├── classifier.joblib          # Serialized XGBoost multi-class decision tree ensemble
│   ├── tfidf.joblib               # Serialized lexical TF-IDF vectorizer
│   ├── ref_embeddings.npy         # Precomputed 384-d MiniLM embeddings of reference jailbreaks
│   ├── reference_jailbreaks.json  # Ingested corpus of AdvBench + DAN + System extraction prompts
│   └── model_metadata.json        # Empirical training metadata (sample counts, 98.41% accuracy)
│
├── LITERATURE_REVIEW.md           # Survey of academic papers, prior art, and attack taxonomies
├── RESULTS_SUMMARY.md             # Reduction to practice technical report with metrics
├── PROFESSOR_PRESENTATION.md      # Detailed academic defense and presentation walkthrough guide
└── PROJECT_CONTEXT_FULL.md        # Comprehensive master context document (this file)
```

---

## 🔌 8. API & Integration Compatibility

The gateway natively implements the standard OpenAI chat completions endpoint:

### Endpoint: `POST /v1/chat/completions`
- **Request Format:** Standard OpenAI JSON payload (`model`, `messages`, `max_tokens`, `temperature`).
- **Plug-and-Play Compatibility:** Compatible with official OpenAI Python/Node SDKs, LangChain, and LlamaIndex simply by configuring `base_url="http://localhost:5000/v1"`.
- **Backend Model Serving:** Routes approved queries to local or remote LLM backends (configured with **Ollama `llama3.2:3b`**).
- **Interactive UI:** A real-time web dashboard accessible at `http://localhost:5000` with live SVG risk telemetry, dual-similarity meters, payload inspection, and copyable JSON diagnostics.

---

## 💻 9. Software & Hardware Requirements

### Software Required:
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Runtime Environment:** Python 3.10 or higher
- **Core Python Libraries:**
  - `flask`, `flask-cors` (REST API & gateway routing)
  - `xgboost` (Multi-class risk classification model)
  - `sentence-transformers` (`all-MiniLM-L6-v2` dense embedding generation)
  - `scikit-learn` (TF-IDF vectorizer & evaluation metrics)
  - `tiktoken` (Fast BPE token calculation)
  - `joblib`, `numpy`, `pandas`, `requests`
- **Local LLM Engine (Optional for live generation):** Ollama running `llama3.2:3b` (or any OpenAI-compatible API endpoint)
- **Web Browser:** Any modern browser (Chrome, Edge, Firefox) for the dashboard

### Hardware Required:
- **Edge Proxy:** Runs efficiently on standard multi-core CPU (no dedicated GPU required for the gateway itself).
- **RAM:** Minimum 4 GB (8 GB recommended for caching sentence-transformer embeddings).
- **Disk:** ~1 GB for model weights and virtual environment.

---

## 🚀 10. How to Run and Test

1. **Activate Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # On Linux/Mac: source .venv/bin/activate
   pip install -r requirements_demo.txt
   ```

2. **Run Verification Test Suite:**
   ```bash
   python verify_empirical_pipeline.py
   ```
   *(Validates all 9 attack & benign vectors; outputs 100% pass rate)*

3. **Launch the Gateway & Web Interface:**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your browser to test interactive prompt inspection, live risk gauges, and dynamic routing.
