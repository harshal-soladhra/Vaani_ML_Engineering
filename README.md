# The Relay — Vaani ML Engineering Assignment

A production-grade, highly resilient LLM pipeline that answers engineering questions over a focused research library of three arXiv papers:
1. **Attention Is All You Need** (arXiv:1706.03762v7)
2. **Moshi: a speech-text foundation model for real-time dialogue** (arXiv:2410.00037v1)
3. **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** (arXiv:2507.07955v2)

The system integrates real Groq free-tier model calls (`openai/gpt-oss-20b`) with continuous rate-limit tracking, alongside a calibrated local FastAPI stand-in simulator replicating latency distribution, load degradation, streaming SSE chunks, and 429 quota exhaustion.

---

## Project Structure

```
ML_Assignment_Harshal/
│
├── .env.example              # Environment template with GROQ_API_KEY placeholder
├── .gitignore                # Protects secrets, cache, and virtual environments
├── requirements.txt          # Production dependencies
├── results.csv               # Complete empirical measurements across all 5 scenarios
├── assignment_report.md      # Full 2-page engineering report matching exact brief headings
├── assignment_report.pdf     # Generated PDF version of the final report
├── run_all_tests.py          # Standalone verification runner for all 5 test suites
├── generate_pdf.py           # Script to build assignment_report.pdf
│
├── data/
│   ├── papers/               # Verified downloaded arXiv PDFs
│   │   ├── 1706.03762v7_attention_is_all_you_need.pdf
│   │   ├── 2410.00037v1_moshi.pdf
│   │   └── 2507.07955v2_dynamic_chunking.pdf
│   └── index/                # Multi-granularity chunk indices
│       ├── chunks_300.json   # 300-token chunks
│       ├── chunks_600.json   # 600-token chunks
│       └── chunks_800.json   # 800-token chunks
│
├── src/
│   ├── config.py             # System configuration, paths, model settings
│   ├── rag/
│   │   ├── downloader.py     # arXiv downloader with retry & header emulation
│   │   ├── indexer.py        # PDF text extractor preserving titles, sections & pages
│   │   └── retriever.py      # TF-IDF sublinear vectorizer with 2-stage lexical reranker
│   ├── llm/
│   │   └── groq_client.py    # Real Groq client with SSE streaming & rate limit tracking
│   ├── simulator/
│   │   └── server.py         # Calibrated FastAPI stand-in simulator
│   └── pipeline/
│       ├── queue.py          # Bounded async queue, concurrency semaphores, leaky bucket
│       └── relay.py          # 4-step pipeline: Query routing, Tool call, Branch, Citation loop
│
├── tests/
│   ├── test_retrieval.py     # Tests chunk sizes, metadata preservation, and reranking
│   ├── test_citations.py     # Tests strict citation syntax & hallucination rejection
│   ├── test_rate_limits.py   # Tests RPM, TPM, deadline timeouts & daily exhaustion
│   ├── test_simulator.py     # Tests FastAPI stand-in endpoints and rate-limit headers
│   └── test_pipeline.py      # Tests query understanding and out-of-scope safe refusal
│
└── benchmarks/
    ├── run_experiments.py   # Benchmark runner executing 15 questions across 5 scenarios
    ├── compute_tables.py    # Reproducible calculation of Table 1, 2a, 2b, and 3
    └── plots.py             # Matplotlib visual generation script
```

---

## Quickstart & Installation

### 1. Set up Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 3. Configure API Key
Create `.env` using `.env.example`:
```ini
GROQ_API_KEY=gsk_your_key_here
```
*(Your API key is automatically loaded and never exposed in logs or commits).*

---

## Execution Guide

### Download and Index Papers
To re-fetch papers and rebuild the chunk indices:
```powershell
.\.venv\Scripts\python src/rag/downloader.py
.\.venv\Scripts\python src/rag/indexer.py
```

### Run Automated Tests
Execute the standalone test suite verifying retrieval, citations, rate limits, simulator, and pipeline routing:
```powershell
.\.venv\Scripts\python run_all_tests.py
```
*(Or run via `pytest`):*
```powershell
.\.venv\Scripts\python -m pytest -v tests
```

### Run Benchmark Experiments & Generate `results.csv`
```powershell
.\.venv\Scripts\python benchmarks/run_experiments.py
```

### Compute Results Tables
Recomputes all percentiles and figures in Table 1, 2a, 2b, and 3 directly from `results.csv`:
```powershell
.\.venv\Scripts\python benchmarks/compute_tables.py
```

### Generate Plots & PDF Report
```powershell
.\.venv\Scripts\python benchmarks/plots.py
.\.venv\Scripts\python generate_pdf.py
```

---

## Key Engineering Decisions

1. **Pre-Execution Evidence Gating**: Refuses out-of-scope questions immediately ($\tau = 0.05$) to conserve rate limit quota and eliminate hallucinations.
2. **Empirical Leaky-Bucket Calibration**: Probed live Groq response headers revealing an $86.4\text{ s}$ per-request reset cycle, calibrating our stand-in with Gaussian TTFT ($\mu = 1200\text{ ms}, \sigma = 150\text{ ms}$).
3. **Bounded Bidirational Citations**: Every answer requires `[Paper Title, Section, Page]`. If citations are missing, a bounded single-loop correction runs; if unavailable, the pipeline refuses rather than guessing.
4. **Resilient Rate Limiter**: Implements sliding-window token buckets and non-blocking 429 daily refusals to prevent cascading retry storms.
