import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure directories exist
for p in [DATA_DIR, PAPERS_DIR, INDEX_DIR, RESULTS_DIR, ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Load environment
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Paper definitions exactly per specification
PAPERS = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762v7",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "fallback_url": "https://arxiv.org/pdf/1706.03762.pdf",
        "filename": "1706.03762v7_attention_is_all_you_need.pdf"
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037v1",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "fallback_url": "https://arxiv.org/pdf/2410.00037.pdf",
        "filename": "2410.00037v1_moshi.pdf"
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955v2",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "fallback_url": "https://arxiv.org/pdf/2507.07955.pdf",
        "alt_url": "https://arxiv.org/pdf/2407.07955v2.pdf",
        "filename": "2507.07955v2_dynamic_chunking.pdf"
    }
}

DEFAULT_MODEL = "openai/gpt-oss-20b"
STANDIN_PORT = 8001
STANDIN_URL = f"http://127.0.0.1:{STANDIN_PORT}"
