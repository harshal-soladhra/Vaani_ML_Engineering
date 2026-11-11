import asyncio
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.retriever import retrieve_passages, get_retriever
from src.pipeline.relay import RelayPipeline
from src.pipeline.queue import RateLimitBucket, ReliabilityQueue
from fastapi.testclient import TestClient
from src.simulator.server import app

def test_retrieval():
    print("[1/5] Testing Retrieval & Chunks...")
    r300 = get_retriever(300)
    assert len(r300.chunks) > 0, "No chunks loaded"
    chunk = r300.chunks[0]
    assert "paper_title" in chunk and "page" in chunk and "section" in chunk, "Missing metadata"
    
    # Setting 1
    c1, s1 = retrieve_passages("scaled dot-product attention", "setting_1")
    assert len(c1) <= 3 and s1 > 0.0, "Setting 1 failed"
    # Setting 2
    c2, s2 = retrieve_passages("Moshi speech language model", "setting_2")
    assert len(c2) <= 5 and s2 > 0.0, "Setting 2 failed"
    # Setting 3
    c3, s3 = retrieve_passages("dynamic chunking sequence modeling", "setting_3")
    assert len(c3) <= 5 and s3 > 0.0, "Setting 3 failed"
    print("  --> Retrieval tests PASSED.")

def test_citations():
    print("[2/5] Testing Citation Validation...")
    pipeline = RelayPipeline(source="stand-in")
    dummy_chunks = [
        {"paper_title": "Attention Is All You Need", "section": "3.2 Attention", "page": 4},
        {"paper_title": "Moshi: a speech-text foundation model", "section": "2 Architecture", "page": 3}
    ]
    valid_text = (
        "Multi-head attention uses scaled dot-product operations [Attention Is All You Need, 3.2 Attention, p. 4]. "
        "Audio codec streams are processed simultaneously [Moshi: a speech-text foundation model, 2 Architecture, p. 3]."
    )
    valid, cits = pipeline.validate_citations(valid_text, dummy_chunks)
    assert valid is True and len(cits) >= 2, "Valid citation check failed"

    uncited_text = "Transformers are neural network models that rely on attention without recurrence."
    valid_bad, _ = pipeline.validate_citations(uncited_text, dummy_chunks)
    assert valid_bad is False, "Uncited text incorrectly passed"
    print("  --> Citation validation tests PASSED.")

async def test_rate_limits():
    print("[3/5] Testing Rate Limiting & Bucket Throttling...")
    bucket = RateLimitBucket(rpm=3, tpm=1000)
    import time
    d = time.time() + 2.0
    assert await bucket.acquire(10, d) is True
    assert await bucket.acquire(10, d) is True
    assert await bucket.acquire(10, d) is True
    # 4th with short deadline should be throttled
    short_d = time.time() + 0.1
    assert await bucket.acquire(10, short_d) is False

    # Daily exhaustion
    rq = ReliabilityQueue(max_queue_size=10, concurrency_limit=2)
    rq.mark_daily_exhausted()
    async def dummy(): return {"status": "ok"}
    res = await rq.submit(dummy)
    assert res["status"] == "rate_limited"
    assert "Daily request budget is exhausted" in res["text"]
    print("  --> Rate limit tests PASSED.")

def test_simulator():
    print("[4/5] Testing Stand-in Simulator...")
    tc = TestClient(app)
    h = tc.get("/health").json()
    assert h["status"] == "healthy"
    
    # Configure daily budget exhausted
    tc.post("/configure", json={"daily_budget_exhausted": True})
    r = tc.post("/openai/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 429
    assert r.headers.get("x-ratelimit-remaining-requests") == "0"
    tc.post("/reset")
    print("  --> Simulator tests PASSED.")

async def test_pipeline():
    print("[5/5] Testing Pipeline Routing & Out-of-Scope Refusal...")
    pipeline = RelayPipeline(source="stand-in")
    res = await pipeline.run("req_test_refusal", "What is the training dataset for Llama 3?")
    assert res["status"] == "ok"
    assert res["answer_grade"] == "refused_correctly"
    assert "does not contain sufficient information" in res["answer"]
    print("  --> Pipeline routing & refusal tests PASSED.")

async def main():
    print("=== Running Relay Verification Test Suite ===")
    test_retrieval()
    test_citations()
    await test_rate_limits()
    test_simulator()
    await test_pipeline()
    print("=== ALL 5 TEST MODULES PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(main())
