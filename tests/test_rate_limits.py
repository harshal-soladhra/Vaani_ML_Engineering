import asyncio
import time
import pytest
from src.pipeline.queue import RateLimitBucket, ReliabilityQueue

@pytest.mark.asyncio
async def test_rpm_bucket_throttling():
    bucket = RateLimitBucket(rpm=3, tpm=1000)
    deadline = time.time() + 2.0
    
    # First 3 should succeed immediately
    assert await bucket.acquire(10, deadline) is True
    assert await bucket.acquire(10, deadline) is True
    assert await bucket.acquire(10, deadline) is True
    
    # 4th should exceed RPM and fail if deadline is short
    short_deadline = time.time() + 0.2
    assert await bucket.acquire(10, short_deadline) is False

@pytest.mark.asyncio
async def test_tpm_bucket_throttling():
    bucket = RateLimitBucket(rpm=10, tpm=100)
    deadline = time.time() + 0.2
    assert await bucket.acquire(80, deadline) is True
    # Next 30 tokens exceed 100 TPM limit
    assert await bucket.acquire(30, deadline) is False

@pytest.mark.asyncio
async def test_queue_daily_exhaustion():
    rq = ReliabilityQueue(max_queue_size=10, concurrency_limit=2)
    rq.mark_daily_exhausted()

    async def dummy_task():
        return {"status": "ok"}

    res = await rq.submit(dummy_task)
    assert res["status"] == "rate_limited"
    assert "Daily request budget is exhausted" in res["text"]
    assert res["retries"] == 0
