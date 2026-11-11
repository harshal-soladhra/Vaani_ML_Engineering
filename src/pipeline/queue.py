import asyncio
import time
from typing import Dict, Any, Optional, Callable, Awaitable

class RateLimitBucket:
    def __init__(self, rpm: int = 30, tpm: int = 8000):
        self.rpm = rpm
        self.tpm = tpm
        self.request_times = []
        self.token_times = []
        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int, deadline: float) -> bool:
        """Wait until RPM and TPM capacity is available, or until deadline expires."""
        while True:
            now = time.time()
            if now > deadline:
                return False

            async with self.lock:
                # Prune older than 60s
                self.request_times = [t for t in self.request_times if now - t < 60.0]
                self.token_times = [(t, tok) for t, tok in self.token_times if now - t < 60.0]

                current_tokens = sum(tok for _, tok in self.token_times)
                rpm_ok = len(self.request_times) < self.rpm
                tpm_ok = (current_tokens + estimated_tokens) <= self.tpm

                if rpm_ok and tpm_ok:
                    self.request_times.append(now)
                    self.token_times.append((now, estimated_tokens))
                    return True

                # Compute wait time until oldest falls off
                wait_time = 0.5
                if not rpm_ok and self.request_times:
                    wait_time = max(0.1, 60.0 - (now - self.request_times[0]))
                elif not tpm_ok and self.token_times:
                    wait_time = max(0.1, 60.0 - (now - self.token_times[0][0]))

            # Check deadline before sleeping
            if now + wait_time > deadline:
                return False
            await asyncio.sleep(min(wait_time, 1.0))

class ReliabilityQueue:
    def __init__(
        self,
        max_queue_size: int = 60,
        concurrency_limit: int = 4,
        rpm_limit: int = 30,
        tpm_limit: int = 8000,
        request_timeout: float = 12.0,
        max_retries: int = 2
    ):
        self.max_queue_size = max_queue_size
        self.concurrency_limit = concurrency_limit
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.rate_limiter = RateLimitBucket(rpm=rpm_limit, tpm=tpm_limit)
        self.peak_queue_length = 0
        self.is_daily_exhausted = False

    def update_limits(self, rpm: int, tpm: int):
        self.rate_limiter.rpm = rpm
        self.rate_limiter.tpm = tpm

    def mark_daily_exhausted(self):
        self.is_daily_exhausted = True

    def reset(self):
        self.peak_queue_length = 0
        self.is_daily_exhausted = False
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break

    async def submit(
        self,
        task_fn: Callable[..., Awaitable[Dict[str, Any]]],
        *args,
        estimated_tokens: int = 300,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Submits request through the bounded queue, respecting concurrency,
        rate limits, deadlines, and bounded retries.
        """
        start_time = time.time()
        deadline = start_time + self.request_timeout

        # Check daily quota immediately
        if self.is_daily_exhausted:
            return {
                "status": "rate_limited",
                "text": "Refusal: Daily request budget is exhausted. Please try again tomorrow.",
                "first_token_ms": 0.0,
                "total_ms": (time.time() - start_time) * 1000.0,
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "answer_grade": "refused_correctly",
                "error": "Daily quota exhausted"
            }

        # Track queue length
        cur_qsize = self.queue.qsize() + 1
        if cur_qsize > self.peak_queue_length:
            self.peak_queue_length = cur_qsize

        # Check queue overflow
        if self.queue.full():
            return {
                "status": "rate_limited",
                "text": "Refusal: System queue capacity exceeded. Server busy.",
                "first_token_ms": 0.0,
                "total_ms": (time.time() - start_time) * 1000.0,
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "answer_grade": "refused_correctly",
                "error": "Queue capacity exceeded"
            }

        # Enter queue
        token_obj = object()
        await self.queue.put(token_obj)

        try:
            # Wait for rate limit allocation
            acquired = await self.rate_limiter.acquire(estimated_tokens, deadline)
            if not acquired:
                return {
                    "status": "timeout" if time.time() >= deadline else "rate_limited",
                    "text": "Request timed out waiting for available rate limit quota.",
                    "first_token_ms": 0.0,
                    "total_ms": (time.time() - start_time) * 1000.0,
                    "retries": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "answer_grade": "refused_correctly",
                    "error": "Rate limit acquire timeout"
                }

            # Acquire concurrency semaphore
            sem_acquired = False
            try:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    raise asyncio.TimeoutError()
                await asyncio.wait_for(self.semaphore.acquire(), timeout=remaining_time)
                sem_acquired = True
            except (asyncio.TimeoutError, TimeoutError):
                return {
                    "status": "timeout",
                    "text": "Request timed out waiting for execution concurrency slot.",
                    "first_token_ms": 0.0,
                    "total_ms": (time.time() - start_time) * 1000.0,
                    "retries": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "answer_grade": "refused_correctly",
                    "error": "Semaphore wait timeout"
                }

            # Execute with bounded retries
            retries = 0
            while True:
                try:
                    time_left = max(0.5, deadline - time.time())
                    result = await asyncio.wait_for(task_fn(*args, **kwargs), timeout=time_left)
                    result["retries"] = retries
                    return result
                except (asyncio.TimeoutError, TimeoutError) as e:
                    if retries >= self.max_retries or time.time() >= deadline:
                        return {
                            "status": "timeout",
                            "text": "Request timed out during model execution.",
                            "first_token_ms": 0.0,
                            "total_ms": (time.time() - start_time) * 1000.0,
                            "retries": retries,
                            "tokens_in": estimated_tokens,
                            "tokens_out": 0,
                            "answer_grade": "refused_correctly",
                            "error": str(e)
                        }
                    retries += 1
                    await asyncio.sleep(0.5 * retries)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "daily" in err_msg or "daily_quota_exceeded" in err_msg:
                        self.mark_daily_exhausted()
                        return {
                            "status": "rate_limited",
                            "text": "Refusal: Daily request budget is exhausted.",
                            "first_token_ms": 0.0,
                            "total_ms": (time.time() - start_time) * 1000.0,
                            "retries": retries,
                            "tokens_in": estimated_tokens,
                            "tokens_out": 0,
                            "answer_grade": "refused_correctly",
                            "error": "Daily quota exhausted"
                        }
                    if "429" in err_msg or "rate limit" in err_msg:
                        if retries >= self.max_retries or time.time() >= deadline:
                            return {
                                "status": "rate_limited",
                                "text": "Request rejected due to rate limits.",
                                "first_token_ms": 0.0,
                                "total_ms": (time.time() - start_time) * 1000.0,
                                "retries": retries,
                                "tokens_in": estimated_tokens,
                                "tokens_out": 0,
                                "answer_grade": "refused_correctly",
                                "error": "429 Rate limit exceeded"
                            }
                        retries += 1
                        await asyncio.sleep(min(1.0 * (2 ** (retries - 1)), 3.0))
                    else:
                        if retries >= self.max_retries:
                            return {
                                "status": "error",
                                "text": f"Pipeline execution error: {e}",
                                "first_token_ms": 0.0,
                                "total_ms": (time.time() - start_time) * 1000.0,
                                "retries": retries,
                                "tokens_in": estimated_tokens,
                                "tokens_out": 0,
                                "answer_grade": "refused_correctly",
                                "error": str(e)
                            }
                        retries += 1
                        await asyncio.sleep(0.5)

            if sem_acquired:
                self.semaphore.release()

        finally:
            # Drain queue token
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                pass
            if 'sem_acquired' in locals() and sem_acquired:
                try:
                    self.semaphore.release()
                except ValueError:
                    pass
