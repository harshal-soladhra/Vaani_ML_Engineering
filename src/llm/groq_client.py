import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import GROQ_API_KEY, DEFAULT_MODEL

class GroqRateLimitExceeded(Exception):
    def __init__(self, message: str, retry_after: float = 1.0, is_daily: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_daily = is_daily

class GroqClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 12.0,
        max_retries: int = 2
    ):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.last_rate_limits: Dict[str, Any] = {}

    def _extract_rate_limits(self, headers: httpx.Headers) -> Dict[str, Any]:
        info = {}
        for k in [
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "retry-after"
        ]:
            if k in headers:
                info[k] = headers[k]
        self.last_rate_limits = info
        return info

    def _parse_retry_after(self, headers: httpx.Headers, default: float = 1.0) -> Tuple[float, bool]:
        """Returns (retry_after_seconds, is_daily_exhaustion)"""
        rem_req = headers.get("x-ratelimit-remaining-requests")
        reset_req = headers.get("x-ratelimit-reset-requests", "")
        
        # Check if daily limit reached
        is_daily = False
        if rem_req == "0" or "h" in reset_req:
            is_daily = True

        retry_header = headers.get("retry-after")
        if retry_header:
            try:
                return float(retry_header), is_daily
            except ValueError:
                pass

        reset_tokens = headers.get("x-ratelimit-reset-tokens")
        if reset_tokens and reset_tokens.endswith("ms"):
            try:
                ms = float(reset_tokens.replace("ms", ""))
                return max(0.2, ms / 1000.0), is_daily
            except ValueError:
                pass
        return default, is_daily

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_override: Optional[float] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams response chunks from Groq API with bounded exponential backoff retries.
        Yields events:
        {"type": "token", "content": "..."}
        {"type": "metrics", "first_token_ms": float, "total_ms": float, "tokens_in": int, "tokens_out": int, "retries": int, "headers": dict}
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        call_timeout = timeout_override or self.timeout
        retries = 0

        while True:
            t_start = time.perf_counter()
            first_token_time: Optional[float] = None
            tokens_out = 0
            # Rough estimate of tokens in based on chars
            prompt_str = " ".join([m["content"] for m in messages])
            tokens_in = max(1, int(len(prompt_str) / 4))

            try:
                async with httpx.AsyncClient(timeout=call_timeout) as client:
                    async with client.stream("POST", self.endpoint, headers=headers, json=payload) as resp:
                        rate_limits = self._extract_rate_limits(resp.headers)

                        if resp.status_code == 429:
                            retry_after, is_daily = self._parse_retry_after(resp.headers)
                            if is_daily or retries >= self.max_retries:
                                raise GroqRateLimitExceeded(
                                    f"Groq 429 Rate Limit Exceeded (Daily: {is_daily})",
                                    retry_after=retry_after,
                                    is_daily=is_daily
                                )
                            retries += 1
                            wait_time = min(retry_after * (2 ** (retries - 1)), 5.0)
                            await asyncio.sleep(wait_time)
                            continue

                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            raise RuntimeError(f"Groq API error {resp.status_code}: {err_body.decode('utf-8', errors='ignore')}")

                        # Stream reading
                        buffer = ""
                        async for chunk_bytes in resp.aiter_bytes():
                            buffer += chunk_bytes.decode("utf-8", errors="ignore")
                            lines = buffer.split("\n")
                            buffer = lines.pop() # preserve incomplete trailing line

                            for line in lines:
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    continue
                                try:
                                    parsed = json.loads(data_str)
                                    choices = parsed.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            if first_token_time is None:
                                                first_token_time = time.perf_counter()
                                            tokens_out += 1
                                            yield {"type": "token", "content": content}
                                    # Usage if sent in final stream chunk
                                    if "usage" in parsed and parsed["usage"]:
                                        u = parsed["usage"]
                                        tokens_in = u.get("prompt_tokens", tokens_in)
                                        tokens_out = u.get("completion_tokens", tokens_out)
                                except json.JSONDecodeError:
                                    pass

                t_end = time.perf_counter()
                total_ms = (t_end - t_start) * 1000.0
                first_token_ms = (first_token_time - t_start) * 1000.0 if first_token_time else total_ms

                yield {
                    "type": "metrics",
                    "first_token_ms": first_token_ms,
                    "total_ms": total_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "retries": retries,
                    "headers": rate_limits
                }
                return

            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                if retries >= self.max_retries:
                    raise TimeoutError(f"Groq API timed out after {call_timeout}s: {e}")
                retries += 1
                await asyncio.sleep(0.5 * retries)
            except GroqRateLimitExceeded:
                raise
            except Exception as e:
                if retries >= self.max_retries:
                    raise
                retries += 1
                await asyncio.sleep(0.5 * retries)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Non-streaming collector helper"""
        text_parts = []
        metrics = {}
        async for event in self.generate_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_override=timeout_override
        ):
            if event["type"] == "token":
                text_parts.append(event["content"])
            elif event["type"] == "metrics":
                metrics = event

        full_text = "".join(text_parts)
        metrics["text"] = full_text
        return metrics
