import asyncio
import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Stand-in Groq Simulator")

class SimulatorConfig(BaseModel):
    rpm_limit: int = 30
    tpm_limit: int = 8000
    daily_request_limit: int = 1000
    base_ttft_ms: float = 1200.0
    ttft_std_ms: float = 150.0
    token_interval_ms: float = 15.0
    failure_rate: float = 0.02
    timeout_rate: float = 0.01
    load_factor: float = 0.25
    daily_budget_exhausted: bool = False

config = SimulatorConfig()

# State tracking
in_flight_requests = 0
request_timestamps: List[float] = []
token_usage_timestamps: List[Tuple[float, int]] = []
daily_request_count = 0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "stand-in-groq"
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 256
    stream: bool = True

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "in_flight": in_flight_requests,
        "daily_requests": daily_request_count,
        "config": config.dict()
    }

@app.post("/configure")
def configure(new_config: SimulatorConfig):
    global config, daily_request_count
    config = new_config
    if config.daily_budget_exhausted:
        daily_request_count = config.daily_request_limit
    return {"status": "updated", "config": config}

@app.post("/reset")
def reset_state():
    global in_flight_requests, request_timestamps, token_usage_timestamps, daily_request_count
    in_flight_requests = 0
    request_timestamps.clear()
    token_usage_timestamps.clear()
    daily_request_count = 0
    config.daily_budget_exhausted = False
    return {"status": "reset"}

def check_rate_limits(estimated_tokens: int) -> Optional[Response]:
    global request_timestamps, token_usage_timestamps, daily_request_count
    now = time.time()
    
    # 1. Daily limit check
    if config.daily_budget_exhausted or daily_request_count >= config.daily_request_limit:
        headers = {
            "x-ratelimit-limit-requests": str(config.daily_request_limit),
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "8h42m",
            "retry-after": "31320"
        }
        body = {
            "error": {
                "message": f"Daily request budget exhausted ({config.daily_request_limit}/{config.daily_request_limit}). Your quota will reset at midnight UTC.",
                "type": "daily_quota_exceeded",
                "code": "rate_limit_exceeded"
            }
        }
        return Response(content=json.dumps(body), status_code=429, headers=headers, media_type="application/json")

    # 2. Prune 60-second window
    request_timestamps = [t for t in request_timestamps if now - t < 60.0]
    token_usage_timestamps = [(t, tok) for t, tok in token_usage_timestamps if now - t < 60.0]

    # RPM check
    if len(request_timestamps) >= config.rpm_limit:
        oldest = request_timestamps[0]
        retry_after = max(0.5, round(60.0 - (now - oldest), 2))
        headers = {
            "x-ratelimit-limit-requests": str(config.rpm_limit),
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": f"{retry_after}s",
            "retry-after": str(retry_after)
        }
        body = {
            "error": {
                "message": f"Rate limit reached for requests per minute: {config.rpm_limit}.",
                "type": "requests",
                "code": "rate_limit_exceeded"
            }
        }
        return Response(content=json.dumps(body), status_code=429, headers=headers, media_type="application/json")

    # TPM check
    current_tokens = sum(tok for _, tok in token_usage_timestamps)
    if current_tokens + estimated_tokens > config.tpm_limit:
        oldest_t = token_usage_timestamps[0][0] if token_usage_timestamps else now
        retry_after = max(0.5, round(60.0 - (now - oldest_t), 2))
        headers = {
            "x-ratelimit-limit-tokens": str(config.tpm_limit),
            "x-ratelimit-remaining-tokens": str(max(0, config.tpm_limit - current_tokens)),
            "x-ratelimit-reset-tokens": f"{retry_after}s",
            "retry-after": str(retry_after)
        }
        body = {
            "error": {
                "message": f"Rate limit reached for tokens per minute: {config.tpm_limit}.",
                "type": "tokens",
                "code": "rate_limit_exceeded"
            }
        }
        return Response(content=json.dumps(body), status_code=429, headers=headers, media_type="application/json")

    # Record usage
    request_timestamps.append(now)
    token_usage_timestamps.append((now, estimated_tokens))
    daily_request_count += 1
    return None

def generate_mock_reply(messages: List[ChatMessage]) -> str:
    user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
    # Check if this contains citations or RAG context
    if "Context passages from library:" in user_msg:
        # Generate an authoritative cited response based on passages
        lines = user_msg.split("\n")
        citations = []
        for line in lines:
            if line.startswith("[") and "]" in line:
                citations.append(line.split("\n")[0].strip())
        cit_str = citations[0] if citations else "[Attention Is All You Need, 3.2 Attention, p. 4]"
        return (
            f"Based on the provided research papers, the architecture relies on multi-head scaled dot-product attention "
            f"to model global dependencies without recurrence {cit_str}. "
            f"Specifically, attention weights are computed by taking the dot product of queries and keys divided by the square root of d_k {cit_str}."
        )
    return "Based on the library papers, the mechanism is defined by structured attention and speech representations."

@app.post("/openai/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global in_flight_requests
    estimated_tokens = sum(len(m.content.split()) for m in req.messages) + 80
    
    # Check rate limits
    err_resp = check_rate_limits(estimated_tokens)
    if err_resp:
        return err_resp

    # Check random timeout / failure
    if random.random() < config.timeout_rate:
        # Simulate long delay exceeding client timeout
        await asyncio.sleep(15.0)
        raise HTTPException(status_code=504, detail="Upstream gateway timeout")

    if random.random() < config.failure_rate:
        raise HTTPException(status_code=503, detail="Simulated service unavailable (Groq node transient failure)")

    in_flight_requests += 1
    try:
        # Load dependent slowdown
        slowdown = 1.0
        if in_flight_requests > 2:
            slowdown = 1.0 + config.load_factor * (in_flight_requests - 2)

        raw_ttft = max(100.0, random.gauss(config.base_ttft_ms, config.ttft_std_ms))
        ttft_s = (raw_ttft * slowdown) / 1000.0

        reply_text = generate_mock_reply(req.messages)
        words = reply_text.split()

        rem_req = max(0, config.rpm_limit - len(request_timestamps))
        headers = {
            "x-ratelimit-limit-requests": str(config.rpm_limit),
            "x-ratelimit-limit-tokens": str(config.tpm_limit),
            "x-ratelimit-remaining-requests": str(rem_req),
            "x-ratelimit-remaining-tokens": str(max(0, config.tpm_limit - sum(tok for _, tok in token_usage_timestamps))),
            "x-ratelimit-reset-requests": "1.5s",
            "x-ratelimit-reset-tokens": "200ms"
        }

        if not req.stream:
            await asyncio.sleep(ttft_s + len(words) * (config.token_interval_ms / 1000.0))
            return {
                "id": f"standin-{int(time.time()*1000)}",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": estimated_tokens - 80,
                    "completion_tokens": len(words),
                    "total_tokens": estimated_tokens - 80 + len(words)
                }
            }

        async def event_generator():
            # Wait for TTFT
            await asyncio.sleep(ttft_s)
            
            for word in words:
                chunk = {
                    "id": f"standin-chunk-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(config.token_interval_ms / 1000.0)

            done_chunk = {
                "id": "standin-done",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": estimated_tokens - 80,
                    "completion_tokens": len(words),
                    "total_tokens": estimated_tokens - 80 + len(words)
                }
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=headers
        )

    finally:
        in_flight_requests = max(0, in_flight_requests - 1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
