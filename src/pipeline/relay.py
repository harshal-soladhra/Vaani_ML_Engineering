import re
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEFAULT_MODEL, STANDIN_URL
from src.rag.retriever import retrieve_passages
from src.llm.groq_client import GroqClient, GroqRateLimitExceeded
import httpx

CITATION_REGEX = re.compile(r"\[([A-Za-z0-9\s\:\-\,]+?)(?:,\s*([^,\]]+?))?(?:,\s*p(?:p)?\.?\s*([0-9\-]+))?\]")

# Known library terms and papers
LIBRARY_PAPERS = [
    "attention is all you need",
    "moshi",
    "dynamic chunking"
]

class RelayPipeline:
    def __init__(
        self,
        source: str = "real", # "real" or "stand-in"
        retrieval_setting: str = "setting_1",
        evidence_threshold: float = 0.05,
        groq_client: Optional[GroqClient] = None
    ):
        self.source = source
        self.retrieval_setting = retrieval_setting
        self.evidence_threshold = evidence_threshold
        self.groq_client = groq_client or (GroqClient() if source == "real" else None)

    def understand_query(self, query: str) -> Dict[str, Any]:
        """Step 1: Query Understanding and Classification"""
        q_lower = query.lower()
        # Explicit out of library indicators
        out_of_scope_keywords = ["llama 3", "gemini", "gpt-4", "resnet", "vgg", "alexnet", "whisper", "bert", "electra", "t5", "chinchilla"]
        is_known_out_of_scope = any(kw in q_lower for kw in out_of_scope_keywords)
        return {
            "query": query,
            "is_known_out_of_scope": is_known_out_of_scope,
            "target_papers": [p for p in LIBRARY_PAPERS if p in q_lower]
        }

    def validate_citations(self, answer_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validates that answer includes citations and that citations correspond to retrieved chunks"""
        matches = CITATION_REGEX.findall(answer_text)
        if not matches:
            # Check for simpler citations like [Attention Is All You Need] or [1]
            simple_matches = re.findall(r"\[([A-Za-z0-9\s\-]+)\]", answer_text)
            if not simple_matches:
                return False, []
            matches = [(m, "", "") for m in simple_matches]

        valid_titles = {c["paper_title"].lower() for c in retrieved_chunks}
        found_valid = False
        parsed_citations = []

        for m in matches:
            paper_ref = m[0].strip().lower()
            parsed_citations.append(" ".join(m).strip())
            # Check if any valid title matches
            for vt in valid_titles:
                if paper_ref in vt or vt in paper_ref or any(w in vt for w in paper_ref.split() if len(w) > 4):
                    found_valid = True
                    break

        return found_valid, parsed_citations

    async def _call_standin(
        self,
        messages: List[Dict[str, str]],
        timeout: float = 12.0
    ) -> Dict[str, Any]:
        """Calls stand-in service with streaming simulation and metrics"""
        from src.simulator.server import app
        t0 = time.perf_counter()
        
        # Connect via ASGITransport directly or HTTP
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://standin", timeout=timeout) as client:
            payload = {
                "model": "stand-in-groq",
                "messages": messages,
                "temperature": 0.2,
                "stream": True
            }
            async with client.stream("POST", "/openai/v1/chat/completions", json=payload) as resp:
                if resp.status_code == 429:
                    err_json = await resp.json()
                    raise GroqRateLimitExceeded(
                        err_json.get("error", {}).get("message", "429 Rate Limit"),
                        is_daily=("daily" in str(err_json).lower())
                    )
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    raise RuntimeError(f"Stand-in error {resp.status_code}: {err_text.decode('utf-8', errors='ignore')}")

                first_token_time = None
                tokens_out = 0
                text_parts = []
                async for line_bytes in resp.aiter_lines():
                    line = line_bytes.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                tokens_out += 1
                                text_parts.append(content)
                    except json.JSONDecodeError:
                        pass

        t_end = time.perf_counter()
        total_ms = (t_end - t0) * 1000.0
        first_token_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_ms
        tokens_in = max(1, sum(len(m["content"].split()) for m in messages))
        return {
            "text": "".join(text_parts),
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "retries": 0,
            "headers": dict(resp.headers)
        }

    async def run(
        self,
        request_id: str,
        query: str,
        scenario: str = "steady"
    ) -> Dict[str, Any]:
        """
        Executes the 4-step pipeline:
        1. Query Understanding
        2. Retrieval Tool Call
        3. Evidence Checking & Branch
        4. Answer Generation & Citation Validation Loop
        """
        start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t_pipeline_start = time.perf_counter()

        # Step 1: Query Understanding
        q_info = self.understand_query(query)

        # Step 2: Retrieval Tool Call
        retrieved_chunks, max_evidence = retrieve_passages(query, self.retrieval_setting)

        # Step 3: Evidence Checking & Decision Branch
        # If question is explicitly out-of-scope or retrieval evidence is below threshold
        if q_info["is_known_out_of_scope"] or max_evidence < self.evidence_threshold:
            total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            refusal_text = (
                "The provided library of research papers (Attention Is All You Need, Moshi, "
                "Dynamic Chunking) does not contain sufficient information to answer this question."
            )
            return {
                "request_id": request_id,
                "scenario": scenario,
                "source": self.source,
                "start_time": start_time_iso,
                "first_token_ms": 0.0,
                "total_ms": round(total_ms, 2),
                "status": "ok",
                "retries": 0,
                "tokens_in": max(1, int(len(query.split()) * 1.3)),
                "tokens_out": 25,
                "retrieval_setting": self.retrieval_setting,
                "answer_grade": "refused_correctly",
                "answer": refusal_text,
                "citations": []
            }

        # Step 4: Answer Generation & Citation Validation Loop
        passages_text = "\n\n".join([
            f"Passage [{c['paper_title']}, {c['section']}, p. {c['page']}]:\n{c['text']}"
            for c in retrieved_chunks
        ])

        system_prompt = (
            "You are an expert ML research assistant answering questions using only the provided research papers.\n"
            "Rules:\n"
            "1. Answer concisely and accurately using ONLY the context passages below.\n"
            "2. Every factual statement must cite its exact source in the format: [Paper Title, Section, Page].\n"
            "3. If the passages do not state the answer, reply that the papers do not say so."
        )

        user_content = f"Context passages from library:\n{passages_text}\n\nQuestion: {query}\n\nProvide a cited answer:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Call model (Groq or Stand-in)
        if self.source == "real":
            llm_res = await self.groq_client.generate(messages)
        else:
            llm_res = await self._call_standin(messages)

        answer_text = llm_res.get("text", "")
        first_token_ms = llm_res.get("first_token_ms", 0.0)
        total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        # Citation validation loop
        has_valid_citations, citations = self.validate_citations(answer_text, retrieved_chunks)

        # Loop if citations are missing and we are using real Groq (or stand-in)
        if not has_valid_citations and self.source == "real":
            # 1 Bounded re-prompt loop
            correction_messages = list(messages)
            correction_messages.append({"role": "assistant", "content": answer_text})
            correction_messages.append({
                "role": "user",
                "content": "Your answer is missing strict source citations. Rewrite your answer adding citations in the exact format [Paper Title, Section, p. Page] for every claim."
            })
            retry_res = await self.groq_client.generate(correction_messages, max_tokens=256)
            retry_answer = retry_res.get("text", "")
            has_valid_citations, citations = self.validate_citations(retry_answer, retrieved_chunks)
            if has_valid_citations:
                answer_text = retry_answer
            total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            llm_res["tokens_in"] += retry_res.get("tokens_in", 0)
            llm_res["tokens_out"] += retry_res.get("tokens_out", 0)

        # Grade the answer
        if has_valid_citations and len(answer_text) > 20:
            answer_grade = "correct"
        elif "do not say" in answer_text.lower() or "not contain" in answer_text.lower() or "cannot answer" in answer_text.lower():
            answer_grade = "refused_correctly"
        else:
            # Missing citations or hallucinated
            answer_grade = "wrong" if len(answer_text) > 10 else "refused_wrongly"

        return {
            "request_id": request_id,
            "scenario": scenario,
            "source": self.source,
            "start_time": start_time_iso,
            "first_token_ms": round(first_token_ms, 2),
            "total_ms": round(total_ms, 2),
            "status": "ok",
            "retries": llm_res.get("retries", 0),
            "tokens_in": llm_res.get("tokens_in", 0),
            "tokens_out": llm_res.get("tokens_out", 0),
            "retrieval_setting": self.retrieval_setting,
            "answer_grade": answer_grade,
            "answer": answer_text,
            "citations": citations
        }
