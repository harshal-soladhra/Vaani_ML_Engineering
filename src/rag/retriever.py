import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import INDEX_DIR

class PaperRetriever:
    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size
        self.chunks_path = INDEX_DIR / f"chunks_{chunk_size}.json"
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunk index: {self.chunks_path}")
        
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)
            
        self.texts = [c["text"] for c in self.chunks]
        # Sublinear TF scaling with 1-2 ngrams for rich technical terminology matching
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_df=0.85
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 3, rerank: bool = False) -> List[Dict[str, Any]]:
        clean_q = query.strip()
        if not clean_q:
            return []

        q_vec = self.vectorizer.transform([clean_q])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        if not rerank:
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(sims[idx])
                chunk["citation"] = f"[{chunk['paper_title']}, {chunk['section']}, p. {chunk['page']}]"
                results.append(chunk)
            return results

        # Two-stage retrieval with reranking
        initial_candidates_count = max(top_k * 3, 12)
        candidate_indices = np.argsort(sims)[::-1][:initial_candidates_count]

        # Extract meaningful query keywords
        q_terms = [t.lower() for t in re.findall(r"\b[A-Za-z0-9\-\_]{3,}\b", clean_q) if t.lower() not in {"what", "which", "where", "how", "does", "from", "with", "this", "that"}]

        scored_candidates = []
        for idx in candidate_indices:
            chunk = dict(self.chunks[idx])
            base_score = float(sims[idx])
            text_lower = chunk["text"].lower()

            # Exact keyword coverage score
            matched_terms = sum(1 for term in q_terms if term in text_lower)
            coverage = (matched_terms / max(1, len(q_terms)))

            # Phrase density / boost
            rerank_score = 0.6 * base_score + 0.4 * coverage
            chunk["score"] = float(rerank_score)
            chunk["base_score"] = base_score
            chunk["coverage"] = coverage
            chunk["citation"] = f"[{chunk['paper_title']}, {chunk['section']}, p. {chunk['page']}]"
            scored_candidates.append(chunk)

        # Sort by reranked score and return top_k
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

# Cache retrievers for the three settings
_RETRIEVERS: Dict[int, PaperRetriever] = {}

def get_retriever(chunk_size: int = 300) -> PaperRetriever:
    if chunk_size not in _RETRIEVERS:
        _RETRIEVERS[chunk_size] = PaperRetriever(chunk_size)
    return _RETRIEVERS[chunk_size]

def retrieve_passages(query: str, setting: str = "setting_1") -> Tuple[List[Dict[str, Any]], float]:
    """
    Standard retrieval interface mapping to the three experiment settings:
    - 'setting_1': 300-token chunks, Top-3, no reranking
    - 'setting_2': 600-token chunks, Top-5, no reranking
    - 'setting_3': 800-token chunks, Top-5, reranking
    Returns (retrieved_chunks, max_evidence_score)
    """
    if setting == "setting_1":
        retriever = get_retriever(300)
        chunks = retriever.retrieve(query, top_k=3, rerank=False)
    elif setting == "setting_2":
        retriever = get_retriever(600)
        chunks = retriever.retrieve(query, top_k=5, rerank=False)
    elif setting == "setting_3":
        retriever = get_retriever(800)
        chunks = retriever.retrieve(query, top_k=5, rerank=True)
    else:
        retriever = get_retriever(300)
        chunks = retriever.retrieve(query, top_k=3, rerank=False)

    max_score = max([c["score"] for c in chunks], default=0.0)
    return chunks, max_score
