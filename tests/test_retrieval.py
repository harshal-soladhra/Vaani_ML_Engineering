import pytest
from src.rag.retriever import retrieve_passages, PaperRetriever, get_retriever

def test_retriever_initialization():
    r300 = get_retriever(300)
    assert len(r300.chunks) > 0
    assert r300.tfidf_matrix.shape[0] == len(r300.chunks)

def test_metadata_preservation():
    r = get_retriever(300)
    chunk = r.chunks[0]
    assert "paper_id" in chunk
    assert "paper_title" in chunk
    assert "page" in chunk
    assert "section" in chunk
    assert "text" in chunk
    assert chunk["page"] >= 1
    assert len(chunk["paper_title"]) > 0

def test_setting_1_retrieval():
    chunks, score = retrieve_passages("scaled dot-product attention", "setting_1")
    assert len(chunks) <= 3
    assert score > 0.0
    assert any("attention" in c["paper_title"].lower() for c in chunks)

def test_setting_2_retrieval():
    chunks, score = retrieve_passages("Moshi speech language model audio tokens", "setting_2")
    assert len(chunks) <= 5
    assert score > 0.0
    assert any("moshi" in c["paper_title"].lower() for c in chunks)

def test_setting_3_reranking():
    chunks, score = retrieve_passages("dynamic chunking hierarchical sequence modeling", "setting_3")
    assert len(chunks) <= 5
    assert score > 0.0
    # Check that reranked scores are sorted descending
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)
