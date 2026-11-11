import pytest
from src.pipeline.relay import RelayPipeline

def test_valid_citation_validation():
    pipeline = RelayPipeline(source="stand-in")
    dummy_chunks = [
        {"paper_title": "Attention Is All You Need", "section": "3.2 Attention", "page": 4},
        {"paper_title": "Moshi: a speech-text foundation model", "section": "2 Architecture", "page": 3}
    ]
    valid_text = (
        "Multi-head attention uses scaled dot-product operations [Attention Is All You Need, 3.2 Attention, p. 4]. "
        "Audio codec streams are processed simultaneously [Moshi: a speech-text foundation model, 2 Architecture, p. 3]."
    )
    valid, citations = pipeline.validate_citations(valid_text, dummy_chunks)
    assert valid is True
    assert len(citations) >= 2

def test_missing_citation_rejection():
    pipeline = RelayPipeline(source="stand-in")
    dummy_chunks = [{"paper_title": "Attention Is All You Need", "section": "3.2", "page": 4}]
    uncited_text = "Transformers are neural network models that rely on attention without recurrence."
    valid, citations = pipeline.validate_citations(uncited_text, dummy_chunks)
    assert valid is False
    assert len(citations) == 0

def test_hallucinated_paper_citation():
    pipeline = RelayPipeline(source="stand-in")
    dummy_chunks = [{"paper_title": "Attention Is All You Need", "section": "3.2", "page": 4}]
    hallucinated_text = "The model uses residual streams [Llama 3 Technical Report, Section 2, p. 10]."
    valid, citations = pipeline.validate_citations(hallucinated_text, dummy_chunks)
    assert valid is False
