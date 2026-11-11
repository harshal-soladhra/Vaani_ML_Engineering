import pytest
from src.pipeline.relay import RelayPipeline

@pytest.mark.asyncio
async def test_pipeline_out_of_scope_refusal():
    pipeline = RelayPipeline(source="stand-in")
    out_of_scope_q = "What is the parameter size and training dataset for Llama 3?"
    res = await pipeline.run("req_test_refusal", out_of_scope_q)
    assert res["status"] == "ok"
    assert res["answer_grade"] == "refused_correctly"
    assert "does not contain sufficient information" in res["answer"]

@pytest.mark.asyncio
async def test_pipeline_query_understanding():
    pipeline = RelayPipeline(source="stand-in")
    res = pipeline.understand_query("How does Moshi process audio?")
    assert "moshi" in res["target_papers"]
    assert res["is_known_out_of_scope"] is False
