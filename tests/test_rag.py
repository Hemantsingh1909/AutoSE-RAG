from app.rag import Retriever


def test_retrieval_returns_relevant_evidence():
    r = Retriever()
    results = r.retrieve("reject invalid sensor value and create diagnostic event")
    assert results
    combined = " ".join(x["text"].lower() for x in results)
    assert "diagnostic" in combined
    assert "invalid" in combined
