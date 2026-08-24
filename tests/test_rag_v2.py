import pytest
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.tfidf_retriever import TFIDFRetriever
from app.rag.dense_retriever import DenseRetriever
from app.rag.chunker import load_all_knowledge_documents, BASE_KB_DIR


def test_knowledge_base_loading():
    docs = load_all_knowledge_documents(BASE_KB_DIR)
    assert len(docs) >= 5
    assert any("iso26262" in d["id"] for d in docs)
    assert any("autosar" in d["id"] for d in docs)


def test_tfidf_retrieval():
    docs = load_all_knowledge_documents(BASE_KB_DIR)
    tfidf = TFIDFRetriever(docs)
    results = tfidf.retrieve("watchdog heartbeat timer and safe state", top_k=3)
    assert len(results) > 0
    assert any("watchdog" in r["text"].lower() or "heartbeat" in r["text"].lower() for r in results)


def test_dense_faiss_retrieval():
    docs = load_all_knowledge_documents(BASE_KB_DIR)
    dense = DenseRetriever(docs)
    results = dense.retrieve("CRC checksum and sequence alive counter", top_k=3)
    assert len(results) > 0


def test_hybrid_rrf_retrieval():
    retriever = HybridRetriever()
    results = retriever.retrieve("dual redundant potentiometer pedal sensor", top_k=4, mode="hybrid")
    assert len(results) > 0
    assert results[0]["retriever"] == "hybrid_rrf"
    assert "score" in results[0]
