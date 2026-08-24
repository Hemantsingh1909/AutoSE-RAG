from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from .chunker import load_all_knowledge_documents
from .tfidf_retriever import TFIDFRetriever
from .dense_retriever import DenseRetriever


BASE_KB_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"


class HybridRetriever:
    """
    Hybrid Retriever combining Sparse (TF-IDF) and Dense (FAISS) via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, kb_dir: Path = BASE_KB_DIR, rrf_k: int = 60):
        self.kb_dir = kb_dir
        self.rrf_k = rrf_k
        self.docs = load_all_knowledge_documents(self.kb_dir)
        self.tfidf = TFIDFRetriever(self.docs)
        self.dense = DenseRetriever(self.docs)

    def retrieve(self, query: str, top_k: int = 4, mode: str = "hybrid") -> List[Dict]:
        """
        mode: 'hybrid' | 'dense' | 'tfidf'
        """
        if mode == "tfidf":
            return self.tfidf.retrieve(query, top_k)
        elif mode == "dense":
            return self.dense.retrieve(query, top_k)

        # Hybrid RRF
        sparse_res = self.tfidf.retrieve(query, top_k=top_k * 2)
        dense_res = self.dense.retrieve(query, top_k=top_k * 2)

        rrf_scores = defaultdict(float)
        item_map = {}

        for rank, item in enumerate(sparse_res, start=1):
            doc_id = item["id"]
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            item_map[doc_id] = item

        for rank, item in enumerate(dense_res, start=1):
            doc_id = item["id"]
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)
            item_map[doc_id] = item

        # Sort items by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_results = []
        for rank, doc_id in enumerate(sorted_ids[:top_k], start=1):
            item = dict(item_map[doc_id])
            item["score"] = round(float(rrf_scores[doc_id] * 100), 4)
            item["retriever"] = "hybrid_rrf"
            item["rank"] = rank
            final_results.append(item)

        return final_results


# Compatibility alias for existing codebase
Retriever = HybridRetriever
