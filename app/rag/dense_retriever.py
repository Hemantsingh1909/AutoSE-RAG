from typing import List, Dict, Optional
import numpy as np
import os
import faiss


class DenseRetriever:
    """
    Dense Semantic Retriever using Sentence-Transformers and FAISS vector indexing.
    """
    def __init__(self, documents: List[Dict[str, str]], model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.model_name = model_name
        self.model = None
        self.dimension = 384
        self.index = None
        self._init_embeddings()

    def _init_embeddings(self):
        if not self.documents:
            return

        texts = [f"{d.get('section', '')} {d['text']}" for d in self.documents]

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self.dimension = embeddings.shape[1]
        except Exception:
            # Fallback deterministic pseudo-dense embedding if model download is offline
            from sklearn.feature_extraction.text import HashingVectorizer
            hv = HashingVectorizer(n_features=384, alternate_sign=False)
            embeddings = hv.transform(texts).toarray().astype(np.float32)
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = (embeddings / norms).astype(np.float32)
            self.dimension = 384

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.documents or self.index is None:
            return []

        if self.model is not None:
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        else:
            from sklearn.feature_extraction.text import HashingVectorizer
            hv = HashingVectorizer(n_features=384, alternate_sign=False)
            q_emb = hv.transform([query]).toarray().astype(np.float32)
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm

        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        scores, indices = self.index.search(q_emb, min(top_k, len(self.documents)))

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1 and score > 0.001:
                results.append({
                    **self.documents[idx],
                    "score": round(float(score), 4),
                    "retriever": "dense_faiss",
                    "rank": rank + 1
                })
        return results
