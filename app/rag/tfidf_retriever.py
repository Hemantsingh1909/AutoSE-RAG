from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    """
    Sparse Lexical Retriever using TF-IDF with n-gram features.
    """
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        texts = [f"{d.get('section', '')} {d['text']}" for d in self.documents]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.documents:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            score = float(scores[idx])
            if score > 0.001:
                results.append({
                    **self.documents[idx],
                    "score": round(score, 4),
                    "retriever": "tfidf",
                    "rank": rank + 1
                })
        return results
