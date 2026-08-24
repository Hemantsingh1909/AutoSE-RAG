from pathlib import Path
from typing import List, Dict
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent / "knowledge_base"


def _chunk(text: str, size: int = 120, overlap: int = 20) -> List[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def load_documents() -> List[Dict[str, str]]:
    docs = []
    for path in sorted(BASE.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk(text)):
            docs.append({"id": f"{path.stem}-{i}", "source": path.name, "text": chunk})
    return docs


class Retriever:
    def __init__(self):
        self.docs = load_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.docs])

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, str]]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [
            {**self.docs[i], "score": round(float(scores[i]), 4)}
            for i in ranked
            if scores[i] > 0
        ]
