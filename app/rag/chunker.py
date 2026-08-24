from pathlib import Path
from typing import List, Dict
import re

BASE_KB_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_base"


def chunk_markdown(text: str, source_name: str, max_words: int = 100, overlap: int = 15) -> List[Dict[str, str]]:
    """
    Splits markdown documents into semantic chunks preserving section header context.
    """
    sections = re.split(r"(?m)^(?=#{1,3}\s+)", text)
    chunks = []
    chunk_idx = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract header if present
        header_match = re.match(r"^(#{1,3}\s+[^\n]+)", section)
        header = header_match.group(1).strip() if header_match else "General"

        words = re.findall(r"\S+", section)
        if len(words) <= max_words:
            chunks.append({
                "id": f"{Path(source_name).stem}-{chunk_idx}",
                "source": source_name,
                "section": header,
                "text": section
            })
            chunk_idx += 1
        else:
            start = 0
            while start < len(words):
                end = min(start + max_words, len(words))
                sub_text = " ".join(words[start:end])
                chunks.append({
                    "id": f"{Path(source_name).stem}-{chunk_idx}",
                    "source": source_name,
                    "section": header,
                    "text": sub_text
                })
                chunk_idx += 1
                if end == len(words):
                    break
                start = end - overlap

    return chunks


def load_all_knowledge_documents(kb_dir: Path) -> List[Dict[str, str]]:
    all_chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(content, path.name))
    return all_chunks
