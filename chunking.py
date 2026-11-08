from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    strategy: str
    index: int


def load_documents(directory: Path) -> list[Document]:
    docs = []
    for path in sorted(directory.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        docs.append(Document(path.stem, lines[0].lstrip("# "), " ".join(lines[1:]).strip()))
    return docs


def sentences(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?])\s+", text.strip()) if value.strip()]


def fixed_chunks(doc: Document, size: int = 360, overlap: int = 80) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be nonnegative and smaller than chunk size")
    parts = []
    for start in range(0, len(doc.text), size - overlap):
        parts.append(doc.text[start : start + size])
        if start + size >= len(doc.text):
            break
    return [
        Chunk(f"{doc.document_id}:fixed:{i}", doc.document_id, doc.title, text, "fixed", i)
        for i, text in enumerate(parts)
    ]


def sentence_chunks(doc: Document, per_chunk: int = 2) -> list[Chunk]:
    if per_chunk < 1:
        raise ValueError("At least one sentence per chunk is required")
    parts = sentences(doc.text)
    return [
        Chunk(
            f"{doc.document_id}:sentence:{i // per_chunk}",
            doc.document_id,
            doc.title,
            " ".join(parts[i : i + per_chunk]),
            "sentence",
            i // per_chunk,
        )
        for i in range(0, len(parts), per_chunk)
    ]


CHUNKERS = {"fixed": fixed_chunks, "sentence": sentence_chunks}
