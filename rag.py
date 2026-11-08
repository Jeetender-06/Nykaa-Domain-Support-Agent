from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import hashlib
import json
import unicodedata
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from chunking import CHUNKERS, Document, load_documents
from config import Settings
from governance import RequestBudget
from models import Source

FALLBACK = "I don't know based on the available support knowledge. Please contact human support."
CALIBRATION_QUERIES = [
    ("How many days do I have to return an unworn pair of shoes?", True),
    ("When will a cash on delivery refund appear in my bank account?", True),
    ("How long does standard delivery take to an Indian metro city?", True),
    ("Can I cancel an order after warehouse packing has started?", True),
    ("What evidence is needed to report an item that arrived damaged?", True),
    ("Do you ship orders to countries outside India?", True),
    ("Explain the orbital period of Neptune.", False),
    ("Write a JavaScript function to sort a binary tree.", False),
    ("Who won the 1983 cricket world cup?", False),
]


def normalize_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


@dataclass
class RagResult:
    query: str
    answer: str
    sources: list[Source]
    supported: bool
    similarity: float
    cache_hit: bool = False

    def as_dict(self) -> dict:
        result = asdict(self)
        result["sources"] = [source.model_dump() for source in self.sources]
        return result


@dataclass
class RagCounters:
    searches: int = 0
    generations: int = 0
    cache_hits: int = 0


class RagEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not (settings.model_path / "config.json").exists():
            raise RuntimeError(
                "Local embedding model missing. Run: python scripts/download_model.py"
            )
        self.encoder = SentenceTransformer(
            str(settings.model_path),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )
        self.client = chromadb.PersistentClient(
            path=str(settings.runtime / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collections = {
            strategy: self.client.get_or_create_collection(
                name=f"nykaa_{strategy}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            for strategy in CHUNKERS
        }
        self.documents = load_documents(settings.kb_path)
        self.cache: OrderedDict[str, RagResult] = OrderedDict()
        self.counters = RagCounters()
        self.calibration: dict[str, Any] = {}
        self.strategy = "sentence"

    def fingerprint(self) -> str:
        content = json.dumps([asdict(doc) for doc in self.documents], sort_keys=True)
        return hashlib.sha256(
            (content + "fixed360/80:sentence2:MiniLM:c9745ed1").encode()
        ).hexdigest()

    def index(self) -> dict[str, int]:
        counts = {}
        for strategy, chunker in CHUNKERS.items():
            chunks = [chunk for document in self.documents for chunk in chunker(document)]
            ids = [chunk.chunk_id for chunk in chunks]
            collection = self.collections[strategy]
            old_ids = set(collection.get()["ids"]) - set(ids)
            if old_ids:
                collection.delete(ids=sorted(old_ids))
            embeddings = self.encoder.encode(
                [f"{chunk.title}: {chunk.text}" for chunk in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {"document_id": chunk.document_id, "title": chunk.title, "index": chunk.index}
                    for chunk in chunks
                ],
            )
            counts[strategy] = collection.count()
        self.cache.clear()
        return counts

    def retrieve(self, query: str, strategy: str | None = None) -> list[Source]:
        self.counters.searches += 1
        collection = self.collections[strategy or self.strategy]
        vector = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        result = collection.query(
            query_embeddings=vector,
            n_results=min(self.settings.top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Source(
                document_id=meta["document_id"],
                chunk_id=chunk_id,
                text=text,
                similarity=round(1 - float(distance), 8),
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def calibrate(self) -> dict:
        report = {
            "corpus_fingerprint": self.fingerprint(),
            "strategies": {},
            "selected_strategy": self.strategy,
        }
        for strategy in CHUNKERS:
            rows = []
            for query, in_scope in CALIBRATION_QUERIES:
                hits = self.retrieve(query, strategy)
                rows.append(
                    {"query": query, "in_scope": in_scope, "top1_cosine": hits[0].similarity}
                )
            low_in = min(row["top1_cosine"] for row in rows if row["in_scope"])
            high_out = max(row["top1_cosine"] for row in rows if not row["in_scope"])
            if low_in <= high_out:
                raise RuntimeError(
                    f"Calibration clusters overlap for {strategy}: {low_in} <= {high_out}; inspect corpus/queries"
                )
            report["strategies"][strategy] = {
                "measurements": rows,
                "min_in_scope": low_in,
                "max_out_of_scope": high_out,
                "threshold": round((low_in + high_out) / 2, 8),
            }
        self.calibration = report
        self.save_calibration()
        return report

    def save_calibration(self) -> None:
        self.settings.calibration_path.write_text(json.dumps(self.calibration, indent=2) + "\n", encoding="utf-8")

    def prepare(self) -> None:
        # Always upsert: idempotent and removes stale chunks after source edits.
        self.index()
        if self.settings.calibration_path.exists():
            report = json.loads(self.settings.calibration_path.read_text(encoding="utf-8"))
            if report.get("corpus_fingerprint") == self.fingerprint():
                self.calibration = report
                self.strategy = report["selected_strategy"]
                return
        self.calibrate()

    def generate(self, query: str, budget: RequestBudget | None = None) -> RagResult:
        key = f"{self.fingerprint()}:{self.strategy}:{normalize_query(query)}"
        if key in self.cache:
            self.counters.cache_hits += 1
            result = self.cache[key]
            self.cache.move_to_end(key)
            return RagResult(
                result.query,
                result.answer,
                result.sources[:],
                result.supported,
                result.similarity,
                True,
            )
        sources = self.retrieve(query)
        similarity = sources[0].similarity if sources else -1
        threshold = self.calibration["strategies"][self.strategy]["threshold"]
        supported = similarity >= threshold
        selected = (
            [source for source in sources if source.document_id == sources[0].document_id]
            if sources
            else []
        )
        # MOCK_LLM's grounded generator is deliberately extractive: every claim is
        # a verbatim retrieved passage. Never expands to an unretrieved parent doc.
        answer = (
            "\n".join(f"[{source.document_id}] {source.text}" for source in selected)
            if supported
            else FALLBACK
        )
        if budget:
            prompt = json.dumps(
                {
                    "instruction": "Answer only from context or abstain",
                    "query": query,
                    "context": [s.model_dump() for s in sources],
                }
            )
            budget.charge(prompt, answer)
        self.counters.generations += 1
        result = RagResult(query, answer, sources, supported, similarity)
        self.cache[key] = result
        if len(self.cache) > self.settings.cache_capacity:
            self.cache.popitem(last=False)
        return result

    def add_document(self, document: Document) -> dict[str, int]:
        if any(doc.document_id == document.document_id for doc in self.documents):
            raise ValueError(
                "Document ID already exists; replacement requires a reviewed source edit"
            )
        # A serving-time addition is kept in the runtime overlay, never overwrites
        # authored policy files, and invalidates both the cache and calibration.
        self.documents.append(document)
        try:
            counts = self.index()
            self.calibrate()
        except Exception:
            self.documents.pop()
            self.index()
            self.calibrate()
            raise
        overlay = self.settings.runtime / "documents"
        overlay.mkdir(exist_ok=True)
        (overlay / f"{document.document_id}.md").write_text(
            f"# {document.title}\n\n{document.text}\n", encoding="utf-8"
        )
        self.cache.clear()
        return counts
