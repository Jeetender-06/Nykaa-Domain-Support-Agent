from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest

from dataset import CATEGORIES, ORDERS, STATUSES, generate_orders, validate_orders
from chunking import (
    Document,
    fixed_chunks,
    load_documents,
    sentence_chunks,
    sentences,
)
from evaluation import compare_strategies, load_cases
from rag import FALLBACK


def test_dataset_reproducible_and_all_constraints():
    assert generate_orders() == ORDERS == generate_orders()
    summary = validate_orders(ORDERS)
    assert summary["count"] == 60
    assert all(summary["counts_per_category"][value] >= 3 for value in CATEGORIES)
    assert all(summary["counts_per_status"][value] >= 1 for value in STATUSES)
    assert 10 <= summary["delayed_percentage"] <= 30
    assert summary["age_p80_nearest_rank"] == 24


@pytest.mark.parametrize(
    "field,value",
    [
        ("days_since_created", True),
        ("delayed_shipment", 1),
        ("order_value_inr", 10),
        ("record_id", "NYK-0002"),
    ],
)
def test_dataset_rejects_invalid_records(field, value):
    orders = deepcopy(ORDERS)
    orders[0][field] = value
    with pytest.raises(ValueError):
        validate_orders(orders)


def test_chunkers_preserve_content_and_parent_ids(settings):
    docs = load_documents(settings.kb_path)
    assert len(docs) == 12
    for doc in docs:
        assert 2 <= len(sentences(doc.text)) <= 5
        fixed = fixed_chunks(doc)
        assert all(chunk.document_id == doc.document_id for chunk in fixed)
        reconstructed = fixed[0].text + "".join(chunk.text[80:] for chunk in fixed[1:])
        assert reconstructed == doc.text
        assert " ".join(chunk.text for chunk in sentence_chunks(doc)) == doc.text
    with pytest.raises(ValueError):
        fixed_chunks(Document("bad", "bad", "bad"), size=100, overlap=100)


def test_real_embedding_collections_and_measured_calibration(service):
    rag = service.rag
    assert rag.collections["fixed"].name != rag.collections["sentence"].name
    for strategy, collection in rag.collections.items():
        assert collection.count() >= 12
        data = rag.calibration["strategies"][strategy]
        assert data["max_out_of_scope"] < data["threshold"] < data["min_in_scope"]
        assert data["threshold"] == pytest.approx(
            (data["max_out_of_scope"] + data["min_in_scope"]) / 2
        )
        hits = rag.retrieve("When does a cash on delivery refund reach my bank?", strategy)
        assert hits[0].document_id == "cod_refunds"
    dimensions = len(rag.encoder.encode("refund"))
    assert dimensions == 384


def test_grounded_generation_refusal_and_normalized_cache(service):
    rag = service.rag
    before = asdict(rag.counters)
    first = rag.generate("When does a cash on delivery refund arrive?")
    middle = asdict(rag.counters)
    second = rag.generate("  WHEN does a cash on delivery refund arrive?  ")
    after = asdict(rag.counters)
    assert first.supported and "5 to 7 business days" in first.answer
    assert not first.cache_hit and second.cache_hit and second.answer == first.answer
    assert middle["generations"] == before["generations"] + 1
    assert middle["generations"] == after["generations"]
    assert middle["searches"] == after["searches"]
    unrelated = rag.generate("Explain how tectonic plates create volcanoes on Mars.")
    assert not unrelated.supported and unrelated.answer == FALLBACK


def test_both_strategy_metrics_use_deduplicated_documents(service):
    comparison = compare_strategies(service.rag, load_cases(service.settings.root))
    for strategy in ("fixed", "sentence"):
        rows = comparison["strategies"][strategy]["rows"]
        assert len(rows) == 12
        for row in rows:
            assert len(row["retrieved"]) == len(set(row["retrieved"]))
            tp = len(set(row["retrieved"]) & set(row["relevant"]))
            assert row["precision"] == tp / len(row["retrieved"])
            assert row["recall"] == tp / len(row["relevant"])
