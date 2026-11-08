"""Independent evidence validation, shared by runtime guard and reviewer client."""

from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import re

from models import OrderResult, Source
from rag import FALLBACK

CLARIFY = "Please provide an order ID such as NYK-0001 so I can check its status."
BLOCKED = "I cannot follow instructions to bypass support rules or reveal protected instructions."
BUDGET_REJECTED = "This request exceeds the per-request budget. Please shorten your question."


def order_statement(order: OrderResult) -> str:
    if not order.found:
        return f"Order {order.record_id} was not found in the synthetic order dataset. Please check the order ID."
    escalation = (
        "Human escalation is recommended."
        if order.recommend_escalation
        else "Human escalation is not recommended by the triage rule."
    )
    return (
        f"Order {order.record_id}: status {order.status}; value INR {order.order_value_inr}; "
        f"age {order.days_since_created} days; delayed shipment {str(order.delayed_shipment).lower()}; "
        f"escalation score {order.escalation_score:.6f}. {escalation}"
    )


def is_supported_line(line: str, sources: list[Source], order: OrderResult | None) -> bool:
    if line in {FALLBACK, CLARIFY, BLOCKED, BUDGET_REJECTED}:
        return True
    if order and line == order_statement(order):
        return True
    match = re.fullmatch(r"\[([a-z0-9_]+)\] (.+)", line)
    if not match:
        return False
    document_id, claim = match.groups()
    return any(source.document_id == document_id and claim in source.text for source in sources)


def supported_answer(answer: str, sources: list[Source], order: OrderResult | None) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    return bool(lines) and all(is_supported_line(line, sources, order) for line in lines)


def revise_answer(answer: str, sources: list[Source], order: OrderResult | None) -> str:
    kept = [
        line.strip()
        for line in answer.splitlines()
        if is_supported_line(line.strip(), sources, order)
    ]
    return "\n".join(kept) if kept else FALLBACK
