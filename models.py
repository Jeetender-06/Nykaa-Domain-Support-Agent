from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    query: str = Field(min_length=1, max_length=180000)
    session_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Source(StrictModel):
    document_id: str
    chunk_id: str
    text: str
    similarity: float


class OrderResult(StrictModel):
    record_id: str
    found: bool
    status: str | None = None
    category: str | None = None
    order_value_inr: int | None = None
    days_since_created: int | None = None
    delayed_shipment: bool | None = None
    escalation_score: float = Field(default=0, ge=0, le=1)
    recommend_escalation: bool = False


class CrewResponse(StrictModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    order: OrderResult | None = None
    needs_escalation: bool = False
    outcome: Literal["answered", "fallback", "clarification", "blocked", "budget_rejected"]


class ReviewVerdict(StrictModel):
    approved: bool
    final_answer: str
    reason: str


class ToolEvent(StrictModel):
    agent: str
    tool: str
    arguments: dict[str, str]
    cache_hit: bool = False


class Usage(StrictModel):
    prompt_tokens_upper_bound: int = 0
    completion_tokens_upper_bound: int = 0
    simulated_cost_usd: float = 0
    actual_cost_usd: float = 0
    model_calls: int = 0


class AskResponse(CrewResponse):
    trace_id: str
    session_id: str
    masked_query: str
    review: ReviewVerdict | None = None
    tool_calls: list[ToolEvent] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    elapsed_ms: float = 0


class AddDocumentRequest(StrictModel):
    document_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,48}$")
    title: str = Field(min_length=3, max_length=160)
    text: str = Field(min_length=40, max_length=4000)


class AddDocumentResponse(StrictModel):
    document_id: str
    indexed_chunks: dict[str, int]
    cache_invalidated: bool
    trace_id: str


class ErrorResponse(StrictModel):
    error: str
    trace_id: str
