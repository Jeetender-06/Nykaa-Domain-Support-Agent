"""Executable CrewAI Flow: resolve memory → kickoff crew → review → output guard."""

from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import asyncio
import re

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, ConfigDict, Field

from crew import ORDER_ID, ORDER_INTENT, POLICY_INTENT, CrewContext, run_crew
from governance import BudgetExceeded, RequestBudget
from grounding import BUDGET_REJECTED, supported_answer
from models import CrewResponse, ReviewVerdict
from rag import FALLBACK, RagEngine
from review import review_draft


class SupportState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str = ""
    history: list[dict[str, str]] = Field(default_factory=list)
    resolved_query: str = ""
    record_id: str | None = None
    draft: CrewResponse | None = None
    verdict: ReviewVerdict | None = None
    response: CrewResponse | None = None
    guardrails: list[str] = Field(default_factory=list)
    review_messages: list[dict] = Field(default_factory=list)


class SupportFlow(Flow[SupportState]):
    def __init__(self, rag: RagEngine, budget: RequestBudget):
        super().__init__(tracing=False, suppress_flow_events=True)
        self.rag, self.budget = rag, budget
        self.context: CrewContext | None = None

    @start()
    def resolve_memory(self) -> None:
        query = self.state.query
        explicit = ORDER_ID.findall(query)
        ambiguous = len({value.upper() for value in explicit}) > 1
        wants_order = bool(explicit) or bool(
            re.search(r"\b(?:status|track|tracking)\b|\b(?:is|was) it delayed\b", query, re.I)
        )
        # Only resolve anaphora for order-related follow-ups; never borrow another
        # session's ID, and never silently select one of multiple explicit IDs.
        if ambiguous:
            explicit = []
            wants_order = True
        record_id = explicit[-1].upper() if explicit else None
        if not record_id and not ambiguous and wants_order and ORDER_INTENT.search(query):
            for message in reversed(self.state.history):
                if message["type"] == "human":
                    prior = ORDER_ID.findall(message["content"])
                    if len({value.upper() for value in prior}) > 1:
                        break
                    if prior:
                        record_id = prior[-1].upper()
                        self.state.guardrails.append("session_order_resolved")
                        break
        wants_policy = bool(POLICY_INTENT.search(query)) if wants_order else True
        self.state.record_id = record_id
        self.state.resolved_query = query
        self.context = CrewContext(
            query, self.rag, self.budget, record_id, wants_order, wants_policy
        )

    @listen(resolve_memory)
    async def kickoff_crew(self) -> None:
        try:
            self.state.draft = await asyncio.to_thread(run_crew, self.context)
        except BudgetExceeded:
            self.state.response = CrewResponse(answer=BUDGET_REJECTED, outcome="budget_rejected")
            self.state.guardrails.append("runtime_budget_rejected")

    @listen(kickoff_crew)
    async def review_composer(self) -> None:
        if self.state.response is not None:
            return
        try:
            self.state.verdict, self.state.review_messages = await review_draft(
                self.state.draft, self.budget
            )
        except BudgetExceeded:
            self.state.response = CrewResponse(answer=BUDGET_REJECTED, outcome="budget_rejected")
            self.state.guardrails.append("runtime_budget_rejected")

    @listen(review_composer)
    def output_guard(self) -> CrewResponse:
        if self.state.response is not None:
            return self.state.response
        draft, verdict = self.state.draft, self.state.verdict
        answer = verdict.final_answer
        # Similarity is the answerability signal. Citation validation separately
        # prevents an editor or composer from adding unsupported factual claims.
        if self.context.retrieval and not self.context.retrieval.supported:
            self.state.guardrails.append("groundedness_refusal")
        if not supported_answer(answer, draft.sources, draft.order):
            answer = FALLBACK
            self.state.guardrails.append("unsupported_output_refused")
        self.state.response = CrewResponse.model_validate({**draft.model_dump(), "answer": answer})
        if answer == FALLBACK:
            self.state.response.outcome = "fallback"
        return self.state.response
