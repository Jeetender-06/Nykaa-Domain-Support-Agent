from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import asyncio
import time
import uuid

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from audit import AuditLog
from config import Settings
from flow import SupportFlow
from governance import BudgetExceeded, RequestBudget
from grounding import BLOCKED, BUDGET_REJECTED
from guardrails import guard_input
from memory import SessionMemory
from models import AskRequest, AskResponse, CrewResponse
from rag import RagEngine


class SupportService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.rag = RagEngine(self.settings)
        self.rag.prepare()
        overlay = self.settings.runtime / "documents"
        if overlay.exists():
            from chunking import load_documents

            self.rag.documents.extend(load_documents(overlay))
            self.rag.prepare()
        self.memory = SessionMemory(self.settings.max_sessions, self.settings.history_turns)
        self.audit = AuditLog(self.settings.runtime / "requests.jsonl")
        # Serialize shared Chroma/cache writes and in-process histories. Each
        # request still owns its own crew, Flow, budget and AutoGen team.
        self.lock = asyncio.Lock()
        self.last_flow: SupportFlow | None = None

    async def ask(self, request: AskRequest, trace_id: str | None = None) -> AskResponse:
        started = time.perf_counter()
        trace_id = trace_id or uuid.uuid4().hex
        guarded = guard_input(request.query)
        budget = RequestBudget(self.settings.max_tokens, self.settings.max_cost_usd)
        events = guarded.events[:]
        verdict = None
        tools = []
        async with self.lock:
            if guarded.blocked:
                result = CrewResponse(answer=BLOCKED, outcome="blocked")
            else:
                history = self.memory.get(request.session_id)
                try:
                    budget.preflight(
                        guarded.text, " ".join(message.content for message in history.messages)
                    )
                except BudgetExceeded:
                    result = CrewResponse(answer=BUDGET_REJECTED, outcome="budget_rejected")
                    events.append("runtime_budget_rejected")
                else:

                    async def invoke(inputs: dict) -> dict:
                        nonlocal verdict, tools
                        flow = SupportFlow(self.rag, budget)
                        self.last_flow = flow
                        response = await flow.kickoff_async(
                            inputs={
                                "query": inputs["query"],
                                "history": [
                                    {"type": message.type, "content": message.content}
                                    for message in inputs["history"]
                                ],
                            }
                        )
                        events.extend(flow.state.guardrails)
                        verdict = flow.state.verdict
                        tools = flow.context.events if flow.context else []
                        return {"result": response, "history_answer": response.answer}

                    pipeline = RunnableWithMessageHistory(
                        RunnableLambda(invoke),
                        self.memory.get,
                        input_messages_key="query",
                        history_messages_key="history",
                        output_messages_key="history_answer",
                    )
                    payload = await pipeline.ainvoke(
                        {"query": guarded.text},
                        config={"configurable": {"session_id": request.session_id}},
                    )
                    result = payload["result"]
        return AskResponse(
            **result.model_dump(),
            trace_id=trace_id,
            session_id=request.session_id,
            masked_query=guarded.text,
            review=verdict,
            tool_calls=tools,
            guardrails=events,
            usage=budget.usage,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
