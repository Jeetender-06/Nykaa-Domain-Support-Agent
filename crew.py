"""A real three-agent CrewAI crew driven by a deterministic BaseLLM adapter."""

from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import json
import re
from dataclasses import dataclass, field
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from governance import (
    COMPOSER_ROLE,
    LOOKUP_ROLE,
    RETRIEVAL_ROLE,
    RequestBudget,
    check_order_status,
    enforce_tool_access,
)
from grounding import CLARIFY, order_statement
from models import CrewResponse, OrderResult, ToolEvent
from rag import FALLBACK, RagEngine, RagResult

ORDER_ID = re.compile(r"\bNYK-\d{4}\b", re.I)
ORDER_INTENT = re.compile(r"\b(?:order|shipment|status|tracking|it|its)\b", re.I)
POLICY_INTENT = re.compile(
    r"\b(?:polic\w*|return\w*|refund\w*|deliver\w*|pickup|warranty|cancel\w*|loyalty|"
    r"points|payment|retry|exchange|damag\w*|international|escalat\w*|shipping)\b",
    re.I,
)


@dataclass
class CrewContext:
    query: str
    rag: RagEngine
    budget: RequestBudget
    record_id: str | None = None
    wants_order: bool = False
    wants_policy: bool = True
    retrieval: RagResult | None = None
    order: OrderResult | None = None
    events: list[ToolEvent] = field(default_factory=list)


class RagArgs(BaseModel):
    query: str = Field(description="The masked customer policy question")


class LookupArgs(BaseModel):
    record_id: str = Field(description="Synthetic order reference, for example NYK-0001")


class RagTool(BaseTool):
    # Deliberately includes 'lookup': dispatch must use the schema, not this name.
    name: str = "rag_lookup"
    description: str = "Retrieve authored support policy passages and answer only from them."
    args_schema: type[BaseModel] = RagArgs
    _context: CrewContext = PrivateAttr()

    def __init__(self, context: CrewContext):
        super().__init__()
        self._context = context

    def _run(self, query: str) -> str:
        ctx = self._context
        ctx.retrieval = ctx.rag.generate(query, ctx.budget)
        ctx.events.append(
            ToolEvent(
                agent=RETRIEVAL_ROLE,
                tool=self.name,
                arguments={"query": query},
                cache_hit=ctx.retrieval.cache_hit,
            )
        )
        return json.dumps(ctx.retrieval.as_dict())


class LookupTool(BaseTool):
    name: str = "check_order_status"
    description: str = (
        "Read a synthetic order status and deterministic escalation score. Read only."
    )
    args_schema: type[BaseModel] = LookupArgs
    _context: CrewContext = PrivateAttr()
    _owner: str = PrivateAttr()

    def __init__(self, context: CrewContext, owner: str = LOOKUP_ROLE):
        enforce_tool_access(owner, set(LookupArgs.model_fields))
        super().__init__()
        self._context, self._owner = context, owner

    def _run(self, record_id: str) -> str:
        enforce_tool_access(self._owner, set(self.args_schema.model_fields))
        self._context.order = OrderResult.model_validate(check_order_status(record_id))
        self._context.events.append(
            ToolEvent(agent=self._owner, tool=self.name, arguments={"record_id": record_id})
        )
        return self._context.order.model_dump_json()


def assigned_tools(role: str, tools: list[BaseTool]) -> list[BaseTool]:
    for tool in tools:
        enforce_tool_access(role, set(tool.args_schema.model_fields))
    return tools


def compose(ctx: CrewContext) -> CrewResponse:
    answers = []
    if ctx.wants_order:
        answers.append(order_statement(ctx.order) if ctx.order else CLARIFY)
    if ctx.wants_policy and ctx.retrieval:
        answers.append(ctx.retrieval.answer)
    answer = "\n".join(answers) or FALLBACK
    if ctx.wants_order and not ctx.record_id and not (ctx.retrieval and ctx.retrieval.supported):
        outcome = "clarification"
    elif ctx.order and ctx.order.found or ctx.retrieval and ctx.retrieval.supported:
        outcome = "answered"
    else:
        outcome = "fallback"
    return CrewResponse(
        answer=answer,
        sources=ctx.retrieval.sources if ctx.retrieval else [],
        order=ctx.order,
        needs_escalation=bool(ctx.order and ctx.order.recommend_escalation),
        outcome=outcome,
    )


class MockCrewLLM(BaseLLM):
    """Uses CrewAI's native tool protocol; never scans ReAct prompt examples.

    Agent-local request state determines which declared schema to invoke. CrewAI
    executes the returned tool call, and subsequent calls use its actual result.
    """

    _context: CrewContext = PrivateAttr()
    _role: str = PrivateAttr()
    _tool_requested: bool = PrivateAttr(default=False)

    def __init__(self, context: CrewContext, role: str):
        super().__init__(
            model="mock/nykaa-deterministic", temperature=0, context_window_size=200000
        )
        self._context, self._role = context, role

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        return len(json.dumps({"messages": messages, "tools": tools}, default=str).encode())

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs) -> Any:
        ctx = self._context
        response: Any = None
        if tools and not self._tool_requested:
            for tool in tools:
                function = tool.get("function", tool)
                fields = set(function.get("parameters", {}).get("properties", {}))
                enforce_tool_access(self._role, fields)
                arguments = None
                if "query" in fields and ctx.wants_policy:
                    arguments = {"query": ctx.query}
                elif "record_id" in fields and ctx.wants_order and ctx.record_id:
                    arguments = {"record_id": ctx.record_id}
                if arguments is not None:
                    response = [
                        {
                            "id": f"call_{len(ctx.events) + 1}",
                            "type": "function",
                            "function": {
                                "name": function["name"],
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                    self._tool_requested = True
                    break
        if response is None:
            if self._role == COMPOSER_ROLE:
                response = "Final Answer: " + compose(ctx).model_dump_json()
            elif self._role == RETRIEVAL_ROLE:
                response = (
                    json.dumps(ctx.retrieval.as_dict())
                    if ctx.retrieval
                    else '{"skipped":"no policy question"}'
                )
            else:
                response = (
                    ctx.order.model_dump_json()
                    if ctx.order
                    else '{"skipped":"no order ID available"}'
                )
        ctx.budget.charge(
            json.dumps(
                {
                    "messages": messages,
                    "tools": tools,
                    "response_schema": kwargs.get("response_model"),
                },
                default=str,
            ),
            json.dumps(response, default=str),
        )
        return response


def run_crew(context: CrewContext) -> CrewResponse:
    retrieval = Agent(
        role=RETRIEVAL_ROLE,
        goal="Retrieve support policy evidence with rag_lookup.",
        backstory="Policy researcher with access only to the authored knowledge base.",
        tools=assigned_tools(RETRIEVAL_ROLE, [RagTool(context)]),
        llm=MockCrewLLM(context, RETRIEVAL_ROLE),
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
    lookup = Agent(
        role=LOOKUP_ROLE,
        goal="Read the requested synthetic order record when an ID exists.",
        backstory="Read-only order specialist, the sole agent allowed to look up orders.",
        tools=assigned_tools(LOOKUP_ROLE, [LookupTool(context)]),
        llm=MockCrewLLM(context, LOOKUP_ROLE),
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
    composer = Agent(
        role=COMPOSER_ROLE,
        goal="Compose a structured answer from supplied tool evidence only.",
        backstory="Support writer who has no tools and cannot modify orders.",
        tools=[],
        llm=MockCrewLLM(context, COMPOSER_ROLE),
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )
    research_task = Task(
        description="Retrieve policy evidence if needed for this masked question: {query}",
        expected_output="Retrieved passages, citations and grounded policy answer or a skipped result.",
        agent=retrieval,
    )
    lookup_task = Task(
        description="Look up an order only if an ID is available for: {query}",
        expected_output="The exact order tool result or an explicit skipped result.",
        agent=lookup,
    )
    compose_task = Task(
        description="Combine the evidence into the declared CrewResponse schema. Question: {query}",
        expected_output="Valid CrewResponse JSON with citations and no unsupported claims.",
        agent=composer,
        context=[research_task, lookup_task],
        output_pydantic=CrewResponse,
    )
    crew = Crew(
        agents=[retrieval, lookup, composer],
        tasks=[research_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
        tracing=False,
    )
    result = crew.kickoff(inputs={"query": context.query})
    if result.pydantic is None:
        return CrewResponse.model_validate_json(result.raw.removeprefix("Final Answer: "))
    return CrewResponse.model_validate(result.pydantic.model_dump())
