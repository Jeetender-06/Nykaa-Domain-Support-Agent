"""A bounded, actual AutoGen RoundRobinGroupChat with independent mock clients."""

from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import CreateResult, RequestUsage, UserMessage
from autogen_ext.models.replay import ReplayChatCompletionClient

from governance import RequestBudget
from grounding import is_supported_line, revise_answer, supported_answer
from models import CrewResponse, OrderResult, ReviewVerdict, Source


class EvidenceReviewClient(ReplayChatCompletionClient):
    """Dynamic offline client: verdicts are computed from messages, never prerecorded."""

    def __init__(self, role: str, budget: RequestBudget):
        super().__init__(
            [],
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )
        self.role, self.budget = role, budget

    async def create(self, messages, **kwargs) -> CreateResult:
        payload = None
        for message in messages:
            if isinstance(message, UserMessage) and isinstance(message.content, str):
                try:
                    candidate = json.loads(message.content)
                    if "draft" in candidate and "retrieved_context" in candidate:
                        payload = candidate
                        break
                except (ValueError, TypeError):
                    continue
        if payload is None:
            raise ValueError("Review requires explicit draft and original tool evidence")
        sources = [Source.model_validate(source) for source in payload["retrieved_context"]]
        order = OrderResult.model_validate(payload["order"]) if payload["order"] else None
        draft = payload["draft"]
        approved = supported_answer(draft, sources, order)
        if self.role == "policy":
            text = json.dumps(
                {
                    "grounded": approved,
                    "unsupported_lines": [
                        line
                        for line in draft.splitlines()
                        if not is_supported_line(line, sources, order)
                    ],
                    "instruction": "Keep only claims supported by supplied policy passages or order fields.",
                }
            )
        else:
            # The editor receives the policy agent's message through the real team
            # and independently checks the supplied evidence before editing.
            text = ReviewVerdict(
                approved=approved,
                final_answer=draft if approved else revise_answer(draft, sources, order),
                reason="Every claim matches supplied policy or order evidence."
                if approved
                else "Removed unsupported claims from the draft; preserved evidence-backed text.",
            ).model_dump_json()
        prompt = json.dumps([message.model_dump(mode="json") for message in messages])
        self.budget.charge(prompt, text)
        return CreateResult(
            finish_reason="stop",
            content=text,
            usage=RequestUsage(
                prompt_tokens=len(prompt.encode()), completion_tokens=len(text.encode())
            ),
            cached=False,
        )


async def review_draft(
    draft: CrewResponse, budget: RequestBudget
) -> tuple[ReviewVerdict, list[dict]]:
    policy_client = EvidenceReviewClient("policy", budget)
    editor_client = EvidenceReviewClient("editor", budget)
    policy = AssistantAgent(
        "Policy_Compliance_Reviewer",
        model_client=policy_client,
        system_message="Independently verify every draft claim against original retrieved context and read-only order evidence; identify unsupported text.",
    )
    editor = AssistantAgent(
        "Final_Editor",
        model_client=editor_client,
        system_message="Read the policy review and evidence. Approve an entirely supported answer unchanged; otherwise remove unsupported claims. Return the verdict schema.",
        output_content_type=ReviewVerdict,
    )
    team = RoundRobinGroupChat(
        [policy, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    payload = json.dumps(
        {
            "draft": draft.answer,
            "retrieved_context": [source.model_dump() for source in draft.sources],
            "order": draft.order.model_dump() if draft.order else None,
        }
    )
    try:
        result = await team.run(task=payload)
        final = result.messages[-1]
        if not isinstance(final, StructuredMessage) or not isinstance(final.content, ReviewVerdict):
            raise ValueError("AutoGen final editor did not return the required structured verdict")
        transcript = [
            {
                "source": message.source,
                "content": message.content.model_dump()
                if isinstance(message, StructuredMessage)
                else message.content,
            }
            for message in result.messages[1:]
        ]
        return final.content, transcript
    finally:
        await policy_client.close()
        await editor_client.close()
