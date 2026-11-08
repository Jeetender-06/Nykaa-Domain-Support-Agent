from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

from dataclasses import dataclass, field

from dataset import ORDERS, validate_orders

from models import OrderResult, Usage

LOOKUP_ROLE = "Lookup Agent"
RETRIEVAL_ROLE = "Retrieval Agent"
COMPOSER_ROLE = "Response Composer"
AGE_P80 = validate_orders(ORDERS)["age_p80_nearest_rank"]
ESCALATION_THRESHOLD = round(0.65 + 0.35 * AGE_P80 / 30, 6)


def check_order_status(record_id: str) -> dict:
    row = next((row for row in ORDERS if row["record_id"] == record_id.upper()), None)
    if row is None:
        return OrderResult(record_id=record_id.upper(), found=False).model_dump()
    # Older orders have a larger normalized age/recency signal. Historical delay
    # remains visible in the score, but resolved statuses are never auto-escalated.
    score = round(0.65 * int(row["delayed_shipment"]) + 0.35 * row["days_since_created"] / 30, 6)
    result = OrderResult(
        **row,
        found=True,
        escalation_score=score,
        recommend_escalation=(
            score >= ESCALATION_THRESHOLD and row["status"] in {"Placed", "Shipped"}
        ),
    )
    return result.model_dump()


class AutonomyViolation(PermissionError):
    pass


def enforce_tool_access(role: str, argument_fields: set[str]) -> None:
    if "record_id" in argument_fields and role != LOOKUP_ROLE:
        raise AutonomyViolation("Only Lookup Agent may receive or execute the order lookup tool")


class BudgetExceeded(RuntimeError):
    pass


def token_upper_bound(value: str) -> int:
    # A UTF-8 byte count is a conservative token bound, including non-Latin input.
    # MOCK_LLM has no tokenizer or billing; this is an enforceable simulation.
    return len(value.encode("utf-8"))


@dataclass
class RequestBudget:
    max_tokens: int
    max_cost_usd: float
    usage: Usage = field(default_factory=Usage)

    def preflight(self, query: str, history: str = "") -> None:
        estimate = 6 * token_upper_bound(query + history) + 30000
        self._check(estimate, estimate * 2 / 1_000_000)

    def _check(self, tokens: int, cost: float) -> None:
        used = self.usage.prompt_tokens_upper_bound + self.usage.completion_tokens_upper_bound
        if (
            used + tokens > self.max_tokens
            or self.usage.simulated_cost_usd + cost > self.max_cost_usd
        ):
            raise BudgetExceeded("Request exceeds the configured token or simulated cost budget")

    def charge(self, prompt: str, completion: str) -> None:
        incoming, outgoing = token_upper_bound(prompt), token_upper_bound(completion)
        cost = (incoming + 2 * outgoing) / 1_000_000
        self._check(incoming + outgoing, cost)
        self.usage.prompt_tokens_upper_bound += incoming
        self.usage.completion_tokens_upper_bound += outgoing
        self.usage.simulated_cost_usd = round(self.usage.simulated_cost_usd + cost, 8)
        self.usage.model_calls += 1
