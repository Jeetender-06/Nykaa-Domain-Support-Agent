from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import re
import unicodedata
from dataclasses import dataclass

PHONE = re.compile(r"(?<!\w)(?:\+?91[\s.-]?)?(?:\d[\s.-]?){9}\d(?!\w)")
CARD_LAST4 = re.compile(
    r"(?i)(\b(?:card[_ -]?(?:last[_ -]?4|ending(?:\s+in)?)|last\s*[- ]?4(?:\s+digits)?|"
    r"card\s+(?:last\s*4|ending(?:\s+in)?))\s*[:=\-]?\s*)(\d{4})(?!\d)"
)
FULL_CARD = re.compile(r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)")
INJECTION_PATTERNS = (
    r"ignore\s+(?:(?:all|the)\s+)?(?:previous|prior|above|system)\s+instructions",
    r"(?:reveal|print|show|expose)\s+(?:the\s+|your\s+)?(?:system\s+prompt|secrets?|api\s+keys?)",
    r"(?:bypass|disable|override)\s+(?:the\s+|all\s+)?(?:guardrails?|safety|polic(?:y|ies)|rules?)",
    r"(?:you\s+are\s+now|act\s+as)\s+(?:an?\s+)?(?:admin|system|developer|unrestricted)",
    r"<\s*/?\s*(?:system|developer)\s*>",
)


@dataclass(frozen=True)
class GuardedInput:
    text: str
    events: list[str]
    blocked: bool


def mask_pii(text: str) -> str:
    text = FULL_CARD.sub("[PAYMENT_CARD]", text)
    text = PHONE.sub("[PHONE]", text)
    return CARD_LAST4.sub(r"\1[CARD_LAST4]", text)


def guard_input(text: str) -> GuardedInput:
    # Normalize compatibility characters before matching; remove zero-width formatting.
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    masked = mask_pii(text)
    events = ["pii_masked"] if masked != text else []
    blocked = any(re.search(pattern, masked, re.I) for pattern in INJECTION_PATTERNS)
    if blocked:
        events.append("prompt_injection_blocked")
    return GuardedInput(masked, events, blocked)
