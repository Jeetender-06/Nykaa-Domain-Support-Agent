# Four-layer AI governance policy

## System classification and scope

**Risk level: Medium.** The system assists with retail customer-support questions and synthetic order-status triage, matching the capstone's Medium customer-support category. It does not handle real payment credentials, make credit or hiring decisions, process medical records, or act on accounts; order values are fabricated and read only. A production integration with real identity, financial, medical, or other sensitive data would require a fresh assessment and stronger controls before use.

The business owner for a real rollout would be the retailer's support-policy owner; the technical owner would be the service maintainer. For this capstone those are review roles, not claims that any Nykaa team has approved the project. Policy changes and escalation rules require human review through source control.

## Layer 1 — Business and organizational governance

Purpose is limited to explaining the authored demonstration policies, reporting a requested synthetic order, and recommending human support. The assistant cannot promise real delivery, approve a return, issue a refund, cancel an order, or make a medical diagnosis. The escalation matrix assigns routine inquiries to Level 1, logistics/payments issues to Level 2, and unresolved grievances to Level 3; the machine's score only helps prioritize a human queue.

The release evidence is the task map, measured retrieval comparison, 15-query mock evaluation, negative-control tests, and offline verification report. Changes to KB text, chunking, embedding model, scoring, dependencies, or guards require rerunning the appropriate checks and regenerating evidence. Perfect mock scores are not permission to deploy against real customer data.

## Layer 2 — Data governance

The 60 records are generated from the documented seed and contain only a synthetic ID, category, status, integer value, integer age, and delay flag. No real customer name, delivery address, phone, or card field is stored in the dataset. KB documents are fictional, original capstone policies, stored as reviewable text and indexed with parent-document provenance.

Fixed-format phone and context-labelled card-last-4 values are masked before agent input, history storage, RAG cache lookup, and request logging. Malformed HTTP bodies are represented as `<invalid_json>` instead of being logged verbatim. Raw headers, admin tokens, and exception representations are not logged. The supplied PII examples are fabricated test fixtures. Free-text names and addresses are out of scope for this deterministic masker and must remain fabricated in demonstrations.

Policy additions require the administrator token, 2–5 sentences, a new safe document ID, and clean PII/injection checks. Both collections are reindexed and calibrated; the cache is invalidated. Runtime additions persist under `.runtime/documents`, separate from the reviewed source KB. Logs, model weights, credentials, and runtime files are ignored by Git and excluded from the handoff archive.

Histories are process-local, bounded, and resettable. No durable customer history is promised. A real deployment must define log access, encryption, deletion, and retention according to its actual data classification and obligations.

## Layer 3 — Application governance

**Least autonomy is enforced by tool capability assignment.** Retrieval Agent receives only `rag_lookup`; Lookup Agent receives only `check_order_status`; Response Composer receives no tools. All agents have `allow_delegation=False`. Tool assignment and the mock schema dispatcher reject a tool schema with a `record_id` field unless its role is exactly Lookup Agent, and the lookup tool validates its owner again at execution. `transcripts/least_autonomy.json` records an actual blocked attempt to attach the lookup tool to the Composer. This is an application-level capability boundary; developers with direct Python process access can still call the exported lookup function.

The mock LLM extends CrewAI's own `BaseLLM`. Native tool calls select arguments from declared schemas, independent of tool names. The crew produces a Pydantic `CrewResponse`, and a separate AutoGen team with no tools inspects its draft against the original evidence. The Final Editor returns a registered `StructuredMessage[ReviewVerdict]` after the two-agent round robin. A post-review output guard independently validates the final text again.

Prompt injection uses explicit normalized-text patterns. Retrieval answerability uses a per-collection threshold measured from support/unrelated calibration samples; exact citation matching checks whether output claims occur in original retrieved passages. These methods catch the demonstrated cases but do not establish robust semantic security. A quoted statement can be accurate yet irrelevant; near-domain unsupported questions and obfuscated attacks need broader evaluation.

## Layer 4 — Runtime governance

Each customer request owns one shared budget across grounded generation, all crew model calls, and both review agents. Defaults are 120,000 conservative token units and USD 0.20 simulated cost, with rate assumptions of USD 1/million prompt units and USD 2/million completion units. Token units are UTF-8 bytes rather than an unverified four-characters-per-token guess. Actual cost in `MOCK_LLM` is USD 0; the simulated cost is an enforceable governance exercise, not a provider price claim.

Preflight estimates repeated input/history overhead and rejects oversized input before any model/tool invocation. Each mock client then accounts for its own prompt and completion before returning output, so a request cannot silently cross the budget. A deliberately oversized request is recorded as `budget_rejected`, with zero model calls, in `transcripts/guardrails_and_budget.json`. No external action is taken on budget rejection.

The input body is capped at 256,000 bytes at HTTP ingress; the provided WebSocket server also limits frames. Pydantic caps query length, and the history store is bounded to 128 sessions and 6 prior exchanges at next invocation. The RAG cache is a 256-entry LRU keyed by normalized text, strategy, and corpus/chunking/model fingerprint. An authenticated policy update clears it and recalibrates both collections.

The service uses a single process and lock for shared in-memory state. Each request creates its own crew, flow, model adapters, budget, and review team. The WebSocket handler catches disconnects so other clients continue. Structured JSONL logs record trace IDs and timings once per HTTP request or incoming WebSocket message; API transport logs do not expose raw URLs or input bodies.

`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, tracing switches, and offline model flags are set before dependency imports. `scripts/verify_offline.py` blocks and counts network attempts before importing the application; it fails even when a dependency catches a forbidden connection internally. The checked-in report is the verification evidence, rather than an assumption that mock mode alone prevents telemetry.
