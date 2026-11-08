# Nykaa Domain Support Agent (CrewAI) — Final Capstone

**Track: E-commerce & Retail (Nykaa).** A synthetic customer-support project that implements the 16 capstone tasks, including retrieval from 12 authored fictional policies, order lookup from 60 generated orders, CrewAI agents, LangChain memory, an AutoGen reviewer team, guardrails, and a FastAPI backend. **No real Nykaa services or customer data are accessed.**

`MOCK_LLM` is enabled by default. After installing Python dependencies and downloading the sentence-embedding model once, all model inference uses local files; API keys are not required. `bootstrap.py` disables tracing and telemetry **before** framework modules load.

## Quick start (Ola-style: `pip` + separate Python scripts)

**Required:** Python **3.11** (use `py -3.11` on Windows if the normal `python` resolves to Python 3.15). Python 3.11 was used for the earlier modular version; this reorganized archive is **not yet end-to-end verified on a clean machine**. You need internet **only for installation and downloading the embedding model**.

### Windows PowerShell

Open the **extracted project folder** in VS Code, then run each command in the terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe calibrate.py
.\.venv\Scripts\python.exe diagnose.py
.\.venv\Scripts\python.exe part1_demo.py
.\.venv\Scripts\python.exe part2_demo.py
.\.venv\Scripts\python.exe part3_demo.py
.\.venv\Scripts\python.exe part4_demo.py
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/verify_offline.py --evidence
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Use `python -m venv .venv` if Python 3.11 is already your default and `py` is unavailable. Using the virtual environment's Python directly avoids PowerShell execution-policy issues. The final command starts the API and stays running; open <http://127.0.0.1:8000/docs> in a browser. Stop it with Ctrl+C.

### Linux / macOS

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_model.py
.venv/bin/python calibrate.py
.venv/bin/python part1_demo.py
.venv/bin/python part2_demo.py
.venv/bin/python part3_demo.py
.venv/bin/python part4_demo.py
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/verify_offline.py --evidence
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### Commands and files

| Command | What it runs |
|---|---|
| `python dataset.py` | Regenerate/validate the synthetic order dataset |
| `python calibrate.py` | Build Chroma indexes and calibrate retrieval thresholds |
| `python diagnose.py` | Report model/data/dependency availability |
| `python part1_demo.py` | Dataset, embedding, retrieval comparison |
| `python part2_demo.py` | CrewAI, tools, LangChain memory, guards, structured output, review/caching checks |
| `python part3_demo.py` | HTTP/WebSocket demonstrations and 15-case evaluation |
| `python part4_demo.py` | Complete Part 1–4 evidence regeneration, including AutoGen governance and cache |
| `python evaluate.py` | Run the evaluation independently |
| `python -m uvicorn app:app --port 8000` | Start FastAPI and Swagger UI |
| `python -m pytest tests -q` | Run automated tests |
| `python scripts/verify_offline.py --evidence` | Repeat tests/evidence with external network blocked |
| `python cli.py chat` | Interactive question-and-answer terminal |

The module code is deliberately separated; **there is no large single-file implementation**. The `part4_demo.py` entry point runs the *full* evidence generator, so the Part 1–3 checks are repeated there to test the complete pipeline.

### Troubleshooting

| Problem | Fix |
|---|---|
| `py -3.11` fails | Install Python 3.11, or use `python` if your active interpreter is already 3.11. |
| `No module named crewai` | Re-run `python -m pip install -r requirements.txt` with your venv interpreter. |
| `Local embedding model missing` | Run `python scripts/download_model.py` and verify `models/all-MiniLM-L6-v2/config.json` exists. |
| PowerShell activation is blocked | Don't activate; use `.\.venv\Scripts\python.exe` directly as shown above. |
| `/docs` does not open | Keep the uvicorn terminal running; use <http://127.0.0.1:8000/docs>. |
| An evaluation fails before calibration | Run `python calibrate.py` first. |
| `401/403` on `/add-document` | The optional administration endpoint needs an admin token; `/ask` does not. |
| LangChain/CrewAI deprecation warnings | Warnings alone do not fail the tests. |

## Repository structure

```text
nykaa-support-agent/
├── README.md                 # this guide, design choices and measured evidence
├── requirements.txt          # pip dependencies pinned to previous lockfile versions
├── bootstrap.py              # offline environment + telemetry configuration
├── config.py                 # runtime configuration and local paths
├── dataset.py                # order generator and validator
├── chunking.py               # fixed and sentence chunking
├── rag.py                    # local embedding + Chroma retrieval, thresholds, cache
├── crew.py                   # three CrewAI agents and tool integration
├── flow.py                   # retrieval/lookup/composer orchestration and review
├── service.py                # support service, sessions, request path
├── memory.py                 # LangChain session memory
├── models.py                 # Pydantic schemas
├── governance.py             # access controls, escalation, request budget
├── guardrails.py             # input privacy, injection checks
├── grounding.py              # output checking
├── review.py                 # AutoGen reviewer team
├── app.py                    # FastAPI HTTP endpoints and WebSocket
├── audit.py                  # JSONL trace logs
├── evaluation.py             # retrieval quality and rubric scoring
├── demo.py                   # shared evidence functions
├── cli.py                    # chat / prepare / demo / evaluate / serve
├── calibrate.py              # start indexing + threshold calibration
├── diagnose.py               # local health check
├── part1_demo.py             # retrieval-focused demonstration
├── part2_demo.py             # crew, tools and safety demonstration
├── part3_demo.py             # API and evaluation demonstration
├── part4_demo.py             # complete integrated governance/review demonstration
├── evaluate.py               # standalone evaluation entry point
├── run_all.py                # optional sequential run
├── scripts/                  # model download and offline verification
├── knowledge_base/           # 12 fictional policy documents
├── data/                     # 60 orders and 15 evaluation queries
├── tests/                    # regression and integration tests
├── reports/                  # saved calibration, evaluation and retrieval results
├── transcripts/              # saved execution evidence
└── docs/                     # requirements, governance, references
```

## Task → file map

| Task | Files | Evidence |
|---|---|---|
| 1. Dataset | `dataset.py` | `data/orders.json`, `reports/dataset_summary.json` |
| 2. Knowledge base | `knowledge_base/*.md` | All 12 fictional policy files |
| 3. Chunking/index | `chunking.py`, `rag.py` | Both Chroma indexes created by `calibrate.py` |
| 4. Retrieval/calibration | `rag.py`, `calibrate.py` | `reports/calibration.json` and grounded examples |
| 5. Precision/recall | `evaluation.py`, `part1_demo.py` | `reports/retrieval_comparison.md` |
| 6. Order status/score | `dataset.py`, `governance.py` | `transcripts/crew_tools.json` |
| 7. Three-agent CrewAI | `crew.py`, `flow.py` | `transcripts/crew_tools.json` |
| 8. Session memory | `memory.py`, `service.py` | `transcripts/memory.json` |
| 9. Structured output | `models.py`, `crew.py` | JSON response transcripts |
| 10. Guardrails | `guardrails.py`, `grounding.py` | `transcripts/guardrails_and_budget.json` |
| 11. API and WebSocket | `app.py` | `transcripts/api.json` |
| 12. Audit logs | `audit.py`, `app.py` | Masked sample API logs |
| 13. Evaluation | `evaluation.py`, `evaluate.py` | `reports/evaluation.json` |
| 14. AutoGen independent review | `review.py` | `transcripts/autogen_reviews.json` |
| 15. Governance | `governance.py`, `crew.py` | `docs/GOVERNANCE.md` |
| 16. Cache | `rag.py` | `transcripts/cache.json` |

---

## Part 1 — Dataset, policies, and retrieval (Tasks 1–5)

### Task 1: Dataset decisions

Running `python dataset.py` regenerates and validates **60 orders** with `SEED = 2026`; `python dataset.py --export data/orders.json` writes the same records. Each row has the specified `record_id`, `category`, `status`, `order_value_inr`, `days_since_created`, and `delayed_shipment` fields.

- Category sampling weights (not final counts): Apparel **20**, Electronics **15**, Home **15**, Footwear **20**, Beauty **30**.
- Status weights: Placed **15**, Shipped **25**, Delivered **40**, Returned **10**, Refunded **10**.
- Order value: uniformly sampled integer from **₹299–₹14,999**, representing low-value cosmetics through mid-priced electronics; age: uniform integer **0–30 days**.
- Historical delay flag: independent **20%** probability, retained even if an order has been delivered or refunded.

The saved sample has Apparel **8**, Electronics **6**, Home **14**, Footwear **12**, Beauty **20**; Placed **13**, Shipped **13**, Delivered **24**, Returned **5**, Refunded **5**. **11/60 (18.33%)** have `delayed_shipment=true`. The validator checks the required ranges and the minimum category/status counts.

### Task 2: Knowledge base

`knowledge_base/` contains **12 authored, fictional policies**, with four sentences in each: returns, COD refunds, delivery SLAs, reverse pickup, warranties, cancellation, loyalty points, payment retry, size exchange, damage claims, international shipping, and support escalation. These are demonstration rules, **not verified real Nykaa policies**.

### Task 3: Chunking and ChromaDB

`chunking.py` implements two strategies:

- **Fixed:** 360-character chunks with an 80-character overlap.
- **Sentence:** two sentences per chunk, preserving sentence boundaries.

Each policy produces two chunks with the supplied text: **24 fixed chunks and 24 sentence chunks** in separate cosine-distance ChromaDB collections (`nykaa_fixed`, `nykaa_sentence`). The local `sentence-transformers/all-MiniLM-L6-v2` model creates embeddings; input to embedding is `"<document title>: <chunk text>"`. Chroma uses `upsert`, and the code deletes stale chunk IDs on reindexing. The model revision is pinned in `scripts/download_model.py`.

### Task 4: Empirical similarity threshold

The implementation measures top-1 cosine similarity on **6 in-scope** and **3 out-of-scope** questions. For each index:

`threshold = (minimum in-scope similarity + maximum out-of-scope similarity) / 2`

| Chunking | Minimum in-scope | Maximum out-of-scope | Chosen threshold |
|---|---:|---:|---:|
| Fixed | 0.65352970 | 0.10611784 | **0.37982377** |
| Sentence | 0.66729444 | 0.11165541 | **0.38947493** |

The actual questions and scores are in `reports/calibration.json`. The output uses retrieved passages with `[document_id]` citations, not fabricated policy text. Out-of-scope questions below the threshold receive an explicit knowledge-base fallback. `reports/grounded_examples.json` records the examples generated by the previous run.

### Task 5: Compare retrieval quality

The same **12 policy questions** are evaluated against each index, using the top **3 chunks**, mapped back to distinct parent document IDs. Precision and recall use `TP / retrieved` and `TP / relevant`; the full per-question arithmetic is in `reports/retrieval_comparison.md`.

| Strategy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|
| Fixed | **0.500** | **1.000** | **0.667** |
| Sentence | **0.500** | **1.000** | **0.667** |

**Selected: sentence-based.** Since scores tie on this small set, the implementation breaks the tie in favor of complete sentences. These queries overlap the development/evaluation examples, so the result should not be treated as an independent benchmark. Retrieving top-3 also includes unrelated neighbors; precision is a known limitation.

---

## Part 2 — CrewAI, order tool, memory, and guards (Tasks 6–10)

### Task 6: Order lookup and escalation

The read-only tool `check_order_status(record_id)` returns an order's real fields **from the synthetic dataset**, its calculated score, and whether to recommend a human escalation. Unknown IDs return a `found=false` result.

`escalation_score = round(0.65 × delayed_shipment + 0.35 × days_since_created / 30, 6)`

Escalation is recommended only if the score is at least **0.93** **and** the status is `Placed` or `Shipped`. The threshold uses the dataset's **80th-percentile age (24 days)** in the score expression for delayed orders; it is *not* a claim that 0.93 is the 80th percentile of all escalation scores. In the checked-in dataset, this rule flags **2 of 60** orders. Resolved states never trigger a new escalation recommendation.

### Task 7: Actual CrewAI execution

`crew.py` builds a sequential `Crew` with a Retrieval Agent (RAG tool), Lookup Agent (order tool), and Response Composer (no tools); the crew is executed using `.kickoff()`. The `MockCrewLLM` subclass of CrewAI `BaseLLM` emits native function calls from the tools' declared argument schemas and uses deterministic outputs. `flow.py` coordinates the crew with the post-generation checks.

`transcripts/crew_tools.json` records separate policy-only, order-only, and combined queries and which tools were actually invoked. This is a mock **LLM**, not a pretend CrewAI class; the framework runs the tasks and tools.

### Tasks 8–9: Session memory and structured response

LangChain `InMemoryChatMessageHistory` and `RunnableWithMessageHistory` keep context for follow-ups within one running process. `transcripts/memory.json` shows a follow-up finding `NYK-0001`, the same wording in a new session failing to recover the ID, and the context disappearing after reset. The composer declares a Pydantic `CrewResponse`, and the service/API validate their own Pydantic models.

### Task 10: Guardrails

Before model or history processing, the input is normalized and fixed-format Indian phone numbers, complete card numbers, and context-labeled card-last-four values are masked. Explicit prompt-injection phrases are blocked. After review, the output is checked against retrieved source passages and order evidence; unsupported text is removed or refused. `transcripts/guardrails_and_budget.json` contains examples of masking, injection rejection, out-of-scope refusal, a forged shipping claim, and an oversized-request rejection.

These rules cover the **demonstrated patterns**, not every form of PII or adversarial prompt. Use fabricated inputs only.

---

## Part 3 — API, logging, and evaluation (Tasks 11–13)

### Tasks 11–12: FastAPI and JSONL logging

FastAPI exposes the following local routes:

| Route | Use |
|---|---|
| `POST /ask` | Ask the policy/order support agent |
| `POST /add-document` | Optional authorized insertion of a new fictional policy |
| `GET /health` | Health and current index strategy |
| `DELETE /sessions/{session_id}` | Clear one in-process conversation |
| `WS /ws/{session_id}` | Multi-message WebSocket conversation |

FastAPI automatically supplies **Swagger UI** at <http://127.0.0.1:8000/docs> for the HTTP routes. The `openapi.json` link is the OpenAPI specification, not an OpenAI watermark; WebSocket testing requires a WebSocket client or automated tests. No separate frontend is needed.

Try `POST /ask` in Swagger using:

```json
{
  "query": "What is the return policy for beauty products?",
  "session_id": "demo-1"
}
```

The optional document-ingestion endpoint is disabled until an `NYKAA_ADMIN_TOKEN` is configured; authenticated calls must supply the `X-Admin-Token` header. This endpoint is **not needed** to answer ordinary questions. The application logs masked request metadata as JSON lines under the ignored `.runtime/requests.jsonl`, with a trace ID and elapsed time. HTTP response validation and WebSocket disconnect handling have tests.

### Task 13: Evaluation

The saved `data/evaluation_queries.json` contains **15 cases**: one for each of the 12 policies, one order lookup, one unrelated question, and one prompt-injection attempt. `evaluation.py` executes the full agent and scores **Accuracy, Grounding, Completeness, and Safety** with a deterministic, explicit `MockJudge` rubric (`prompts/judge.txt`). Each query's scores and reasons, and the averages, appear in `reports/evaluation.json` and `.md`.

The saved results are **1.000** on each metric, using a **0–1 scale**. These perfect numbers are **not** real-world model accuracy estimates: the cases were authored for the project and the judge checks reference fields, cited evidence, literal required facts, and simple safety rules. A held-out human-reviewed evaluation would be needed for a broader performance claim.

---

## Part 4 — AutoGen review, governance, and cache (Tasks 14–16)

### Task 14: Two-agent AutoGen review

`review.py` runs a two-turn `RoundRobinGroupChat`: `Policy_Compliance_Reviewer` checks the draft against original passages/order fields, then `Final_Editor` returns a structured Pydantic verdict (`approved`, `final_answer`, `reason`). The mock reviewer clients compute their findings from supplied evidence rather than replaying fixed strings. `transcripts/autogen_reviews.json` shows a clean draft approved unchanged and an altered draft revised to remove an unsupported claim.

### Task 15: Least autonomy and runtime governance

Only the Lookup Agent receives the order tool; the retrieval agent cannot modify orders, and the composer has no tools. Role/schema checks prevent lookup access from being assigned to an unauthorized agent; `transcripts/least_autonomy.json` records a blocked assignment attempt. The assigned risk level is **medium** for synthetic customer-support triage, explained across organizational, data, application, and runtime controls in `docs/GOVERNANCE.md`.

Each request has a shared budget, with preflight rejection before tool/model execution for oversized inputs. The accounting uses UTF-8 byte lengths as **conservative token units** and **simulated** cost (actual mock-provider spend is zero); it is not a real provider billing report.

### Task 16: Cache demonstration

`rag.py` caches normalized policy questions in a bounded in-memory LRU, keyed with the selected chunking strategy and corpus fingerprint. `transcripts/cache.json` shows the first and repeated queries and their search/generation counters; an indexed document update invalidates the cache.

---

## Test evidence and known limitations

- The **previous modular version** completed its normal Windows run with **40 passed** (with deprecation warnings). This edition moves the same application modules to the repository root and switches packaging to `pip`/`requirements.txt`. It includes additional runnable demo wrappers and the targeted offline-check adjustment. **The refactored ZIP has not been fully rerun with CrewAI/AutoGen dependencies in this environment.**
- The checked-in `reports/offline_verification.json` is a **saved earlier result** from a Python 3.11.13 macOS run (`reports/environment.json`). The older checker failed in a later Windows attempt because it blocked local sockets required by asyncio; `scripts/verify_offline.py` now allows loopback and continues blocking external connections. **Run it again on your own Windows machine before claiming a verified Windows offline pass.**
- Document ingestion and index invalidation work in the exercised test. In one earlier Windows run, a new `gift_wrap` document was present in ChromaDB but did not rank top-1 for a related question. The revised ingestion assertion checks indexing in both collections; it does **not** prove the search-ranking issue is resolved.
- The knowledge base and dataset are fictional; the project does not provide real order tracking or real Nykaa policy guidance. In-process memory/cache are lost on restart. No external hosting, authentication for `/ask`, production monitoring, or real payment integration is claimed.

**Before submitting to GitHub:** run `prepare`, `demo`, `evaluate`, `pytest -q`, and `scripts/verify_offline.py --evidence` in the extracted folder. If any command fails, preserve the error and fix the cause rather than changing saved metrics or tests to conceal it. `.gitignore` excludes downloaded model weights, virtual environments, Chroma runtime files, secrets, and logs; the repository contains only code and text/JSON evidence.
