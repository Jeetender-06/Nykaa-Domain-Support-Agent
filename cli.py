from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import argparse
import asyncio
import json

from config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Nykaa-domain support capstone")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Index both strategies, calibrate, compare, and select")
    ask = sub.add_parser("ask", help="Run one request through the full Flow")
    ask.add_argument("query")
    ask.add_argument("--session", default="cli")
    sub.add_parser("chat", help="Interactive multi-turn chat in one process")
    sub.add_parser("demo", help="Execute all demonstrations and generate reports/transcripts")
    sub.add_parser("evaluate", help="Evaluate the 15 queries through crew + review")
    serve = sub.add_parser("serve", help="Run the FastAPI HTTP and WebSocket server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "serve":
        import uvicorn

        # Disable uvicorn's access logger: URL paths may contain user-controlled
        # values. The application's sanitized JSONL audit covers every request.
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            access_log=False,
            ws_max_size=256000,
        )
        return
    if args.command == "demo":
        from demo import generate_evidence

        print(json.dumps(generate_evidence(settings), indent=2))
        return
    from evaluation import (
        compare_strategies,
        evaluate,
        load_cases,
        write_evaluation_markdown,
        write_json,
        write_retrieval_markdown,
    )
    from models import AskRequest
    from service import SupportService

    service = SupportService(settings)
    if args.command == "prepare":
        report = compare_strategies(service.rag, load_cases(settings.root))
        write_json(settings.root / "reports" / "calibration.json", service.rag.calibration)
        write_json(settings.root / "reports" / "retrieval_comparison.json", report)
        write_retrieval_markdown(settings.root / "reports" / "retrieval_comparison.md", report)
        print(
            json.dumps(
                {
                    "collections": {
                        key: value.count() for key, value in service.rag.collections.items()
                    },
                    "selected_strategy": service.rag.strategy,
                    "thresholds": {
                        key: value["threshold"]
                        for key, value in service.rag.calibration["strategies"].items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "ask":
        response = asyncio.run(service.ask(AskRequest(query=args.query, session_id=args.session)))
        service.audit.write(
            event="cli_request",
            trace_id=response.trace_id,
            query=response.masked_query,
            outcome=response.outcome,
            elapsed_ms=response.elapsed_ms,
        )
        print(response.model_dump_json(indent=2))
    elif args.command == "evaluate":
        report = asyncio.run(evaluate(service))
        write_json(settings.root / "reports" / "evaluation.json", report)
        write_evaluation_markdown(settings.root / "reports" / "evaluation.md", report)
        print(json.dumps(report["averages"], indent=2))
    elif args.command == "chat":

        async def chat():
            print("Nykaa capstone · fictional policies · MOCK_LLM. Enter /reset or /quit.")
            while True:
                try:
                    query = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if query == "/quit":
                    break
                if query == "/reset":
                    service.memory.clear("chat")
                    print("Conversation cleared.")
                elif query:
                    response = await service.ask(AskRequest(query=query, session_id="chat"))
                    service.audit.write(
                        event="cli_request",
                        trace_id=response.trace_id,
                        query=response.masked_query,
                        outcome=response.outcome,
                        elapsed_ms=response.elapsed_ms,
                    )
                    print(response.answer)

        asyncio.run(chat())


if __name__ == "__main__":
    main()
