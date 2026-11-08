from __future__ import annotations

import bootstrap  # offline/telemetry settings before framework imports

import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from audit import AuditLog
from chunking import Document, sentences
from config import Settings
from guardrails import guard_input
from models import (
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
    ErrorResponse,
)
from service import SupportService


def create_app(settings: Settings | None = None, service: SupportService | None = None) -> FastAPI:
    settings = settings or Settings()
    audit = service.audit if service else AuditLog(settings.runtime / "requests.jsonl")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or SupportService(settings)
        yield

    app = FastAPI(
        title="Nykaa Domain Support Agent",
        version="1.0.0",
        description="Synthetic capstone support API. All policies are fictional; MOCK_LLM is offline.",
        lifespan=lifespan,
    )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=str(exc.detail), trace_id=request.state.trace_id
            ).model_dump(),
        )

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next):
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        request.state.trace_id = trace_id
        query = ""
        status = 500
        outcome = "http_error"
        try:
            # Enforce actual body length, including chunked requests, before parsing.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 256000:
                    status = 413
                    return JSONResponse(
                        status_code=413,
                        content=ErrorResponse(
                            error="Request body is too large", trace_id=trace_id
                        ).model_dump(),
                        headers={"X-Trace-ID": trace_id},
                    )
            request._body = bytes(body)
            if body:
                try:
                    payload = json.loads(body)
                    if isinstance(payload, dict):
                        query = str(payload.get("query", payload.get("text", "")))
                    else:
                        query = "<non_object_json>"
                except (ValueError, UnicodeDecodeError):
                    query = "<invalid_json>"
            response = await call_next(request)
            status = response.status_code
            outcome = getattr(request.state, "outcome", "http_completed")
            response.headers["X-Trace-ID"] = trace_id
            return response
        except Exception:
            # No exception repr/body is logged or returned: third-party errors can
            # embed raw customer input. The trace ID supports controlled diagnosis.
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal support service error", trace_id=trace_id
                ).model_dump(),
                headers={"X-Trace-ID": trace_id},
            )
        finally:
            audit.write(
                event="http_request",
                trace_id=trace_id,
                method=request.method,
                path=getattr(request.scope.get("route"), "path", "<unmatched>"),
                query=query,
                status=status,
                outcome=outcome,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )

    @app.get("/health")
    async def health(request: Request) -> dict:
        return {
            "status": "ok",
            "mode": "MOCK_LLM",
            "strategy": request.app.state.service.rag.strategy,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, request: Request) -> AskResponse:
        result = await request.app.state.service.ask(payload, request.state.trace_id)
        request.state.outcome = result.outcome
        return result

    @app.post(
        "/add-document",
        response_model=AddDocumentResponse,
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
    )
    async def add_document(
        payload: AddDocumentRequest,
        request: Request,
        x_admin_token: str | None = Header(default=None),
    ) -> AddDocumentResponse:
        if not settings.admin_token:
            raise HTTPException(
                status_code=403,
                detail="Document ingestion is disabled until NYKAA_ADMIN_TOKEN is configured",
            )
        if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(status_code=401, detail="Invalid admin token")
        guarded = guard_input(payload.title + "\n" + payload.text)
        if guarded.blocked or "pii_masked" in guarded.events:
            raise HTTPException(
                status_code=422,
                detail="Documents must contain no detected injection or fixed-format PII",
            )
        if not 2 <= len(sentences(payload.text)) <= 5:
            raise HTTPException(
                status_code=422, detail="A policy document must contain 2 to 5 sentences"
            )
        current = request.app.state.service
        async with current.lock:
            try:
                counts = current.rag.add_document(
                    Document(payload.document_id, payload.title, payload.text)
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
        return AddDocumentResponse(
            document_id=payload.document_id,
            indexed_chunks=counts,
            cache_invalidated=True,
            trace_id=request.state.trace_id,
        )

    @app.delete("/sessions/{session_id}")
    async def reset_session(session_id: str, request: Request) -> dict:
        current = request.app.state.service
        async with current.lock:
            current.memory.clear(session_id)
        return {"reset": True, "trace_id": request.state.trace_id}

    @app.websocket("/ws/{session_id}")
    async def websocket_chat(websocket: WebSocket, session_id: str):
        await websocket.accept()
        current = websocket.app.state.service
        try:
            while True:
                started = time.perf_counter()
                trace_id = uuid.uuid4().hex
                text = await websocket.receive_text()
                query = "<invalid_websocket_payload>"
                outcome = "validation_error"
                try:
                    if len(text.encode()) > 256000:
                        await websocket.send_json(
                            ErrorResponse(
                                error="WebSocket message is too large", trace_id=trace_id
                            ).model_dump()
                        )
                        continue
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        query = str(payload.get("query", ""))
                    payload = AskRequest.model_validate({**payload, "session_id": session_id})
                    result = await current.ask(payload, trace_id)
                    outcome = result.outcome
                    await websocket.send_json(result.model_dump())
                except (ValueError, TypeError, ValidationError):
                    await websocket.send_json(
                        ErrorResponse(
                            error="Send an object with a nonempty query field", trace_id=trace_id
                        ).model_dump()
                    )
                finally:
                    audit.write(
                        event="websocket_request",
                        trace_id=trace_id,
                        query=query,
                        outcome=outcome,
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    )
        except WebSocketDisconnect:
            # Per-client disconnect ends only this handler; other clients continue.
            return

    return app


app = create_app()
