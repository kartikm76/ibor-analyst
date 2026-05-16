from __future__ import annotations
import json
from uuid import uuid4, UUID
from datetime import date
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from ai_gateway.config.settings import settings
from ai_gateway.model.schemas import (
    ChatRequest, IborAnswer, PnLRequest, PositionsRequest, PricesRequest, TradesRequest, QuotaStatus,
)
from ai_gateway.service.ibor_service import IborService
from ai_gateway.service.llm_service import LlmService
from ai_gateway.service.conversation_service import ConversationService
from ai_gateway.service.quota_service import QuotaService

def make_analyst_router(
    service: IborService,
    agent: LlmService,
    conversation_service: ConversationService = None,
    quota_service: QuotaService = None
) -> APIRouter:
    router = APIRouter(prefix="/analyst", tags=["Analyst"])

    @router.post("/positions", response_model=IborAnswer)
    async def positions(body: PositionsRequest) -> IborAnswer:
        try:
            return await service.positions(
                portfolio_code=body.portfolio_code,
                as_of=body.as_of,
                base_currency=body.base_currency,
                source=body.source,
                page=body.page,
                size=body.size,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/trades", response_model=IborAnswer)
    async def trades(body: TradesRequest) -> IborAnswer:
        try:
            return await service.trades(
                portfolio_code=body.portfolio_code,
                instrument_code=body.instrument_code,
                as_of=body.as_of,
                page=body.page,
                size=body.size,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/prices", response_model=IborAnswer)
    async def prices(body: PricesRequest) -> IborAnswer:
        try:
            return await service.prices(
                instrument_code=body.instrument_code,
                from_date=body.from_date,
                to_date=body.to_date,
                source=body.source,
                base_currency=body.base_currency,
                page=body.page,
                size=body.size,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/pnl", response_model=IborAnswer)
    async def pnl(body: PnLRequest) -> IborAnswer:
        try:
            return await service.pnl(
                portfolio_code=body.portfolio_code,
                as_of=body.as_of,
                prior=body.prior,
                instrument_code=body.instrument_code,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/quota")
    async def quota(request: Request) -> dict:
        """Return current quota status for the caller without consuming a question.
        Used by the UI on chat-modal open so the 'X of N questions remaining'
        counter shows from the start, not only after the first question."""
        if not quota_service:
            return {"questions_today": 0, "questions_limit": 0, "questions_remaining": 0,
                    "quota_exceeded": False, "reset_time": None}
        client_ip = request.client.host if request.client else "unknown"
        if x_forwarded := request.headers.get("X-Forwarded-For"):
            client_ip = x_forwarded.split(",")[0].strip()
        return await quota_service.check_quota(client_ip)

    @router.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        try:
            if not settings.chat_enabled:
                msg = settings.chat_disabled_message
                async def disabled_stream():
                    yield f"data: {json.dumps({'type': 'text', 'content': msg})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'summary': msg, 'data': {}, 'gaps': []})}\n\n"
                return StreamingResponse(disabled_stream(), media_type="text/event-stream")

            client_ip = request.client.host if request.client else "unknown"
            if x_forwarded := request.headers.get("X-Forwarded-For"):
                client_ip = x_forwarded.split(",")[0].strip()

            portfolio_code = body.portfolio_code or "P-ALPHA"
            analyst_id = client_ip
            session_id = body.session_id or str(uuid4())
            market_contents = body.market_contents if body.market_contents is not None else True
            mode = body.mode if body.mode in ("brief", "detailed") else "detailed"

            # Quota check before streaming starts
            quota_status = None
            if quota_service:
                quota_check = await quota_service.check_quota(client_ip)
                quota_status = QuotaStatus(**quota_check)
                if quota_status.quota_exceeded:
                    msg = f"❌ Daily question limit reached ({quota_status.questions_limit} questions). Come back tomorrow at {quota_status.reset_time}"
                    async def quota_exceeded_stream():
                        yield f"data: {json.dumps({'type': 'text', 'content': msg})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'summary': msg, 'data': {}, 'gaps': [], 'quota_status': quota_status.model_dump(mode='json')})}\n\n"
                    return StreamingResponse(quota_exceeded_stream(), media_type="text/event-stream")

            # Conversation setup
            conversation_id = None
            if conversation_service:
                conv = await conversation_service.get_or_create_conversation(
                    analyst_id=analyst_id,
                    session_id=session_id,
                    context_type="portfolio",
                    context_id=portfolio_code
                )
                conversation_id = conv["conversation_id"]
                await conversation_service.save_message(
                    conversation_id=conversation_id,
                    role="analyst",
                    content=body.question
                )

            # RAG retrieval
            prior_context = []
            if conversation_service:
                prior_context = await conversation_service.search_similar_conversations(
                    query=body.question,
                    context_type="portfolio",
                    context_id=portfolio_code,
                    analyst_id=analyst_id,
                    top_k=settings.rag_top_k,
                    min_similarity=settings.rag_min_similarity,
                )

            # Refresh quota after processing
            if quota_status is None and quota_service:
                quota_check = await quota_service.check_quota(client_ip)
                quota_status = QuotaStatus(**quota_check)

            quota_dict = quota_status.model_dump(mode='json') if quota_status else None
            collected_summary = []

            async def generate():
                async for chunk in agent.chat_stream(
                    question=body.question,
                    market_contents=market_contents,
                    prior_context=prior_context,
                    mode=mode,
                ):
                    if chunk["type"] == "text":
                        collected_summary.append(chunk["content"])
                    elif chunk["type"] == "done":
                        chunk["quota_status"] = quota_dict
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Save AI response after stream completes
                if conversation_service and conversation_id:
                    await conversation_service.save_message(
                        conversation_id=conversation_id,
                        role="ai",
                        content="".join(collected_summary)
                    )

            return StreamingResponse(generate(), media_type="text/event-stream")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/summarize")
    async def summarize(body: dict) -> dict:
        """Compress verbose summary into bullet points per instrument."""
        try:
            verbose_text = body.get("summary", "")
            return await agent.summarize(verbose_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
