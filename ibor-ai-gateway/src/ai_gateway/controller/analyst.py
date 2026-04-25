from __future__ import annotations
from uuid import uuid4
from datetime import date
from fastapi import APIRouter, HTTPException, Request
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

    @router.post("/chat", response_model=IborAnswer)
    async def chat(body: ChatRequest, request: Request) -> IborAnswer:
        try:
            # Extract client IP (handle X-Forwarded-For for proxies)
            client_ip = request.client.host if request.client else "unknown"
            if x_forwarded := request.headers.get("X-Forwarded-For"):
                client_ip = x_forwarded.split(",")[0].strip()

            # Auto-capture context (behind-the-scenes)
            portfolio_code = body.portfolio_code or "P-ALPHA"
            analyst_id = client_ip  # Use IP as analyst_id for quota tracking
            session_id = str(uuid4())  # Auto-generate fresh session per request
            market_contents = body.market_contents if body.market_contents is not None else True

            # Check quota BEFORE processing (if quota service available)
            quota_status = None
            if quota_service:
                quota_check = await quota_service.check_quota(client_ip)
                quota_status = QuotaStatus(**quota_check)

                # If quota exceeded, return early with error response
                if quota_status.quota_exceeded:
                    return IborAnswer(
                        question=body.question,
                        as_of=date.today(),
                        summary=f"❌ Daily question limit reached ({quota_status.questions_limit} questions). Come back tomorrow at {quota_status.reset_time}",
                        quota_status=quota_status
                    )

            # Load or create conversation (if service is available)
            conversation_id = None
            if conversation_service:
                conv = await conversation_service.get_or_create_conversation(
                    analyst_id=analyst_id,
                    session_id=session_id,
                    context_type="portfolio",
                    context_id=portfolio_code
                )
                conversation_id = conv["conversation_id"]

                # Save analyst question to conversation
                await conversation_service.save_message(
                    conversation_id=conversation_id,
                    role="analyst",
                    content=body.question
                )

            # Call LLM to generate response (pass market_contents flag)
            response = await agent.chat(
                question=body.question,
                market_contents=market_contents
            )

            # Save AI response to conversation
            if conversation_service and conversation_id:
                await conversation_service.save_message(
                    conversation_id=conversation_id,
                    role="ai",
                    content=response.summary or ""
                )

            # Attach quota status to response (always show remaining quota)
            if quota_status is None and quota_service:
                quota_check = await quota_service.check_quota(client_ip)
                quota_status = QuotaStatus(**quota_check)

            response.quota_status = quota_status
            return response
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
