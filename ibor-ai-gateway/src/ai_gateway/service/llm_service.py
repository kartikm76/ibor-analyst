from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from ai_gateway.model.schemas import IborAnswer
from ai_gateway.service.ibor_service import IborService
from ai_gateway.service.instrument_resolver import InstrumentResolver
from ai_gateway.service.market_tools import MarketTools

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """\
You are analyzing a portfolio manager's question to determine what data to fetch.
Today's date: {today}
Market context enabled: {market_contents}

Available IBOR tools:
- positions: portfolio holdings as-of a date         (portfolioCode, asOf required; optional: accountCode)
- pnl:       P&L delta between two dates             (portfolioCode, asOf, prior required)
- trades:    transaction history for one instrument  (portfolioCode, instrumentCode, asOf required)
- prices:    historical price series                 (instrumentCode, fromDate, toDate required)

CRITICAL: You MUST call positions for ANY portfolio-level analysis or discussion.

Call plan_query with your analysis plan. Rules:
- Call positions if the question mentions: my portfolio, my positions, my holdings, concentration, diversification, allocation, exposure, risk, rebalancing, or any portfolio-level assessment. This is non-negotiable.
- Use pnl for "performance", "P&L", "how did I do" — default prior to yesterday.
- Use trades when the user references a specific instrument by name, ticker, or code (e.g., "AAPL", "Apple", "EQ-AAPL").
- Use prices ONLY when price history is explicitly requested.
- For trades/prices, set instrumentType in args if the user specifies a type: "bond", "equity", "option", "future" → BOND, EQUITY, OPT, FUT.
- explicit_tickers: ONLY tickers the user directly names (e.g. "AAPL", "MSFT"). Do NOT infer.
- needs_macro: {market_contents}. Set to false if market context is disabled.
- Always include "portfolioCode": "P-ALPHA" if no other portfolio is mentioned.
- Extract portfolioCode from the question if present; otherwise default to "P-ALPHA".
"""

_SYNTHESIS_SYSTEM_WITH_MARKET = """\
You are a senior portfolio analyst at an asset management firm.
A portfolio manager has asked a question. You have two categories of data:

IBOR DATA — your firm's investment book of record. These numbers are ground truth.
Use them exactly as given. Never estimate, round, or invent figures.

MARKET CONTEXT — live data from Yahoo Finance (prices, news, earnings, macro indices).
Use this to add intelligence, not to contradict IBOR facts.

Write a 4-8 sentence analyst-grade response that:
1. Answers the question directly using IBOR facts (exact numbers, positions, dates).
2. Layers in market context: current price movement, upcoming events, relevant news.
3. Surfaces the key risk or opportunity the PM should be aware of.
4. Ends with a clear, actionable observation.

Rules:
- IBOR numbers are gospel — never contradict them.
- Market data enriches, it does not override.
- Flowing analyst prose — no bullet points, no headers.
- Be direct and confident; the reader is a professional who manages money.
- If any data is missing or a tool failed, mention it once briefly and continue.
- Today's date: {today}
"""

_SYNTHESIS_SYSTEM_IBOR_ONLY = """\
You are a senior portfolio analyst at an asset management firm.
A portfolio manager has asked a question. You are working with IBOR data ONLY.

IBOR DATA — your firm's investment book of record. These numbers are ground truth.
Use them exactly as given. Never estimate, round, or invent figures.

Write a 4-8 sentence analyst-grade response that:
1. Answers the question directly using IBOR facts (exact numbers, positions, dates).
2. Analyzes the positions, composition, and risk factors visible in IBOR.
3. Surfaces the key risk or opportunity the PM should be aware of.
4. Ends with a clear, actionable observation.

Rules:
- IBOR numbers are gospel — never invent or estimate.
- DO NOT reference external market data, news, or sentiment.
- Focus exclusively on portfolio composition, sizing, and internal risk.
- Flowing analyst prose — no bullet points, no headers.
- Be direct and confident; the reader is a professional who manages money.
- If analysis requires market context not available, mention it briefly then continue.
- Today's date: {today}
"""

_SYNTHESIS_SYSTEM = _SYNTHESIS_SYSTEM_WITH_MARKET  # Default for backward compatibility

_MARKET_ONLY_SYNTHESIS_SYSTEM = """\
You are a senior portfolio analyst at an asset management firm.
A portfolio manager has asked a market intelligence question.
Answer using the market data provided. Be concise and professional.
Flowing analyst prose — no bullet points, no headers. 3-5 sentences.
Today's date: {today}
"""

_OFF_TOPIC_MESSAGE = (
    "I'm an institutional IBOR analyst — I help portfolio managers understand their managed books. "
    "I can't advise on personal investments, retirement accounts, or personal financial planning. "
    "Try asking about your portfolio positions, P&L, trade history, or market context for your holdings."
)

# ---------------------------------------------------------------------------
# Guardrail classifier
# ---------------------------------------------------------------------------

_GUARDRAIL_SYSTEM = """\
You are a question classifier for an institutional IBOR analyst tool used by professional portfolio managers.

Classify the question into exactly one of three categories:

"proceed" — Questions about managing institutional portfolios, or anything ambiguous. Default to this.
  Examples: positions, P&L, trades, risk, allocation, rebalancing, "should I buy TSLA" (assume for the portfolio), sector exposure.

"market_only" — Pure market intelligence with no portfolio context required.
  Examples: "big movers in NASDAQ last week", "where is VIX", "what did the Fed do", "summarize NVDA earnings", "what happened to rates".

"reject" — Explicit personal finance signals only (retirement accounts, personal savings, personal tax advice).
  Examples: "what to put in my Roth IRA", "best ETFs for my 401k", "should I sell my house to invest", "college fund stocks".

When in doubt, classify as "proceed". Rejecting a legitimate analyst question is worse than answering an edge case.
"""

_GUARDRAIL_TOOL: Dict[str, Any] = {
    "name": "classify_question",
    "description": "Classify the question as proceed, market_only, or reject.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["proceed", "market_only", "reject"],
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining the classification.",
            },
        },
        "required": ["category", "reason"],
    },
}

# ---------------------------------------------------------------------------
# Intent tool definition  (Anthropic format: input_schema, no type:function wrapper)
# ---------------------------------------------------------------------------

_PLAN_TOOL: Dict[str, Any] = {
    "name": "plan_query",
    "description": "Output the analysis plan: which IBOR tools to call and what market context to fetch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ibor_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": ["positions", "trades", "pnl", "prices"],
                        },
                        "args": {
                            "type": "object",
                            "description": "For trades/prices, include instrumentType (EQUITY, BOND, OPT, FUT) if the user specified it."
                        },
                    },
                    "required": ["tool", "args"],
                },
            },
            "explicit_tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tickers explicitly named in the user question only.",
            },
            "needs_macro": {
                "type": "boolean",
                "description": "True if macro context (VIX, yields, S&P 500) is relevant.",
            },
        },
        "required": ["ibor_calls", "explicit_tickers", "needs_macro"],
    },
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class LlmService:
    """Two-stage fan-out AI analyst.

    Stage 1  — Parse intent: one LLM call extracts which IBOR tools to call
               and any tickers explicitly named in the question.

    Stage 2a — If tickers were explicit: fire IBOR tools + all market tools
               simultaneously in a single asyncio.gather (true Octopus blast).

    Stage 2b — If no explicit tickers: fetch IBOR data first, extract equity
               tickers from the results, then fan-out all market tools in parallel.

    Final    — One synthesis LLM call that combines everything into
               analyst-grade prose.
    """

    def __init__(
        self,
        anthropic_client: AsyncAnthropic,
        service: IborService,
        market_tools: MarketTools,
        resolver: Optional[InstrumentResolver] = None,
        model: Optional[str] = None,
    ) -> None:
        self._anthropic = anthropic_client
        self._service = service
        self._market = market_tools
        self._resolver = resolver
        self._model = model or "claude-sonnet-4-6"

    async def summarize(self, verbose_text: str) -> dict:
        """Compress verbose analysis into bullet-point summary."""
        try:
            resp = await self._anthropic.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SUMMARIZER_SYSTEM,
                messages=[{"role": "user", "content": f"Summarize this:\n\n{verbose_text}"}],
            )
            text_block = next((b for b in resp.content if b.type == "text"), None)
            result_text = text_block.text if text_block else ""

            # Try to parse JSON; if it fails, fall back to line-by-line bullets
            try:
                result_json = json.loads(result_text)
                return {"summary": result_json.get("summary", [result_text])}
            except json.JSONDecodeError:
                # Fallback: split into sentences
                bullets = [s.strip() for s in result_text.split(".") if s.strip()][:5]
                return {"summary": bullets}
        except Exception as exc:
            log.warning("summarize failed: %s", exc)
            return {"summary": [verbose_text[:200] + "..."]}

    async def _classify_question(self, question: str) -> Dict[str, Any]:
        try:
            resp = await self._anthropic.messages.create(
                model=self._model,
                max_tokens=256,
                system=_GUARDRAIL_SYSTEM,
                messages=[{"role": "user", "content": question}],
                tools=[_GUARDRAIL_TOOL],
                tool_choice={"type": "tool", "name": "classify_question"},
            )
            tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
            result = tool_block.input if tool_block else {"category": "proceed", "reason": "fallback"}
            log.info("guardrail: %s — %s", result.get("category"), result.get("reason"))
            return result
        except Exception as exc:
            log.warning("guardrail classify failed: %s — defaulting to proceed", exc)
            return {"category": "proceed", "reason": "fallback"}

    async def _handle_market_only(self, question: str, today: date) -> IborAnswer:
        # Reuse intent parser to extract any explicit tickers; always fetch macro
        plan = await self._parse_intent(question, today, market_contents=True)
        explicit_tickers = [t.upper() for t in plan.get("explicit_tickers", [])]

        labels, coros = self._build_market_coros(explicit_tickers, needs_macro=True)
        market_raw = list(await asyncio.gather(*coros, return_exceptions=True)) if coros else []
        market_context = _collate_market(labels, market_raw)

        payload = {"question": question, "as_of": str(today), "market_context": market_context}
        resp = await self._anthropic.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_MARKET_ONLY_SYNTHESIS_SYSTEM.format(today=today),
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        text_block = next((b for b in resp.content if b.type == "text"), None)
        summary = text_block.text if text_block else ""
        return IborAnswer(
            question=question,
            as_of=today,
            summary=summary,
            data={"market": market_context},
        )

    async def chat(self, question: str, market_contents: bool = True) -> IborAnswer:
        today = date.today()

        # ── Step 0: guardrail ─────────────────────────────────────────────
        guard = await self._classify_question(question)
        category = guard.get("category", "proceed")

        if category == "reject":
            return IborAnswer(question=question, as_of=today, summary=_OFF_TOPIC_MESSAGE)

        if category == "market_only" and market_contents:
            return await self._handle_market_only(question, today)

        # ── Step 1: intent parse ──────────────────────────────────────────
        plan = await self._parse_intent(question, today, market_contents)
        ibor_calls: List[Dict[str, Any]] = plan.get("ibor_calls", [])
        explicit_tickers: List[str] = [t.upper() for t in plan.get("explicit_tickers", [])]
        needs_macro: bool = plan.get("needs_macro", False) if not market_contents else plan.get("needs_macro", True)

        if not ibor_calls:
            api_error = plan.get("_error")
            gaps = [f"Anthropic API error: {api_error}"] if api_error else [
                "Could not determine what IBOR data to fetch for this question."
            ]
            return IborAnswer(question=question, as_of=today, gaps=gaps)

        ibor_coros = [
            self._dispatch_ibor(c["tool"], c.get("args", {}), today)
            for c in ibor_calls
        ]

        # ── Step 2: fan-out ───────────────────────────────────────────────
        market_context: Dict[str, Any] = {}

        if market_contents:
            # Market data enabled: fetch market context
            if explicit_tickers:
                # True octopus: IBOR + market all at once
                market_labels, market_coros = self._build_market_coros(explicit_tickers, needs_macro)
                all_results = await asyncio.gather(*ibor_coros, *market_coros, return_exceptions=True)
                ibor_results: List[Any] = list(all_results[: len(ibor_coros)])
                market_raw: List[Any] = list(all_results[len(ibor_coros) :])
            else:
                # Two-stage: IBOR → extract tickers → market
                ibor_results = list(await asyncio.gather(*ibor_coros, return_exceptions=True))
                tickers = _extract_equity_tickers(ibor_results)
                market_labels, market_coros = self._build_market_coros(tickers, needs_macro)
                market_raw = (
                    list(await asyncio.gather(*market_coros, return_exceptions=True))
                    if market_coros
                    else []
                )
            market_context = _collate_market(market_labels, market_raw)
        else:
            # Market data disabled: IBOR only
            ibor_results = list(await asyncio.gather(*ibor_coros, return_exceptions=True))

        # ── Step 3: synthesis ─────────────────────────────────────────────
        # Short-circuit if any IBOR dispatch needs a clarification from the user
        for result in ibor_results:
            if isinstance(result, IborAnswer) and result.clarification:
                return IborAnswer(question=question, as_of=today, clarification=result.clarification)

        return await self._synthesize(question, today, ibor_calls, ibor_results, market_context, market_contents)

    # ── Instrument resolution ─────────────────────────────────────────────

    def _resolve_instrument(self, raw_code: str, today, type_hint: Optional[str] = None) -> "str | IborAnswer":
        """Resolve LLM-supplied instrument name/ticker/code to canonical instrument_code.
        Returns the canonical code string, or an IborAnswer with clarification if ambiguous/not found.
        type_hint: instrument type extracted by the intent LLM (EQUITY, BOND, etc.)
        """
        if not raw_code or not self._resolver:
            return raw_code

        result = self._resolver.resolve(raw_code, type_hint=type_hint)

        if result.is_ambiguous or not result.matches:
            return IborAnswer(
                question=raw_code,
                as_of=today,
                clarification=result.clarification,
            )

        return result.matches[0].code

    # ── Intent parsing ────────────────────────────────────────────────────

    async def _parse_intent(self, question: str, today: date, market_contents: bool = True) -> Dict[str, Any]:
        try:
            resp = await self._anthropic.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_INTENT_SYSTEM.format(today=today, market_contents=market_contents),
                messages=[{"role": "user", "content": question}],
                tools=[_PLAN_TOOL],
                tool_choice={"type": "tool", "name": "plan_query"},
            )
            tool_block = next((b for b in resp.content if b.type == "tool_use"), None)
            return tool_block.input if tool_block else {}
        except Exception as exc:
            log.error("intent parse failed [model=%s]: %s: %s", self._model, type(exc).__name__, exc)
            return {"ibor_calls": [], "explicit_tickers": [], "needs_macro": False, "_error": str(exc)}

    # ── IBOR dispatch ─────────────────────────────────────────────────────

    async def _dispatch_ibor(
        self, tool: str, args: Dict[str, Any], today: date
    ) -> IborAnswer:
        def _d(key: str, fallback: date) -> date:
            v = args.get(key)
            return date.fromisoformat(v) if isinstance(v, str) else fallback

        try:
            if tool == "positions":
                return await self._service.positions(
                    portfolio_code=args.get("portfolioCode") or "P-ALPHA",
                    as_of=_d("asOf", today),
                    account_code=args.get("accountCode"),
                    base_currency=args.get("baseCurrency"),
                    source=args.get("source"),
                )
            if tool == "pnl":
                return await self._service.pnl(
                    portfolio_code=args.get("portfolioCode") or "P-ALPHA",
                    as_of=_d("asOf", today),
                    prior=_d("prior", today - timedelta(days=1)),
                    instrument_code=args.get("instrumentCode"),
                )
            if tool == "trades":
                resolved = self._resolve_instrument(
                    args.get("instrumentCode", ""), today,
                    type_hint=args.get("instrumentType"),
                )
                if isinstance(resolved, IborAnswer):
                    return resolved
                return await self._service.trades(
                    portfolio_code=args.get("portfolioCode") or "P-ALPHA",
                    instrument_code=resolved,
                    as_of=_d("asOf", today),
                )
            if tool == "prices":
                resolved = self._resolve_instrument(
                    args.get("instrumentCode", ""), today,
                    type_hint=args.get("instrumentType"),
                )
                if isinstance(resolved, IborAnswer):
                    return resolved
                return await self._service.prices(
                    instrument_code=resolved,
                    from_date=_d("fromDate", today - timedelta(days=30)),
                    to_date=_d("toDate", today),
                    source=args.get("source"),
                    base_currency=args.get("baseCurrency"),
                )
        except Exception as exc:
            log.warning("ibor dispatch failed %s %s: %s", tool, args, exc)
            return IborAnswer(
                question=f"{tool}({args})",
                as_of=today,
                gaps=[f"IBOR tool '{tool}' failed: {exc}"],
            )

        return IborAnswer(
            question="",
            as_of=today,
            gaps=[f"Unknown IBOR tool requested: '{tool}'"],
        )

    # ── Market task builder ───────────────────────────────────────────────

    def _build_market_coros(
        self, tickers: List[str], needs_macro: bool
    ) -> Tuple[List[Tuple[str, Optional[str]]], List]:
        labels: List[Tuple[str, Optional[str]]] = []
        coros: List = []
        for ticker in tickers:
            labels.append(("snapshot", ticker))
            coros.append(self._market.get_market_snapshot(ticker))
            labels.append(("news", ticker))
            coros.append(self._market.get_news(ticker))
            labels.append(("earnings", ticker))
            coros.append(self._market.get_earnings(ticker))
        if needs_macro:
            labels.append(("macro", None))
            coros.append(self._market.get_macro_snapshot())
        return labels, coros

    # ── Synthesis ─────────────────────────────────────────────────────────

    async def _synthesize(
        self,
        question: str,
        today: date,
        ibor_calls: List[Dict],
        ibor_results: List,
        market_context: Dict[str, Any],
        market_contents: bool = True,
    ) -> IborAnswer:
        ibor_data: Dict[str, Any] = {}
        gaps: List[str] = []

        for call, result in zip(ibor_calls, ibor_results):
            tool = call["tool"]
            if isinstance(result, Exception):
                gaps.append(f"{tool} failed: {result}")
            elif isinstance(result, IborAnswer):
                gaps.extend(result.gaps)
                ibor_data[tool] = result.data
            else:
                gaps.append(f"{tool} returned unexpected type")

        payload = {
            "question": question,
            "as_of": str(today),
            "ibor_data": ibor_data,
            "market_context": market_context,
        }

        # Use appropriate synthesis prompt based on market_contents flag
        synthesis_prompt = _SYNTHESIS_SYSTEM_WITH_MARKET if market_contents else _SYNTHESIS_SYSTEM_IBOR_ONLY

        resp = await self._anthropic.messages.create(
            model=self._model,
            max_tokens=2048,
            system=synthesis_prompt.format(today=today),
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        text_block = next((b for b in resp.content if b.type == "text"), None)
        summary = text_block.text if text_block else ""

        first_ok = next(
            (r for r in ibor_results if isinstance(r, IborAnswer) and not r.gaps), None
        )
        return IborAnswer(
            question=question,
            as_of=first_ok.as_of if first_ok else today,
            summary=summary,
            data={"ibor": ibor_data, "market": market_context},
            gaps=gaps,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_equity_tickers(ibor_results: List) -> List[str]:
    """Pull equity tickers from IBOR position data (instrument codes like EQ-AAPL → AAPL)."""
    tickers: set[str] = set()
    for result in ibor_results:
        if isinstance(result, IborAnswer):
            for pos in result.data.get("positions", []):
                code = str(pos.get("instrument", ""))
                # Strip EQ- prefix; ignore bonds/futures/options/fx/index
                if code.startswith("EQ-"):
                    ticker = code[3:]  # e.g. "EQ-AAPL" → "AAPL"
                    if re.match(r"^[A-Z]{1,6}$", ticker):
                        tickers.add(ticker)
    return list(tickers)[:10]  # cap to avoid excessive external API calls


def _collate_market(
    labels: List[Tuple[str, Optional[str]]], results: List
) -> Dict[str, Any]:
    """Reconstruct market results from asyncio.gather into a structured dict."""
    context: Dict[str, Any] = {"by_ticker": {}, "macro": None}
    for (label, ticker), result in zip(labels, results):
        if isinstance(result, Exception):
            log.warning("market task %s/%s failed: %s", label, ticker, result)
            continue
        if label == "macro":
            context["macro"] = result
        else:
            context["by_ticker"].setdefault(ticker, {})[label] = result
    return context


# Summarizer prompt
_SUMMARIZER_SYSTEM = """\
You are a portfolio analyst writing crisp, concise bullet-point summaries.

You are given a verbose narrative about a portfolio.
Compress it into 3-5 bullet points, with one per major position or theme.

Each bullet should be:
- 1 sentence max
- A fact + a key insight (e.g. "SPY 200 sh | $129K | Up 1.2% today")
- No jargon; plain English

Format as a JSON object with key "summary" containing the list of bullets.
"""