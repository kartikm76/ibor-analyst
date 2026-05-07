"""
Streaming smoke test
====================
Verifies:
  1. anthropic.messages.stream() syntax works and yields text chunks
  2. SSE formatting (data: {...}\\n\\n) is valid JSON on each line
  3. chat_stream() generator produces the expected chunk types in order

Does NOT need PostgreSQL or Spring Boot — mocks IBOR results.

Run:
    cd ibor-ai-gateway
    uv run pytest tests/test_streaming.py -v -s
"""
from __future__ import annotations

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from anthropic import AsyncAnthropic

from ai_gateway.model.schemas import IborAnswer
from ai_gateway.service.llm_service import LlmService

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── Test 1: raw Anthropic streaming API works ─────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_stream_syntax():
    """Verify messages.stream() produces text chunks with correct syntax."""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    chunks = []

    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",   # cheapest model for smoke test
        max_tokens=50,
        system="You are a helpful assistant. Reply in exactly one sentence.",
        messages=[{"role": "user", "content": "Say hello."}],
    ) as stream:
        async for text in stream.text_stream:
            chunks.append(text)

    full = "".join(chunks)
    print(f"\n  Streamed {len(chunks)} chunks: '{full[:80]}'")

    assert len(chunks) > 0, "No chunks received"
    assert len(full) > 0, "Empty response"
    print("  ✅ Anthropic streaming API syntax OK")


# ── Test 2: chat_stream() chunk types ────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_stream_chunk_types():
    """chat_stream() must yield text chunks then exactly one done chunk."""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Minimal mocks — bypass IBOR, market, guard, intent
    mock_ibor_answer = IborAnswer(
        question="test",
        summary="Portfolio has 10 positions.",
        data={"positions": [{"instrument": "EQ-AAPL", "net_qty": 1000}]},
    )

    service = MagicMock()
    market_tools = MagicMock()
    resolver = MagicMock()

    llm = LlmService(
        anthropic_client=client,
        service=service,
        market_tools=market_tools,
        resolver=resolver,
        model="claude-haiku-4-5-20251001",
        llama_guard=None,
    )

    # Patch classify → proceed, intent → positions call, dispatch → mock answer
    with patch.object(llm, '_classify_question', new=AsyncMock(return_value={"category": "proceed"})), \
         patch.object(llm, '_parse_intent', new=AsyncMock(return_value={
             "ibor_calls": [{"tool": "get_positions", "args": {}}],
             "explicit_tickers": [],
             "needs_macro": False,
         })), \
         patch.object(llm, '_dispatch_ibor', new=AsyncMock(return_value=mock_ibor_answer)):

        chunks = []
        async for chunk in llm.chat_stream(
            question="Tell me about my AAPL position",
            market_contents=False,
            prior_context=[],
        ):
            chunks.append(chunk)
            print(f"  chunk: type={chunk['type']}  content={str(chunk.get('content', ''))[:40]}")

    types = [c["type"] for c in chunks]
    text_chunks = [c for c in chunks if c["type"] == "text"]
    done_chunks = [c for c in chunks if c["type"] == "done"]

    assert len(text_chunks) > 0,   "No text chunks received"
    assert len(done_chunks) == 1,  "Expected exactly one done chunk"
    assert done_chunks[0].get("summary"), "done chunk missing summary"

    full_text = "".join(c["content"] for c in text_chunks)
    assert full_text == done_chunks[0]["summary"], "done.summary must equal concatenated text chunks"

    print(f"\n  {len(text_chunks)} text chunks → '{full_text[:60]}...'")
    print("  ✅ chat_stream() chunk types OK")


# ── Test 3: SSE line format ───────────────────────────────────────────────────

def test_sse_line_format():
    """Every SSE line must be parseable as 'data: <valid JSON>'."""
    sample_chunks = [
        {"type": "text", "content": "Your AAPL position"},
        {"type": "text", "content": " is 1,000 shares."},
        {"type": "done", "summary": "Your AAPL position is 1,000 shares.",
         "data": {}, "gaps": [], "quota_status": None},
    ]

    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in sample_chunks]

    for line in lines:
        assert line.startswith("data: "), f"Bad SSE prefix: {line!r}"
        payload = json.loads(line[6:].strip())
        assert "type" in payload, f"Missing 'type' key in: {payload}"

    print(f"\n  {len(lines)} SSE lines validated")
    print("  ✅ SSE line format OK")
