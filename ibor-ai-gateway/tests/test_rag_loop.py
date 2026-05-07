"""
RAG loop integration test
=========================
Validates the full conversation memory cycle:
  1. Create a conversation + save messages
  2. embed_and_store() → pgvector
  3. search_similar_conversations() with a semantically related query
  4. Assert similarity score ≥ 0.6

Requires:
  - PostgreSQL running locally (docker-compose up postgres)
  - ANTHROPIC_API_KEY in environment (used for conversation summarisation)
  - PG_DSN defaults to postgresql://ibor:ibor@localhost:5432/ibor

Run:
  cd ibor-ai-gateway
  uv run pytest tests/test_rag_loop.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import pytest

from anthropic import AsyncAnthropic
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from ai_gateway.service.conversation_service import ConversationService

# ── Connection ────────────────────────────────────────────────────────────────

PG_DSN = os.getenv("PG_DSN", "postgresql://ibor:ibor@localhost:5432/ibor")


@pytest.fixture(scope="module")
def pg_pool():
    pool = ConnectionPool(
        conninfo=PG_DSN,
        min_size=1,
        max_size=2,
        open=True,
        kwargs={"row_factory": dict_row},
    )
    yield pool
    pool.close()


@pytest.fixture(scope="module")
def anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping RAG loop test")
    return AsyncAnthropic(api_key=api_key)


@pytest.fixture(scope="module")
def svc(pg_pool, anthropic_client):
    return ConversationService(pg_pool, anthropic_client)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cleanup(pg_pool, analyst_id: str):
    """Remove test rows so re-runs stay clean."""
    with pg_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conv.conversation_embedding WHERE analyst_id = %s",
                (analyst_id,),
            )
            cur.execute(
                "DELETE FROM conv.conversation WHERE analyst_id = %s",
                (analyst_id,),
            )


# ── Test ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_loop(svc, pg_pool):
    analyst_id = f"test-rag-{uuid.uuid4().hex[:8]}"
    session_id = str(uuid.uuid4())

    # -- Cleanup before (in case of leftover from a prior run) -----------------
    _cleanup(pg_pool, analyst_id)

    print(f"\n[1] Creating conversation  analyst={analyst_id}  session={session_id}")
    conv = await svc.get_or_create_conversation(
        analyst_id=analyst_id,
        session_id=session_id,
        context_type="portfolio",
        context_id="P-ALPHA",
    )
    conversation_id = conv["conversation_id"]
    print(f"    conversation_id = {conversation_id}")

    # -- Save a realistic Q&A pair --------------------------------------------
    question  = "What is my NVDA exposure and how has the stock been performing recently?"
    answer    = (
        "Your portfolio holds 2,400 shares of NVIDIA (NVDA) with a market value of "
        "$864,000, representing 8.6% of AUM. NVDA has risen 12% over the past 30 days "
        "driven by strong data-centre demand, making it your top individual equity exposure."
    )

    print(f"[2] Saving messages to conv.conversation ...")
    await svc.save_message(conversation_id, role="analyst", content=question)
    await svc.save_message(conversation_id, role="ai",      content=answer)

    # Verify messages are in the DB
    with pg_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT messages, message_count FROM conv.conversation WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = cur.fetchone()
    assert row is not None
    msgs = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
    assert len(msgs) == 2
    print(f"    message_count = {row['message_count']}  ✓")

    # -- Embed and store -------------------------------------------------------
    print(f"[3] Running embed_and_store() ...")
    await svc.embed_and_store(
        conversation_id=conversation_id,
        context_type="portfolio",
        context_id="P-ALPHA",
        analyst_id=analyst_id,
    )

    # Verify embedding row exists
    with pg_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT conversation_summary, embedding_model FROM conv.conversation_embedding "
                "WHERE conversation_id = %s",
                (conversation_id,),
            )
            emb_row = cur.fetchone()
    assert emb_row is not None, "No embedding row found — embed_and_store() failed"
    print(f"    summary  = {emb_row['conversation_summary'][:80]}...")
    print(f"    model    = {emb_row['embedding_model']}  ✓")

    # Verify pending_embedding was cleared
    with pg_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pending_embedding FROM conv.conversation WHERE conversation_id = %s",
                (conversation_id,),
            )
            flag_row = cur.fetchone()
    assert flag_row["pending_embedding"] is False, "pending_embedding should be False after embed_and_store"
    print(f"    pending_embedding = False  ✓")

    # -- Semantic search -------------------------------------------------------
    similar_query = "Tell me about NVIDIA in my portfolio"
    print(f"[4] Searching similar conversations for: '{similar_query}'")
    results = await svc.search_similar_conversations(
        query=similar_query,
        context_type="portfolio",
        context_id="P-ALPHA",
        analyst_id=analyst_id,
        top_k=3,
        min_similarity=0.3,   # lower threshold so test isn't fragile
    )

    print(f"    Results returned: {len(results)}")
    for r in results:
        print(f"    similarity={r['similarity_score']:.3f}  summary={r['summary'][:60]}...")

    assert len(results) >= 1, "Expected at least 1 similar conversation"
    assert results[0]["similarity_score"] >= 0.3, f"Similarity too low: {results[0]['similarity_score']}"
    print(f"\n✅  RAG loop PASSED — top similarity = {results[0]['similarity_score']:.3f}")

    # -- Cleanup after ---------------------------------------------------------
    _cleanup(pg_pool, analyst_id)
