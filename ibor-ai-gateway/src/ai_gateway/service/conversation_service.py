"""
Conversation Service
====================

Unified service for conversation lifecycle + RAG operations.

Handles:
- Load/create conversations by (analyst_id, session_id)
- Save messages to JSONB
- Extract delta for embeddings
- Embed and store in pgvector
- Search similar conversations

All conversation data is isolated per analyst.
"""

from __future__ import annotations

import asyncio
import logging
import json
from uuid import UUID
from datetime import datetime
from typing import Optional, List, Dict, Any
import psycopg
from ai_gateway.service import embedding_service


log = logging.getLogger(__name__)


class ConversationService:
    """Unified service for conversation lifecycle + RAG operations.

    Conversation isolation: Each analyst has their own conversation namespace.
    Composite key (analyst_id, session_id) ensures no collisions across analysts.
    """

    def __init__(self, pg_pool, anthropic_client):
        self._pool = pg_pool
        self._anthropic = anthropic_client

    # ────────────────────────────────────────────────────────────────
    # LIFECYCLE: Get/Create → Save → Retrieve
    # ────────────────────────────────────────────────────────────────

    async def get_or_create_conversation(
        self,
        analyst_id: str,
        session_id: str,
        context_type: str,
        context_id: str
    ) -> Dict[str, Any]:
        """Load existing conversation or create new one.

        Isolation: (analyst_id, session_id) must be unique.
        Same analyst can have multiple sessions (laptop + phone).
        Different analysts can have the same session_id (no collision).

        Args:
            analyst_id: User identifier (e.g., "john.doe")
            session_id: Browser session UUID
            context_type: Type of context ("portfolio", "instrument", etc.)
            context_id: ID within that context ("P-ALPHA", "EQ-AAPL", etc.)

        Returns:
            {
                "conversation_id": UUID,
                "analyst_id": str,
                "session_id": UUID,
                "context_type": str,
                "context_id": str,
                "messages": list,
                "message_count": int,
                "created_at": datetime
            }
        """
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    # Query by composite key (analyst_id, session_id)
                    cur.execute(
                        """
                        SELECT
                            conversation_id, analyst_id, session_id, context_type, context_id,
                            messages, message_count, created_at, updated_at
                        FROM conv.conversation
                        WHERE analyst_id = %s AND session_id = %s
                        """,
                        (analyst_id, session_id)
                    )
                    row = cur.fetchone()

                    # If exists, return it
                    if row:
                        log.info(f"Loaded conversation: analyst={analyst_id}, session={session_id}")
                        return {
                            "conversation_id": str(row["conversation_id"]),
                            "analyst_id": row["analyst_id"],
                            "session_id": str(row["session_id"]),
                            "context_type": row["context_type"],
                            "context_id": row["context_id"],
                            "messages": row["messages"] or [],
                            "message_count": row["message_count"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"]
                        }

                    # Create new conversation
                    log.info(f"Creating new conversation: analyst={analyst_id}, context={context_type}/{context_id}")
                    cur.execute(
                        """
                        INSERT INTO conv.conversation
                        (analyst_id, session_id, context_type, context_id, messages)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING conversation_id, analyst_id, session_id, context_type, context_id,
                                  messages, message_count, created_at, updated_at
                        """,
                        (analyst_id, session_id, context_type, context_id, json.dumps([]))
                    )
                    row = cur.fetchone()

                    return {
                        "conversation_id": str(row["conversation_id"]),
                        "analyst_id": row["analyst_id"],
                        "session_id": str(row["session_id"]),
                        "context_type": row["context_type"],
                        "context_id": row["context_id"],
                        "messages": row["messages"] or [],
                        "message_count": row["message_count"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"]
                    }

        except Exception as e:
            log.error(f"Failed to get_or_create_conversation: {e}")
            raise

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> None:
        """Append message to conversation JSONB array.

        Also:
        - Increments message_count
        - Updates updated_at timestamp
        - Sets pending_embedding = true (for 5-min scheduler)

        Args:
            conversation_id: UUID of the conversation
            role: "analyst" or "ai"
            content: Message text
        """
        try:
            message = {"role": role, "content": content}
            message_json = json.dumps(message)

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE conv.conversation
                        SET
                            messages = messages || %s::jsonb,
                            message_count = message_count + 1,
                            updated_at = now(),
                            pending_embedding = true
                        WHERE conversation_id = %s
                        """,
                        (f"[{message_json}]", conversation_id)
                    )

                    if cur.rowcount == 0:
                        log.warning(f"save_message: no row updated for conversation_id={conversation_id}")
                    else:
                        log.debug(f"Saved {role} message to conversation_id={conversation_id}")

        except Exception as e:
            log.error(f"Failed to save_message: {e}")
            raise

    async def get_conversation_history(
        self,
        conversation_id: str
    ) -> List[Dict[str, str]]:
        """Retrieve full conversation history.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            [
                {"role": "analyst", "content": "Why am I concentrated in tech?"},
                {"role": "ai", "content": "You own AAPL 25%, MSFT 18%..."},
                ...
            ]
        """
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT messages
                        FROM conv.conversation
                        WHERE conversation_id = %s
                        """,
                        (conversation_id,)
                    )
                    row = cur.fetchone()

                    if not row:
                        log.warning(f"get_conversation_history: conversation_id={conversation_id} not found")
                        return []

                    messages = row["messages"]

                    if messages is None:
                        return []

                    if isinstance(messages, str):
                        messages = json.loads(messages)

                    log.debug(f"Retrieved {len(messages)} messages for conversation_id={conversation_id}")
                    return messages

        except Exception as e:
            log.error(f"Failed to get_conversation_history: {e}")
            raise

    # ────────────────────────────────────────────────────────────────
    # RAG: Extract → Embed → Search
    # ────────────────────────────────────────────────────────────────

    async def extract_delta(
        self,
        conversation_id: str
    ) -> str:
        """Extract new messages since last embedding checkpoint.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            Formatted text of messages, or empty string if nothing new
        """
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT messages, last_embedding_checkpoint, message_count
                        FROM conv.conversation
                        WHERE conversation_id = %s
                        """,
                        (conversation_id,)
                    )
                    row = cur.fetchone()

                    if not row:
                        log.warning(f"extract_delta: conversation_id={conversation_id} not found")
                        return ""

                    messages = row["messages"] or []
                    message_count = row["message_count"]

                    if not messages or message_count == 0:
                        return ""

                    delta_text = self._format_messages_for_embedding(messages)

                    if not delta_text.strip():
                        log.debug(f"extract_delta: No messages for conversation_id={conversation_id}")
                        return ""

                    log.debug(f"extract_delta: {len(messages)} messages for conversation_id={conversation_id}")
                    return delta_text

        except Exception as e:
            log.error(f"Failed to extract_delta: {e}")
            raise

    async def embed_and_store(
        self,
        conversation_id: str,
        context_type: str,
        context_id: str,
        analyst_id: str
    ) -> None:
        """Summarize conversation delta, embed with all-MiniLM-L6-v2, store in pgvector.

        Called by the 5-minute background scheduler for every conversation
        with pending_embedding = true.
        """
        try:
            delta = await self.extract_delta(conversation_id)
            if not delta.strip():
                log.debug("embed_and_store: no delta for %s", conversation_id)
                return

            summary = await self._summarize_text_via_openai(delta)
            vector = await asyncio.to_thread(embedding_service.embed, summary)

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conv.conversation_embedding
                            (conversation_id, analyst_id, context_type, context_id,
                             conversation_summary, embedding, embedding_model)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                        """,
                        (conversation_id, analyst_id, context_type, context_id,
                         summary, vector, "all-MiniLM-L6-v2"),
                    )
                    cur.execute(
                        """
                        UPDATE conv.conversation
                        SET pending_embedding = false,
                            last_embedding_checkpoint = now()
                        WHERE conversation_id = %s
                        """,
                        (conversation_id,),
                    )
            log.info("embed_and_store: stored embedding for conversation %s", conversation_id)

        except Exception as e:
            log.error("embed_and_store failed for %s: %s", conversation_id, e)

    async def search_similar_conversations(
        self,
        query: str,
        context_type: str,
        context_id: str,
        analyst_id: str,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Embed the current question and retrieve the most similar past Q&A summaries.

        Results are filtered by analyst and context so one user never sees
        another user's portfolio conversations.

        Returns top_k results with similarity >= min_similarity (cosine distance
        converted to similarity score: 1 - distance).
        """
        try:
            vector = await asyncio.to_thread(embedding_service.embed, query)

            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            conversation_summary,
                            created_at,
                            1 - (embedding <=> %s::vector) AS similarity_score
                        FROM conv.conversation_embedding
                        WHERE analyst_id = %s
                          AND context_type = %s
                          AND context_id   = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vector, analyst_id, context_type, context_id, vector, top_k),
                    )
                    rows = cur.fetchall()

            results = [
                {
                    "summary": row["conversation_summary"],
                    "similarity_score": float(row["similarity_score"]),
                    "created_at": row["created_at"],
                }
                for row in rows
                if float(row["similarity_score"]) >= min_similarity
            ]
            log.info(
                "search_similar_conversations: %d results (threshold=%.2f) for query=%.60s",
                len(results), min_similarity, query,
            )
            return results

        except Exception as e:
            log.warning("search_similar_conversations failed: %s — returning empty", e)
            return []

    # ────────────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────────────

    def _format_messages_for_embedding(self, messages: List[Dict[str, str]]) -> str:
        """Format messages as readable text for embedding."""
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_label = "Analyst" if role == "analyst" else "Assistant"
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    async def _summarize_text_via_openai(self, text: str) -> str:
        """Summarize text via Anthropic Claude (simple summary for RAG)."""
        try:
            response = await self._anthropic.messages.create(
                model="claude-opus-4-6",
                max_tokens=200,
                system="Summarize this conversation in 1-2 sentences. Focus on: question asked, key insight, outcome.",
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            summary = response.content[0].text
            log.debug(f"_summarize_text_via_anthropic: Generated {len(summary)} char summary")
            return summary

        except Exception as e:
            log.error(f"Failed to summarize_text: {e}")
            raise
