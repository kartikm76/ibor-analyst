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

import logging
import json
from uuid import UUID
from datetime import datetime
from typing import Optional, List, Dict, Any
import psycopg


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
        """Summarize delta, embed via OpenAI, store in pgvector.

        Called by 5-minute background scheduler.

        Args:
            conversation_id: UUID
            context_type: "portfolio", "instrument", etc.
            context_id: "P-ALPHA", "EQ-AAPL", etc.
            analyst_id: User identifier
        """
        # Embeddings disabled — no-op until a real provider is configured
        log.debug(f"embed_and_store: skipped (embedding provider disabled) for {conversation_id}")
        return

    async def search_similar_conversations(
        self,
        query: str,
        context_type: str,
        context_id: str,
        analyst_id: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar past conversations via pgvector semantic search.

        Used to provide Claude with context from similar past analyses.

        Args:
            query: Current question (e.g., "Why am I concentrated in tech?")
            context_type: Filter to same context type ("portfolio")
            context_id: Filter to same context_id ("P-ALPHA")
            analyst_id: Filter to same analyst (isolation)
            top_k: Number of results to return (default 3)

        Returns:
            [
                {
                    "conversation_id": UUID,
                    "summary": "...",
                    "similarity_score": 0.85,
                    "created_at": datetime
                },
                ...
            ]
        """
        # Embeddings disabled — returns empty until a real provider is configured
        log.debug("search_similar_conversations: skipped (embedding provider disabled)")
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
