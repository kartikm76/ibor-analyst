"""
Pydantic models for conversation & document RAG.

Phase 1: Conversation storage and embedding
Phase 2: Document upload, chunking, and embedding
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


# =====================================================================
# CONVERSATION MODELS (Phase 1)
# =====================================================================

class MessageEntry(BaseModel):
    """Single message in a conversation."""
    role: str  # "user", "assistant", "tool"
    content: str


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    analyst_id: str
    portfolio_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[MessageEntry] = Field(default_factory=list)


class ConversationUpdate(BaseModel):
    """Request to update a conversation (add message)."""
    messages: List[MessageEntry]
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    """Response: existing conversation with metadata."""
    conversation_id: UUID
    analyst_id: str
    portfolio_id: Optional[str]
    title: Optional[str]
    messages: List[MessageEntry]
    message_count: int
    has_embedding: bool
    embedding_status: str  # "NEEDS_INITIAL_EMBEDDING" | "NEEDS_DELTA_EMBEDDING" | "CURRENT"
    created_at: datetime
    updated_at: datetime
    minutes_since_embedding: Optional[float]

    class Config:
        from_attributes = True


class ConversationEmbeddingCreate(BaseModel):
    """Request to embed a conversation."""
    conversation_id: UUID
    embedding: List[float]  # 1536-dim vector from OpenAI
    embedding_model: str = "text-embedding-3-small"


class ConversationEmbeddingResponse(BaseModel):
    """Response: embedding metadata."""
    conversation_id: UUID
    has_embedding: bool
    embedding_model: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConversationListRequest(BaseModel):
    """Request to list conversations."""
    analyst_id: str
    portfolio_id: Optional[str] = None
    limit: int = 20
    offset: int = 0

class ConversationListResponse(BaseModel):
    """Response: list of conversations."""
    total: int
    conversations: List[ConversationResponse]


# =====================================================================
# DOCUMENT MODELS (Phase 2)
# =====================================================================

class DocumentCreate(BaseModel):
    """Request to register a document (before uploading file)."""
    analyst_id: str
    portfolio_id: Optional[str] = None
    document_type: str  # "earnings_pdf", "regulatory", "internal_memo", "market_research"
    title: str
    description: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Document metadata (response)."""
    document_id: UUID
    analyst_id: str
    portfolio_id: Optional[str]
    document_type: str
    title: str
    description: Optional[str]
    file_size_bytes: Optional[int]
    page_count: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentChunkCreate(BaseModel):
    """Single chunk to be created (called by chunking service)."""
    document_id: UUID
    chunk_index: int
    page_number: Optional[int]
    section_title: Optional[str]
    content: str
    token_count: Optional[int]


class DocumentChunkResponse(BaseModel):
    """Response: document chunk with embedding status."""
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    page_number: Optional[int]
    section_title: Optional[str]
    content: str
    token_count: Optional[int]
    has_embedding: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentChunkEmbeddingCreate(BaseModel):
    """Request to embed a document chunk."""
    chunk_id: UUID
    embedding: List[float]  # 1536-dim vector
    embedding_model: str = "text-embedding-3-small"


class DocumentChunkEmbeddingResponse(BaseModel):
    """Response: chunk embedding metadata."""
    chunk_id: UUID
    embedding_model: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentStatus(BaseModel):
    """Document with chunk & embedding statistics."""
    document_id: UUID
    analyst_id: str
    portfolio_id: Optional[str]
    title: str
    document_type: str
    chunk_count: int
    embedded_chunk_count: int
    embedding_percent: float = Field(description="Percentage of chunks with embeddings")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @property
    def embedding_percent_calc(self) -> float:
        """Calculate embedding coverage."""
        if self.chunk_count == 0:
            return 0.0
        return (self.embedded_chunk_count / self.chunk_count) * 100


class DocumentListRequest(BaseModel):
    """Request to list documents."""
    analyst_id: str
    portfolio_id: Optional[str] = None
    document_type: Optional[str] = None
    limit: int = 20
    offset: int = 0


class DocumentListResponse(BaseModel):
    """Response: list of documents with status."""
    total: int
    documents: List[DocumentStatus]


# =====================================================================
# RAG SEARCH MODELS
# =====================================================================

class RAGSearchRequest(BaseModel):
    """Request to search for relevant conversations & documents."""
    analyst_id: str
    question: str
    question_embedding: List[float]  # 1536-dim vector
    portfolio_id: Optional[str] = None
    search_conversations: bool = True
    search_documents: bool = False  # Phase 2 only
    top_k_conversations: int = 3
    top_k_document_chunks: int = 5
    similarity_threshold: float = 0.7


class RAGSearchResultConversation(BaseModel):
    """Single conversation result from RAG search."""
    conversation_id: UUID
    similarity_score: float
    analyst_id: str
    portfolio_id: Optional[str]
    title: Optional[str]
    message_count: int
    created_at: datetime


class RAGSearchResultDocumentChunk(BaseModel):
    """Single document chunk result from RAG search."""
    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_type: str
    chunk_index: int
    page_number: Optional[int]
    section_title: Optional[str]
    content: str
    similarity_score: float
    created_at: datetime


class RAGSearchResponse(BaseModel):
    """Response: relevant conversations and document chunks."""
    question: str
    search_timestamp: datetime
    conversations: List[RAGSearchResultConversation]
    document_chunks: List[RAGSearchResultDocumentChunk]
    total_results: int


# =====================================================================
# LINKING MODELS (Phase 2)
# =====================================================================

class ConversationDocumentLink(BaseModel):
    """Link between a conversation and a document it references."""
    conversation_id: UUID
    document_id: UUID
    mentioned_at_message_idx: Optional[int]  # Which message mentioned the doc
    created_at: datetime

    class Config:
        from_attributes = True
