"""Query orchestration: optional rewrite → retrieve → context → LLM → session."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from common.config.rag import RagSettings
from common.logger import get_logger
from llm.client import LLMClient
from llm.types import ChatRequest, Message, StreamChunk
from rag.context.builder import ContextBuilder
from rag.errors import RagError
from rag.retrieve.retriever import Retriever
from rag.rewrite.rewriter import QueryRewriter
from rag.session.store import SessionStore
from rag.types import Answer, Citation, SearchHit
from rag.vectorstore.base import VectorStore

log = get_logger(__name__)

EMPTY_STORE_ANSWER = "知识库中没有可用资料，无法回答该问题。"
NO_HITS_ANSWER = "没有检索到与问题相关的资料，无法从已有文档中得到答案。"


@dataclass
class PreparedQuery:
    session_id: str
    query: str
    citations: list[Citation]
    messages: list[Message]
    direct_answer: str | None = None
    retrieval_query: str | None = None


class QueryService:
    def __init__(
        self,
        settings: RagSettings,
        store: VectorStore,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm: LLMClient,
        sessions: SessionStore,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm
        self.sessions = sessions
        self.rewriter = rewriter

    def prepare(
        self,
        question: str,
        *,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> PreparedQuery:
        text = question.strip()
        if not text:
            raise RagError(message="Question must not be empty")

        if session_id:
            session = self.sessions.get(session_id)
        else:
            session = self.sessions.create()

        if document_ids:
            missing = [
                document_id
                for document_id in document_ids
                if self.store.get_document(document_id) is None
            ]
            if missing:
                raise RagError(
                    message="Unknown document: " + ", ".join(missing),
                    type="not_found",
                )

        log.info(
            "query start session_id=%s question_chars=%d filter=%s rewrite=%s",
            session.session_id,
            len(text),
            ",".join(document_ids) if document_ids else "-",
            bool(self.rewriter),
        )

        documents = self.store.list_documents()
        if document_ids is not None:
            allowed = set(document_ids)
            documents = [item for item in documents if item.document_id in allowed]
        if not documents:
            return PreparedQuery(
                session_id=session.session_id,
                query=text,
                citations=[],
                messages=[],
                direct_answer=EMPTY_STORE_ANSWER,
                retrieval_query=text,
            )

        retrieval_query = text
        if self.rewriter is not None:
            retrieval_query = self.rewriter.rewrite(text, session.messages)

        hits = self.retriever.retrieve(retrieval_query, document_ids=document_ids)
        if not hits:
            return PreparedQuery(
                session_id=session.session_id,
                query=text,
                citations=[],
                messages=[],
                direct_answer=NO_HITS_ANSWER,
                retrieval_query=retrieval_query,
            )

        messages, used = self.context_builder.build(
            query=text,
            history=session.messages,
            hits=hits,
        )
        return PreparedQuery(
            session_id=session.session_id,
            query=text,
            citations=_citations(used),
            messages=messages,
            retrieval_query=retrieval_query,
        )

    def complete(self, prepared: PreparedQuery) -> Answer:
        if prepared.direct_answer is not None:
            self.sessions.append(
                prepared.session_id,
                prepared.query,
                prepared.direct_answer,
            )
            return Answer(
                content=prepared.direct_answer,
                citations=prepared.citations,
                session_id=prepared.session_id,
            )
        log.info(
            "query generate start session_id=%s messages=%d",
            prepared.session_id,
            len(prepared.messages),
        )
        try:
            response = self.llm.chat(ChatRequest(messages=prepared.messages))
        except Exception as exc:
            log.error("query generate failed error=%s", exc)
            raise
        self.sessions.append(
            prepared.session_id,
            prepared.query,
            response.content,
        )
        log.info(
            "query done session_id=%s finish_reason=%s citations=%d",
            prepared.session_id,
            response.finish_reason,
            len(prepared.citations),
        )
        return Answer(
            content=response.content,
            citations=prepared.citations,
            session_id=prepared.session_id,
            model=response.model,
            finish_reason=response.finish_reason,
        )

    def iter_stream(self, prepared: PreparedQuery) -> Iterator[StreamChunk]:
        if prepared.direct_answer is not None:
            self.sessions.append(
                prepared.session_id,
                prepared.query,
                prepared.direct_answer,
            )
            yield StreamChunk(type="content", delta=prepared.direct_answer)
            yield StreamChunk(type="finish", finish_reason="stop")
            return
        parts: list[str] = []
        log.info(
            "query generate stream start session_id=%s messages=%d",
            prepared.session_id,
            len(prepared.messages),
        )
        try:
            for chunk in self.llm.chat_stream(ChatRequest(messages=prepared.messages)):
                if chunk.type == "content" and chunk.delta:
                    parts.append(chunk.delta)
                yield chunk
        except Exception as exc:
            log.error("query generate stream failed error=%s", exc)
            raise
        self.sessions.append(
            prepared.session_id,
            prepared.query,
            "".join(parts),
        )
        log.info(
            "query stream done session_id=%s citations=%d",
            prepared.session_id,
            len(prepared.citations),
        )


def _citations(hits: list[SearchHit]) -> list[Citation]:
    return [
        Citation(
            document_id=hit.document_id,
            filename=hit.filename,
            chunk_index=hit.chunk_index,
            score=hit.score,
            text=hit.text,
            chunk_id=hit.chunk_id,
        )
        for hit in hits
    ]
