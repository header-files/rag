"""In-process multi-turn session store."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from llm.types import Message
from rag.errors import RagError


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)


class SessionStore:
    """Process-local session history.

    Concurrent dict access is lock-protected. Concurrent ``/chat`` calls on the
    same ``session_id`` are not strictly serialized.
    """

    def __init__(self, *, max_history_turns: int = 10) -> None:
        self.max_history_turns = max_history_turns
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(session_id=uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RagError(
                    message=f"Unknown session '{session_id}'",
                    type="not_found",
                )
            return Session(
                session_id=session.session_id,
                messages=list(session.messages),
            )

    def append(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RagError(
                    message=f"Unknown session '{session_id}'",
                    type="not_found",
                )
            session.messages.append(Message(role="user", content=user_text))
            session.messages.append(
                Message(role="assistant", content=assistant_text)
            )
            self._trim_locked(session)
            return Session(
                session_id=session.session_id,
                messages=list(session.messages),
            )

    def clear(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RagError(
                    message=f"Unknown session '{session_id}'",
                    type="not_found",
                )
            session.messages.clear()
            return Session(session_id=session.session_id, messages=[])

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _trim_locked(self, session: Session) -> None:
        if self.max_history_turns <= 0:
            session.messages.clear()
            return
        max_messages = self.max_history_turns * 2
        if len(session.messages) > max_messages:
            session.messages[:] = session.messages[-max_messages:]
