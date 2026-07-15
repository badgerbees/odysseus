from types import SimpleNamespace

from core.models import ChatMessage
from core.session_manager import SessionManager
import core.session_manager as SM


def _manager_with(sessions):
    manager = SessionManager.__new__(SessionManager)
    manager.sessions = dict(sessions)
    return manager


# ---------------------------------------------------------------------------
# Plain-Python fake DB — avoids MagicMock / SQLAlchemy interference entirely.
# MagicMock-based chains fail on CI (Python 3.11.15 + SQLAlchemy 2.x) because
# using a MagicMock as a SQLAlchemy model class in .query()/.filter() triggers
# internal SQLAlchemy inspection that raises a silently-caught exception,
# preventing both sessions.pop() and db.add() from ever being reached.
# ---------------------------------------------------------------------------

class _FakeDB:
    """Minimal DB session double. Subclass to customise query results."""

    def __init__(self):
        self.added = []
        self.committed = False
        self.rolled_back = False

    # query chain — model / filter args are intentionally ignored
    def query(self, model):
        return self

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        raise NotImplementedError  # override in subclass

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


class _MissingParentDB(_FakeDB):
    """Simulates a session row that no longer exists in the DB."""

    def first(self):
        return None


class _ExistingParentDB(_FakeDB):
    """Simulates a session row that still exists in the DB."""

    def __init__(self, parent_row):
        super().__init__()
        self._parent = parent_row

    def first(self):
        return self._parent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _re_raise_logger_error(*args, **kwargs):
    import sys
    _, exc, _ = sys.exc_info()
    if exc:
        raise exc
    raise RuntimeError(f"Logger error called: {args}")


def test_persist_message_drops_write_when_parent_session_is_gone(monkeypatch):
    db = _MissingParentDB()
    monkeypatch.setattr(SM, "SessionLocal", lambda: db)
    monkeypatch.setattr(SM.logger, "error", _re_raise_logger_error)

    manager = _manager_with({"deleted": SimpleNamespace(history=[])})
    message = ChatMessage("assistant", "late token")

    manager._persist_message("deleted", message)

    assert "deleted" not in manager.sessions
    assert db.added == []
    assert not db.committed
    assert not db.rolled_back


def test_persist_message_still_writes_when_parent_session_exists(monkeypatch):
    parent = SimpleNamespace(message_count=0, last_accessed=None, last_message_at=None)
    db = _ExistingParentDB(parent)
    monkeypatch.setattr(SM, "SessionLocal", lambda: db)
    monkeypatch.setattr(SM.logger, "error", _re_raise_logger_error)

    message = ChatMessage("user", "hello")
    manager = _manager_with({"sid": SimpleNamespace(history=[message])})

    manager._persist_message("sid", message)

    assert len(db.added) == 1
    assert db.committed
    assert parent.message_count == 1
    assert parent.last_accessed is not None
    assert parent.last_message_at is not None
    assert message.metadata["_db_id"]
    assert message.metadata["timestamp"].endswith("Z")
