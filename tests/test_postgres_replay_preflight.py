"""Least-privilege PostgreSQL replay admission requires cleanup and no admin powers."""
import pytest

from src.serving.postgres_nonce_store import PostgresNonceStore


class FakeCursor:
    def __init__(self, permissions):
        self.permissions = permissions
        self.sql = ""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def execute(self, sql):
        self.sql = sql
    def fetchone(self):
        return self.permissions


class FakeConn:
    def __init__(self, permissions):
        self.cursor_obj = FakeCursor(permissions)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def cursor(self):
        return self.cursor_obj


def replay_store(permissions):
    connections = []
    def connector(url, **opts):
        assert url.startswith("postgresql://restricted:")
        assert opts == {"connect_timeout": 3, "sslmode": "require"}
        conn = FakeConn(permissions)
        connections.append(conn)
        return conn
    store = PostgresNonceStore(
        "postgresql://restricted:private-placeholder@db.example/aerynza",
        connector=connector,
    )
    return store, connections


def test_operator_role_preflight_checks_all_runtime_permissions():
    store, connections = replay_store((True, True, True, True, True))
    store.preflight()
    sql = connections[0].cursor_obj.sql
    for privilege in ("rolsuper", "rolbypassrls", "rolcreaterole",
                      "rolcreatedb", "rolreplication", "'INSERT'", "'SELECT'", "'DELETE'"):
        assert privilege in sql


@pytest.mark.parametrize("permissions", [
    None,
    (False, True, True, True, True),
    (True, False, True, True, True),
    (True, True, False, True, True),
    (True, True, True, False, True),
    (True, True, True, True, False),
])
def test_privileged_or_incomplete_database_role_cannot_be_admitted(permissions):
    store, _ = replay_store(permissions)
    with pytest.raises(RuntimeError, match="not properly isolated"):
        store.preflight()


def test_replay_store_outage_never_succeeds_preflight():
    def unavailable(_url, **_opts):
        raise ConnectionError("PostgreSQL unreachable")
    store = PostgresNonceStore(
        "postgresql://restricted:private-placeholder@db.example/aerynza",
        connector=unavailable,
    )
    with pytest.raises(ConnectionError, match="unreachable"):
        store.preflight()
