"""Pooled Postgres access. Same psycopg_pool approach the Ozone project added
after its own load-test proved the per-call-connection pattern was the first
real bottleneck under concurrency — starting with pooling here instead of
rediscovering that the hard way."""
from contextlib import contextmanager
from functools import wraps

from psycopg_pool import ConnectionPool

from app import config

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        dsn = (
            f"host={config.POSTGRES_HOST} port={config.POSTGRES_PORT} "
            f"user={config.POSTGRES_USER} password={config.POSTGRES_PASSWORD} "
            f"dbname={config.POSTGRES_DB}"
        )
        _pool = ConnectionPool(dsn, max_size=config.POSTGRES_POOL_MAX_SIZE, open=True)
    return _pool


@contextmanager
def get_conn():
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def apply_schema():
    """Run scripts/schema.sql — idempotent, safe to call on every startup."""
    schema_path = config.ROOT / "scripts" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with get_conn() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(782014)")
        conn.execute(sql)
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        for migration in sorted((config.ROOT / "scripts" / "migrations").glob("*.sql")):
            if not conn.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (migration.name,)).fetchone():
                conn.execute(migration.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations(name) VALUES (%s)", (migration.name,))
        conn.commit()


def single_task(name):
    """One scheduler owner across API processes; release the session lock on exit."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with get_conn() as lease:
                acquired=lease.execute('SELECT pg_try_advisory_lock(hashtext(%s))',(name,)).fetchone()[0]
                if not acquired: return 0
                try: return function(*args,**kwargs)
                finally: lease.execute('SELECT pg_advisory_unlock(hashtext(%s))',(name,))
        return wrapped
    return decorate
