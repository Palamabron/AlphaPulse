import os
import psycopg2
from contextlib import contextmanager
from typing import Iterable, Sequence, Any
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json
import numpy as np

DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/market")

@contextmanager
def get_conn():
    conn = psycopg2.connect(DSN)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()

def insert_bars(rows: Iterable[Sequence[Any]]) -> None:
    q = """
    INSERT INTO market.bars_1m
      (ts, symbol, open, high, low, close, volume, vwap, trades_count)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, ts) DO UPDATE SET
      open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close,
      volume=EXCLUDED.volume, vwap=EXCLUDED.vwap, trades_count=EXCLUDED.trades_count;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(q, rows)
        conn.commit()

def latest_bars(symbol: str, limit: int = 100):
    q = """
    SELECT ts, symbol, open, high, low, close, volume, vwap, trades_count
    FROM market.bars_1m
    WHERE symbol = %s
    ORDER BY ts DESC
    LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (symbol, limit))
        return cur.fetchall()

def insert_embedding(text: str, meta: dict, emb: list[float]):
    with get_conn() as conn, conn.cursor() as cur:
        vec = np.array(emb, dtype=float)
        cur.execute(
            "INSERT INTO ai.document_embedding (contents, metadata, embedding) VALUES (%s,%s,%s) RETURNING id",
            (text, Json(meta) if meta is not None else None, vec),
        )
        _id = cur.fetchone()[0]
        conn.commit()
        return _id

def knn(emb: list[float], k: int = 5):
    with get_conn() as conn, conn.cursor() as cur:
        vec = np.array(emb, dtype=float)
        cur.execute(
            "SELECT id, contents FROM ai.document_embedding ORDER BY embedding <=> %s LIMIT %s",
            (vec, k),
        )
        return cur.fetchall()
