import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "watchlist.db")

VALID_TYPES = ("equity", "option")


def _get_conn():
    # A fresh connection per call rather than one long-lived global connection —
    # simplest safe choice for a single-user CLI tool, avoids any lock/thread
    # complications, and the overhead is negligible at this scale.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the watchlist table if it doesn't exist yet. Safe to call every
    time the module is used — CREATE TABLE IF NOT EXISTS is a no-op otherwise."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                instrument_type TEXT NOT NULL,
                label TEXT,
                added_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def add_instrument(instrument_type: str, symbol: str, label: str = None) -> dict:
    """Add an instrument to the watchlist, or update it if the symbol is
    already tracked (upsert on symbol, not a duplicate row)."""
    instrument_type = instrument_type.strip().lower()
    if instrument_type not in VALID_TYPES:
        return {"error": f"instrument_type must be one of {VALID_TYPES}, got '{instrument_type}'"}

    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "symbol cannot be empty"}

    init_db()
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO watchlist (symbol, instrument_type, label, added_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(symbol) DO UPDATE SET
                   instrument_type = excluded.instrument_type,
                   label = excluded.label,
                   added_at = excluded.added_at""",
            (symbol, instrument_type, label, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"symbol": symbol, "instrument_type": instrument_type, "label": label, "status": "tracked"}
    finally:
        conn.close()


def remove_instrument(symbol: str) -> dict:
    """Remove an instrument from the watchlist. Graceful no-op (not an error)
    if the symbol wasn't being tracked in the first place."""
    symbol = symbol.strip().upper()
    init_db()
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
        if cursor.rowcount == 0:
            return {"symbol": symbol, "status": "not_found", "message": f"'{symbol}' was not in the watchlist."}
        return {"symbol": symbol, "status": "removed"}
    finally:
        conn.close()


def list_watchlist() -> list[dict]:
    """Return everything currently tracked, grouped implicitly by instrument_type
    (equities and options are just tagged distinctly in each row — the agent
    is responsible for applying the RIGHT follow-up question to the RIGHT type,
    e.g. margin questions to options, fundamentals to equities)."""
    init_db()
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT symbol, instrument_type, label, added_at FROM watchlist ORDER BY added_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()