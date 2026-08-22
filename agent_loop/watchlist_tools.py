"""
FinSight India — Week 4, Part 2: watchlist tools for the agent.

Thin @tool wrappers over watchlist.py, same pattern as week 3's retrieval.py -> retrieval_tools.py. No storage logic lives here.
"""

import json
from langchain_core.tools import tool

from watchlist import add_instrument, remove_instrument, list_watchlist


@tool
def add_to_watchlist(instrument_type: str, symbol: str, label: str = None) -> str:
    """Add an instrument to the persisted watchlist so it can be referred to
    in later conversations, even after this session ends.

    Args:
        instrument_type: Either 'equity' (a stock) or 'option' (an index/stock
            F&O instrument, e.g. NIFTY options).
        symbol: The ticker/symbol, e.g. 'HDFCBANK.NS' for a stock or 'NIFTY'
            for an index options chain.
        label: Optional human-readable name, e.g. 'HDFC Bank'.
    """
    return json.dumps(add_instrument(instrument_type, symbol, label))


@tool
def remove_from_watchlist(symbol: str) -> str:
    """Remove an instrument from the persisted watchlist.

    Args:
        symbol: The ticker/symbol to stop tracking, e.g. 'HDFCBANK.NS'.
    """
    return json.dumps(remove_instrument(symbol))


@tool
def view_watchlist() -> str:
    """View everything currently on the persisted watchlist, across both
    equities and F&O instruments. Use this whenever the user refers to
    'my watchlist', 'the stocks I'm tracking', 'their margins', etc. — check
    what's actually tracked before answering, don't assume."""
    items = list_watchlist()
    if not items:
        return json.dumps({"watchlist": [], "message": "The watchlist is currently empty."})
    return json.dumps({"watchlist": items})


WATCHLIST_TOOLS = [add_to_watchlist, remove_from_watchlist, view_watchlist]