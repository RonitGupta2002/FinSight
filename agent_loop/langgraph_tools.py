"""
FinSight India — Week 2: LangChain tool wrappers around week 1's plain functions.
No new logic here — this file's only job is adapting week1's functions to the
shape LangGraph/LangChain expects (the @tool decorator + JSON-string returns).
"""

import sys
import os
import json

# Reuse week 1's tools directly instead of duplicating logic
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tool_calling"))
from tools_def import (
    get_stock_price as _get_stock_price,
    get_company_overview as _get_company_overview,
    get_option_chain as _get_option_chain,
    get_repo_rate as _get_repo_rate,
    calculate as _calculate,
)

from langchain_core.tools import tool


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price for an NSE-listed stock.

    Args:
        ticker: NSE ticker with .NS suffix, e.g. 'TCS.NS', 'RELIANCE.NS'
    """
    return json.dumps(_get_stock_price(ticker))


@tool
def get_company_overview(ticker: str) -> str:
    """Get basic fundamentals (PE ratio, market cap, sector) for an NSE-listed stock.

    Args:
        ticker: NSE ticker with .NS suffix, e.g. 'TCS.NS'
    """
    return json.dumps(_get_company_overview(ticker))


@tool
def get_option_chain(symbol: str) -> str:
    """Get the current near-the-money option chain for an index or F&O stock.

    Args:
        symbol: e.g. 'NIFTY', 'BANKNIFTY', or a stock symbol like 'RELIANCE'
    """
    return json.dumps(_get_option_chain(symbol))


@tool
def get_repo_rate() -> str:
    """Get the current RBI repo rate as a percentage."""
    return json.dumps(_get_repo_rate())


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: e.g. '(120.5 - 98.2) / 98.2 * 100'
    """
    return json.dumps(_calculate(expression))


ALL_TOOLS = [get_stock_price, get_company_overview, get_option_chain, get_repo_rate, calculate]