"""
FinSight India — Week 1 tools
Five plain Python functions. No framework, no LLM calls yet.
Get these working standalone first (Day 1-2 goal), THEN wire them to Gemini/Groq (Day 3-7).
"""

import math
import re
import yfinance as yf
from jugaad_data.nse import NSELive  # pip install jugaad-data
import ast
import operator


def _safe_round(value, decimals=2):
    """Round a value, but return None instead of NaN/inf.
    NaN is NOT valid JSON — if this leaks into a tool response the model
    (or anything downstream that json.dumps()'s it) will choke silently.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, decimals)


# ---------- 1. get_stock_price ----------
def get_stock_price(ticker: str) -> dict:
    """Get the latest closing price for an NSE-listed stock.

    Args:
        ticker: NSE ticker with .NS suffix, e.g. 'TCS.NS', 'RELIANCE.NS'
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")

    # Two failure shapes to guard against, not just "empty":
    # (a) truly empty df — bad/delisted ticker, or market never opened for the range
    # (b) non-empty df but the Close value is NaN — happens on some illiquid tickers
    if hist.empty:
        return {"error": f"No data found for '{ticker}'. Check the symbol and .NS/.BO suffix."}

    close = _safe_round(hist["Close"].iloc[-1])
    if close is None:
        return {"error": f"'{ticker}' returned no valid closing price (market may be closed or ticker illiquid)."}

    return {
        "ticker": ticker,
        "close_price": close,
        "date": str(hist.index[-1].date()),
    }


# ---------- 2. get_company_overview ----------
def get_company_overview(ticker: str) -> dict:
    """Get basic fundamentals for an NSE-listed stock.

    Args:
        ticker: NSE ticker with .NS suffix, e.g. 'TCS.NS'
    """
    info = yf.Ticker(ticker).info
    if not info or info.get("longName") is None:
        return {"error": f"No fundamentals found for '{ticker}'. Check the symbol and .NS/.BO suffix."}

    market_cap = info.get("marketCap")  # may be missing key OR present-but-None
    market_cap_cr = _safe_round(market_cap / 1e7) if market_cap else None

    return {
        "ticker": ticker,
        "name": info.get("longName"),
        "sector": info.get("sector"),  # None for indices/ETFs — expected, not an error
        "pe_ratio": _safe_round(info.get("trailingPE")),  # None for loss-making companies — expected
        "market_cap_cr": market_cap_cr,
        "52w_high": _safe_round(info.get("fiftyTwoWeekHigh")),
        "52w_low": _safe_round(info.get("fiftyTwoWeekLow")),
    }


# ---------- 3. get_option_chain ----------
def get_option_chain(symbol: str) -> dict:
    """Get the current option chain for an index or F&O stock.

    Args:
        symbol: e.g. 'NIFTY', 'BANKNIFTY', or a stock symbol like 'RELIANCE'
    """
    n = NSELive()
    try:
        if symbol.upper() in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
            data = n.index_option_chain(symbol.upper())
        else:
            data = n.equities_option_chain(symbol.upper())

        all_strikes = data.get("records", {}).get("data", [])
        if not all_strikes:
            return {"error": f"No option chain data returned for '{symbol}' — check the symbol has active F&O contracts."}

        underlying = all_strikes[0].get("CE", {}).get("underlyingValue")
        if underlying is None:
            return {"error": f"Could not determine underlying price for '{symbol}'."}

        near_atm = sorted(all_strikes, key=lambda r: abs(r["strikePrice"] - underlying))[:10]
        # Round the numeric fields we actually care about so the model isn't
        # handed raw NSE floats with trailing precision noise
        cleaned = []
        for row in near_atm:
            cleaned_row = {"strikePrice": _safe_round(row.get("strikePrice"), 1)}
            for side in ("CE", "PE"):
                if side in row:
                    cleaned_row[side] = {
                        "lastPrice": _safe_round(row[side].get("lastPrice")),
                        "openInterest": row[side].get("openInterest"),  # OI is an integer count, leave as-is
                        "impliedVolatility": _safe_round(row[side].get("impliedVolatility"), 2),
                    }
            cleaned.append(cleaned_row)

        return {
            "symbol": symbol,
            "underlying_value": _safe_round(underlying),
            "near_atm_strikes": cleaned,
        }
    except Exception as e:
        return {"error": f"Could not fetch option chain for '{symbol}': {e}"}


# ---------- 4. get_repo_rate ----------
def get_repo_rate() -> dict:
    """Get the current RBI repo rate as a clean float (e.g. 6.5, not '6.50%')."""
    # jugaad-data's RBI module changes occasionally — check current method name
    # in jugaad_data.rbi if this breaks. Fallback: scrape rbi.org.in press releases.
    from jugaad_data.rbi import RBI
    rbi = RBI()
    try:
        rates = rbi.current_rates()
        raw = rates.get("Policy Repo Rate")
        if raw is None:
            return {"error": "Repo rate not found in RBI response — check jugaad_data.rbi's current field names."}
        # raw is often a string like "6.50%" — strip non-numeric characters before returning
        numeric = re.sub(r"[^\d.]", "", str(raw))
        return {"repo_rate_percent": _safe_round(numeric)}
    except Exception as e:
        return {"error": f"Could not fetch repo rate: {e}"}


# ---------- 5. calculate ----------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def calculate(expression: str) -> dict:
    """Safely evaluate a basic arithmetic expression. No eval() — parses via AST.

    Args:
        expression: e.g. '(120.5 - 98.2) / 98.2 * 100'
    """
    def _eval(node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
            return node.value
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_OPS:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        # Guard against float noise, e.g. 0.1 + 0.2 -> 0.30000000000000004
        result = _safe_round(result, 10) if isinstance(result, float) else result
        return {"expression": expression, "result": result}
    except ZeroDivisionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


if __name__ == "__main__":
    # Day 1-2 sanity check — run this file directly, no LLM involved yet
    print(get_stock_price("TCS.NS"))
    print(get_stock_price("NOTAREALTICKER.NS"))  # should return a clean error, not crash
    print(get_company_overview("INFY.NS"))
    print(calculate("(3500 - 3200) / 3200 * 100"))
    print(calculate("5 / 0"))  # should return a clean error, not crash
    # option chain and repo rate need jugaad-data hitting NSE/RBI live —
    # test these separately once the first three work, NSE can be flaky/rate-limited
    print(get_option_chain("NIFTY"))
    print(get_repo_rate())