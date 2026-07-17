"""Small read-only tools: time, arithmetic, market quotes, web search."""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

import config
from brain import market, websearch

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _check_int_size(value) -> None:
    if isinstance(value, int) and value.bit_length() > config.MAX_CALC_RESULT_BITS:
        raise ValueError("result too large")


def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        if isinstance(node.op, ast.Pow):
            return safe_pow(node)
        result = _BINOPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        _check_int_size(result)
        return result
    raise ValueError("unsupported expression")


def safe_pow(node):
    exponent = safe_eval(node.right)
    if abs(exponent) > config.MAX_CALC_POW_EXPONENT:
        raise ValueError("exponent too large")
    base = safe_eval(node.left)
    if isinstance(base, int) and isinstance(exponent, int) and exponent >= 0:
        if max(base.bit_length(), 1) * exponent > config.MAX_CALC_RESULT_BITS:
            raise ValueError("result too large")
    result = operator.pow(base, exponent)
    _check_int_size(result)
    return result


def current_datetime(args: dict | None = None) -> dict:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": now.tzname() or "local",
    }


def calculate(args: dict) -> dict:
    expr = str((args or {}).get("expression", "")).strip()
    if not expr:
        return {"error": "empty expression"}
    if len(expr) > config.MAX_CALC_EXPRESSION_CHARS:
        return {"error": "expression too long"}
    try:
        value = safe_eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return {"expression": expr, "error": "division by zero"}
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return {"expression": expr, "error": f"could not evaluate: {exc}"}
    if isinstance(value, complex):
        return {"expression": expr, "error": "result is not a real number"}
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"expression": expr, "error": "result is not a finite number"}
        value = round(value, 10)
        if value.is_integer():
            value = int(value)
    return {"expression": expr, "result": value}


def get_stock_quotes(args: dict) -> dict:
    tickers = (args or {}).get("tickers") or []
    if isinstance(tickers, str):
        tickers = [tickers]
    symbols, seen = [], set()
    for raw in tickers:
        sym = str(raw).lstrip("$").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
        if len(symbols) >= config.MAX_QUOTE_TICKERS:
            break
    return market.fetch_quotes(symbols)


def web_search(args: dict) -> dict:
    return websearch.search_web(str((args or {}).get("query", "")).strip())
