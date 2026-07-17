"""The calculate tool must only ever return JSON-serializable, real results.

The result is handed back to the model through the tool channel as json.dumps(...),
which runs OUTSIDE the per-call try/except in llm.chat_with_tools — so a value
json can't encode (a complex number from a fractional power of a negative, or a
non-finite float from overflow) would raise and abort the whole turn. These cases
must come back as clean {"error": ...} dicts instead.
"""
from __future__ import annotations

import json

from brain.tools import _calculate


def _assert_serializable(result: dict) -> None:
    json.dumps(result)  # would raise on complex / would emit invalid JSON tokens


def test_fractional_power_of_negative_is_rejected_not_complex():
    out = _calculate("(-2)**0.5")
    assert "result" not in out
    assert out["error"] == "result is not a real number"
    _assert_serializable(out)


def test_overflow_to_infinity_is_rejected():
    out = _calculate("1e308 * 1e308")
    assert "result" not in out
    assert out["error"] == "result is not a finite number"
    _assert_serializable(out)


def test_plain_arithmetic_still_works():
    assert _calculate("37112.25 + 14830.11")["result"] == 51942.36
    assert _calculate("8 / 2")["result"] == 4          # clean integer float -> int
    assert _calculate("2 ** 0.5")["result"] == 1.4142135624


def test_existing_dos_guards_unaffected():
    assert "too large" in _calculate("((2**100)**100)**100")["error"]
    assert "exponent too large" in _calculate("10**5000")["error"]


def test_every_result_is_json_serializable():
    for expr in ["(-2)**0.5", "1e308*1e308", "2**0.5", "8/2", "1/0", "10**5000", "5+5"]:
        _assert_serializable(_calculate(expr))
