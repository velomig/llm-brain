"""Canonical result contracts for optional cognitive layers."""
from __future__ import annotations

from brain import layer_contract


def test_normalize_legacy_max_depth_string():
    result = layer_contract.normalize("MAX_DEPTH_REACHED")

    assert result["status"] == "exhausted"
    assert result["reason"] == "max_reflection_depth"


def test_normalize_legacy_skipped_depth_dict():
    result = layer_contract.normalize({"stored": 0, "id": None, "skipped": "max_reflection_depth"})

    assert result["status"] == "exhausted"
    assert result["reason"] == "max_reflection_depth"
    assert result["stored"] == 0
    assert result["id"] is None


def test_normalize_plain_metacognition_noop():
    result = layer_contract.normalize({"stored": 0, "id": None})

    assert result["status"] == "skipped"
    assert result["reason"] == "no_action"
