"""Read-only integrity checks for the cognitive pipeline.

The goal is containment: catch malformed records, dangling graph links, and
snapshot drift before they propagate into reflection, retrieval, or evolution.
"""
from __future__ import annotations

import config
from brain import evolution, graph, log, store

GENERATED_CATEGORIES = {
    "insight",
    "dream",
    "episode_digest",
    "sleep_digest",
    config.METACOGNITION_CATEGORY,
}


def _issue(code: str, message: str, severity: str = "error", **fields) -> dict:
    return {"severity": severity, "code": code, "message": message, **fields}


def _check_store_metadata(model: str | None = None) -> dict:
    issues: list[dict] = []
    rows = store.list_records_all(model)
    seen: dict[str, str] = {}
    for row in rows:
        cid = row.get("id")
        meta = row.get("meta") or {}
        if not cid:
            issues.append(_issue("missing_id", "memory row has no id", mode=row.get("mode")))
            continue
        if cid in seen:
            issues.append(
                _issue("duplicate_id", "memory id appears in multiple stores", id=cid, modes=[seen[cid], row.get("mode")])
            )
        seen[cid] = row.get("mode") or "unknown"
        if not row.get("text"):
            issues.append(_issue("empty_text", "memory has empty text", id=cid))
        if not meta.get("category"):
            issues.append(_issue("missing_category", "memory metadata has no category", id=cid))
        try:
            importance = int(meta.get("importance", 3))
        except (TypeError, ValueError):
            importance = 0
        if importance < 1 or importance > 5:
            issues.append(_issue("bad_importance", "memory importance is outside 1-5", id=cid, importance=meta.get("importance")))
        if meta.get("category") in GENERATED_CATEGORIES and not meta.get("source_type"):
            issues.append(
                _issue("generated_missing_source_type", "generated memory lacks source_type provenance", id=cid, category=meta.get("category"))
            )
    return {"ok": not any(i["severity"] == "error" for i in issues), "count": len(rows), "issues": issues}


def _check_graph_edges(model: str | None = None) -> dict:
    issues: list[dict] = []
    edges = graph.export_edges()
    active_collections = {store.collection_name(model), store.collection_name(model, text_only=True), None}
    ids = [edge.get("fact_id") for edge in edges if edge.get("fact_id")]
    # A single memory can support multiple graph edges. Chroma rejects duplicate
    # ids in one get() call, so collapse while preserving first-seen order.
    lookup_ids = list(dict.fromkeys(i for i in ids if isinstance(i, str)))
    resolved = store.get_by_ids(lookup_ids, model=model)
    for edge in edges:
        fact_id = edge.get("fact_id")
        if not fact_id:
            continue
        collection = edge.get("collection")
        if fact_id in resolved:
            continue
        if collection not in active_collections:
            issues.append(
                _issue(
                    "foreign_graph_fact_unresolved",
                    "graph edge points at a fact outside the active collections",
                    "warning",
                    fact_id=fact_id,
                    collection=collection,
                )
            )
        else:
            issues.append(
                _issue("dangling_graph_fact", "graph edge points at a missing memory", fact_id=fact_id, collection=collection)
            )
    return {"ok": not any(i["severity"] == "error" for i in issues), "count": len(edges), "issues": issues}


def _check_evolution_snapshots() -> dict:
    issues: list[dict] = []
    current = evolution.state()
    snapshots = current.get("snapshots") or []
    snapshot_ids = {s.get("id") for s in snapshots if s.get("id")}
    active = current.get("active") or {}
    active_snapshot = active.get("snapshot_id")
    if active_snapshot and active_snapshot not in snapshot_ids:
        issues.append(_issue("missing_active_snapshot", "active snapshot_id is not present in snapshots", snapshot_id=active_snapshot))
    for proposal in current.get("proposals") or []:
        if proposal.get("status") != "approved":
            continue
        if not (proposal.get("approval_checks") or {}).get("canary_review", {}).get("ok"):
            issues.append(_issue("approved_without_canary", "approved proposal lacks a passing canary review", proposal_id=proposal.get("id")))
    return {"ok": not any(i["severity"] == "error" for i in issues), "count": len(snapshots), "issues": issues}


def _check_recent_fallbacks(limit: int = 25) -> dict:
    events = log.recent(limit)
    issues = [
        _issue("recent_fallback", "recent fallback event was logged", "warning", event=e.get("event"), error=e.get("error"), fields=e.get("fields"))
        for e in events
    ]
    return {"ok": True, "count": len(events), "issues": issues}


def run(model: str | None = None, fallback_limit: int = 25) -> dict:
    checks = {
        "store_metadata": _check_store_metadata(model=model),
        "graph_edges": _check_graph_edges(model=model),
        "evolution_snapshots": _check_evolution_snapshots(),
        "recent_fallbacks": _check_recent_fallbacks(limit=fallback_limit),
    }
    errors = sum(1 for check in checks.values() for issue in check["issues"] if issue["severity"] == "error")
    warnings = sum(1 for check in checks.values() for issue in check["issues"] if issue["severity"] == "warning")
    return {"ok": errors == 0, "summary": {"errors": errors, "warnings": warnings}, "checks": checks}


def repair(model: str | None = None, dry_run: bool = True) -> dict:
    """Repair safe integrity drift.

    Currently this only removes graph edges that point at missing memories in the
    active store. Memory rows, foreign collection warnings, and evolution state
    are left untouched.
    """
    graph_check = _check_graph_edges(model=model)
    dangling = [issue for issue in graph_check["issues"] if issue.get("code") == "dangling_graph_fact"]
    fact_ids = [issue.get("fact_id") for issue in dangling if issue.get("fact_id")]
    collections = list(
        dict.fromkeys(issue.get("collection") for issue in dangling if issue.get("collection") in {
            store.collection_name(model),
            store.collection_name(model, text_only=True),
            None,
        })
    )
    deleted = 0 if dry_run else graph.delete_fact_ids(fact_ids, collections=collections)
    return {
        "dry_run": dry_run,
        "repairable": len(dangling),
        "fact_ids": list(dict.fromkeys(fact_ids)),
        "deleted": deleted,
        "report": run(model=model, fallback_limit=0) if not dry_run else None,
    }
