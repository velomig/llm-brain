"""Static link/path contracts for shipped project artifacts."""
from __future__ import annotations

import plistlib
import re
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parent.parent


def test_local_markdown_links_resolve():
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    ignored_parts = {".git", ".venv", "memory_store"}
    broken = []

    for markdown in ROOT.rglob("*.md"):
        if ignored_parts & set(markdown.parts):
            continue
        if any(part.startswith("chroma-index-backup-") for part in markdown.parts):
            continue
        for target in pattern.findall(markdown.read_text(encoding="utf-8", errors="replace")):
            local = target.split("#", 1)[0]
            if not local or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (markdown.parent / local).exists():
                broken.append((str(markdown.relative_to(ROOT)), target))

    assert broken == []


def test_vaultwatch_plist_has_no_unexpanded_shell_paths():
    plist_path = ROOT / "launchd" / "com.neuralmap.vaultwatch.plist"
    raw = plist_path.read_text(encoding="utf-8")
    data = plistlib.loads(raw.encode("utf-8"))

    assert "$HOME" not in raw
    path_values = [
        *data["ProgramArguments"][:3],
        data["WorkingDirectory"],
        data["StandardOutPath"],
        data["StandardErrorPath"],
    ]
    assert all(Path(value).is_absolute() for value in path_values)


def test_required_trusted_prompt_paths_exist():
    required = {
        "source_of_truth": config.SOURCE_OF_TRUTH_PATH,
        "nursing": config.NURSING_PATH,
        "finance": config.FINANCE_PATH,
        "agency_protocol": config.AGENCY_PROTOCOL_PATH,
        "action_package_template": config.ACTION_PACKAGE_TEMPLATE_PATH,
        "self_audit_loop": config.SELF_AUDIT_LOOP_PATH,
        "will_perspective": config.WILL_PERSPECTIVE_PATH,
        "will_perspective_journal": config.WILL_PERSPECTIVE_JOURNAL_PATH,
    }

    missing = {name: str(path) for name, path in required.items() if not path.is_file()}

    assert missing == {}
