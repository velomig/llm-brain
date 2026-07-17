from __future__ import annotations

from brain.documents import find_documents


def _touch(path, text="hello"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_find_documents_skips_hidden_files_and_dirs(tmp_path):
    visible = _touch(tmp_path / "note.md")
    nested = _touch(tmp_path / "Finance" / "plan.md")
    _touch(tmp_path / ".obsidian" / "app.json")
    _touch(tmp_path / ".obsidian" / "plugins" / "readme.md")
    _touch(tmp_path / ".trash" / "deleted.md")
    _touch(tmp_path / "._note.md")  # AppleDouble sidecar
    _touch(tmp_path / "Finance" / ".hidden.md")

    found = find_documents([str(tmp_path)])

    assert found == sorted([visible, nested])


def test_find_documents_still_accepts_explicit_file(tmp_path):
    note = _touch(tmp_path / "note.md")
    assert find_documents([str(note)]) == [note]
