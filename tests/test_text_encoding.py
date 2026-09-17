"""Tracked text files decode as clean UTF-8, with no U+FFFD.

Makes "is this file corrupted?" decidable rather than a matter of how a
cp1252 console happens to render it.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
def _skipped(path: Path) -> bool:
    """Hidden directories, the data cache and build metadata are not ours to check."""
    parts = path.relative_to(ROOT).parts
    return any(p.startswith(".") or p == "data" or p.endswith(".egg-info") for p in parts)


def _candidates() -> list[Path]:
    files = list(ROOT.glob("*.ipynb"))
    for sub in (ROOT, ROOT / "roughvol", ROOT / "tests"):
        files += list(sub.glob("*.py")) + list(sub.glob("*.md"))
    return sorted({f for f in files if not _skipped(f)})


@pytest.mark.parametrize("path", _candidates(), ids=lambda p: str(p.relative_to(ROOT)))
def test_file_is_clean_utf8(path: Path) -> None:
    text = path.read_bytes().decode("utf-8")
    assert chr(0xFFFD) not in text, f"{path} contains a U+FFFD replacement character"
