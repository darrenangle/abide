from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_SURFACE = [
    ROOT / "README.md",
    *sorted((ROOT / "scripts").glob("*.py")),
    *sorted((ROOT / "scripts").glob("*.sh")),
]


def test_repo_surface_has_no_hardcoded_miniconda_paths() -> None:
    for path in REPO_SURFACE:
        text = path.read_text()
        assert "/home/darren/miniconda3" not in text, path.as_posix()


def test_readme_uses_uv_for_repo_workflows() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "uv sync --all-extras" in readme
    assert "pip install abide" not in readme
