# Rebuild every figure and CSV from its seed. Needs uv (https://docs.astral.sh/uv/).
.PHONY: reproduce sync test lint figures

reproduce: sync test figures

sync:
	uv sync --extra dev

test:
	uv run pytest -m "not slow"

lint:
	uv run ruff check .

figures:
	uv run python docs/make_readme_figure.py
	uv run python scripts/figures_rva.py
	uv run python scripts/figures_rvb.py
	uv run python scripts/figures_rvc.py
