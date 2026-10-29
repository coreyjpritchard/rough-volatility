# Rebuild every figure and CSV from its seed. Needs uv (https://docs.astral.sh/uv/).
.PHONY: reproduce sync test lint figures

reproduce: sync test figures

sync:
	uv sync --extra dev

test:
	uv run pytest -m "not slow"

lint:
	uv run ruff check .

# Yahoo's terms do not allow redistribution, so this cache is built locally.
data/spx_ohlc_yahoo.parquet:
	uv run --extra data python -m roughvol.realised yahoo

figures: data/spx_ohlc_yahoo.parquet
	uv run python docs/make_readme_figure.py
	uv run python scripts/figures_rva.py
	uv run python scripts/figures_rvb.py
	uv run python scripts/figures_rvc.py
