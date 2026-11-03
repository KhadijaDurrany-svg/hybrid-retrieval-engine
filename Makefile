.PHONY: install data test lint bench serve clean

install:
	pip install -e ".[dev]"

data:
	python -m hre.cli generate-data --data-dir data

test:
	pytest -q

lint:
	ruff check .

bench: data
	python benchmarks/run_bench.py --data-dir data --out benchmarks/results.json

serve:
	uvicorn hre.service:app --reload

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
