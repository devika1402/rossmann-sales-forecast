.PHONY: install test train baseline predict eda clean lint

PY := .venv/bin/python

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,explain,notebook]"

test:
	$(PY) -m pytest -q

train:
	$(PY) scripts/train.py --model lgbm

baseline:
	$(PY) scripts/train.py --model baseline

predict:
	$(PY) scripts/predict.py

eda:
	$(PY) -m jupyter notebook notebooks/01_eda.ipynb

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts

clean:
	rm -rf data/interim/* data/processed/* models/*.pkl outputs/submission.csv
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
