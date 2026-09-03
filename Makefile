.PHONY: install install-gpu test lint audit check validate clean

install:
	python -m pip install -e '.[dev]'

install-gpu:
	python -m pip install -e '.[gpu,dev]'

test:
	pytest

lint:
	ruff check --select E9,F63,F7 src tests scripts

audit:
	python scripts/audit_public_release.py .

check: lint test audit
	python -m compileall -q src scripts

validate:
	paper1-hef --project-root . validate --dataset exp01_all

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
