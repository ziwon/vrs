set shell := ["bash", "-uc"]

default:
    @just --list

sync cuda="cu128":
    uv sync --extra {{cuda}}

sync-dev cuda="cu128":
    uv sync --extra {{cuda}} --group dev

lock:
    uv lock

lock-check:
    uv lock --check

test *args:
    uv run --frozen python -m pytest {{args}}

test-parallel *args:
    uv run --frozen python -m pytest -n auto {{args}}

cov *args:
    uv run --frozen python -m pytest --cov=vrs --cov-report=term-missing --cov-report=xml {{args}}

lint *args:
    uv run --frozen ruff check . {{args}}

fmt *args:
    uv run --frozen ruff format . {{args}}

fmt-check:
    uv run --frozen ruff format --check .

pre-commit *args:
    uv run --frozen --only-group dev pre-commit {{args}}

hooks-install:
    uv run --frozen --only-group dev pre-commit install

hooks-run:
    uv run --frozen --only-group dev pre-commit run --all-files

build:
    uv build --clear

package: build
    @ls -1 dist

check: lock-check fmt-check lint test build
