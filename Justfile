set shell := ["bash", "-uc"]

compose := "docker compose -f docker-compose.yaml"
compose-local := "docker compose -f docker-compose.yaml -f docker-compose.hf-local.yaml"
dfire_dataset := "/data/vrs/dfire-mini"
dfire_out := "runs/eval-dfire"
dfire_bbox_out := "runs/eval-dfire-bbox"
eval_config := "configs/default.yaml"
eval_policy := "configs/policies/safety.yaml"
dfire_iou := "0.5"

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

_require-dfire-dataset:
    @test -d "{{dfire_dataset}}" || { \
        echo "missing D-Fire dataset: {{dfire_dataset}}" >&2; \
        echo "expected layout:" >&2; \
        echo "  {{dfire_dataset}}/images/" >&2; \
        echo "  {{dfire_dataset}}/labels/" >&2; \
        echo "override with: just dfire_dataset=/path/to/dfire eval-dfire" >&2; \
        exit 1; \
    }
    @test -d "{{dfire_dataset}}/images" || { echo "missing {{dfire_dataset}}/images" >&2; exit 1; }
    @test -d "{{dfire_dataset}}/labels" || { echo "missing {{dfire_dataset}}/labels" >&2; exit 1; }
    @find -L "{{dfire_dataset}}/images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | grep -q . || { \
        echo "no D-Fire images found in {{dfire_dataset}}/images" >&2; \
        echo "copy or symlink the downloaded D-Fire image files there before running eval" >&2; \
        exit 1; \
    }

eval-dfire: _require-dfire-dataset
    uv run --frozen python scripts/eval.py \
        --dataset {{dfire_dataset}} \
        --dataset-format dfire \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --mode detector_only \
        --out {{dfire_out}}

eval-dfire-bbox: _require-dfire-dataset
    uv run --frozen python scripts/eval.py \
        --dataset {{dfire_dataset}} \
        --dataset-format dfire \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --mode detector_only \
        --bbox-iou-threshold {{dfire_iou}} \
        --out {{dfire_bbox_out}}

compose-up *args:
    {{compose}} up -d --build {{args}}

compose-down *args:
    {{compose}} down {{args}}

compose-logs *args:
    {{compose}} logs {{args}}

compose-ps:
    {{compose}} ps -a

local-up *args:
    {{compose-local}} --profile inference up -d --build {{args}}

local-down *args:
    {{compose-local}} --profile inference down {{args}}

local-restart-inference:
    {{compose-local}} --profile inference up -d --force-recreate inference

local-logs *args:
    {{compose-local}} --profile inference logs {{args}}

local-ps:
    {{compose-local}} --profile inference ps -a
