set shell := ["bash", "-uc"]

compose := "docker compose -f docker-compose.yaml"
compose-local := "docker compose -f docker-compose.yaml -f docker-compose.hf-local.yaml"
dfire_dataset := "/data/vrs/dfire-mini"
dfire_out := "runs/eval-dfire"
dfire_bbox_out := "runs/eval-dfire-bbox"
dfire_sweep_out := "runs/eval-dfire-sweep"
eval_config := "configs/tiny.yaml"
eval_policy := "configs/policies/dfire_eval.yaml"
dfire_iou := "0.5"
dfire_thresholds := "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50"
dfire_models := "yoloe-11s-seg.pt,yoloe-11l-seg.pt"
dfire_refresh_models := "yoloe-11l-seg.pt,yoloe-26l-seg.pt"
dfire_model_refresh_out := "runs/eval-dfire-model-refresh"
mivia_root := "/data/vrs/kaggle-fire-detection"
mivia_fire_video := "/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire1.avi"
mivia_negative_video := "/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire/fire15.avi"
mivia_config := "runs/eval-dfire-300-prompts/best_config.yaml"
mivia_policy := "runs/eval-dfire-300-prompts/best_policy.yaml"
mivia_fire_out := "runs/mivia-fire/fire1"
mivia_negative_out := "runs/mivia-fire/fire15-negative"
mivia_rtsp_url := "rtsp://127.0.0.1:8554/mivia-fire"
mivia_rtsp_out := "runs/mivia-fire-rtsp"
mivia_rtsp_runtime_s := "60"
stream_manifest := "configs/local-rtsp-streams.yaml"
stream_config := "configs/tiny.yaml"
stream_policy := "configs/policies/safety.yaml"
stream_out := "runs/local-rtsp-multistream"
verifier_dataset := "/data/vrs/verifier-eval"
verifier_dataset_format := "labeled_dir"
verifier_policy := "configs/policies/safety.yaml"
verifier_out := "runs/verifier-bakeoff"
verifier_candidates := "cosmos=configs/default.yaml,qwen-served=configs/qwen-openai-compatible.yaml"
verifier_positive_root := "/data/vrs/kaggle-fire-detection/mivia_fire/mivia_fire"
verifier_positive_class := "fire"
verifier_positive_limit := "3"
verifier_negative_root := "/data/vrs/fire_dataset/extracted_sample/non_fire"
verifier_negative_limit := "3"
verifier_smoke_config := "configs/qwen-openai-compatible.yaml"
verifier_smoke_policy := "configs/policies/safety.yaml"
verifier_smoke_class := "fire"
verifier_smoke_out := "runs/verifier-smoke/result.json"
vllm_smoke_config := "configs/vllm-cosmos.yaml"
vllm_smoke_policy := "configs/policies/safety.yaml"
vllm_smoke_class := "fire"
vllm_smoke_out := "runs/vllm-smoke/result.json"
vllm_bakeoff_baseline_config := "configs/tiny.yaml"
vllm_bakeoff_out := "runs/verifier-vllm-bakeoff"
web_runs_root := "runs"
web_policy := "configs/policies/safety.yaml"
web_host := "127.0.0.1"
web_api_port := "5445"
web_ui_port := "5173"

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

eval-dfire-sweep: _require-dfire-dataset
    uv run --frozen python scripts/sweep_dfire_thresholds.py \
        --dataset {{dfire_dataset}} \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --thresholds {{dfire_thresholds}} \
        --out {{dfire_sweep_out}}

eval-dfire-sweep-bbox: _require-dfire-dataset
    uv run --frozen python scripts/sweep_dfire_thresholds.py \
        --dataset {{dfire_dataset}} \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --thresholds {{dfire_thresholds}} \
        --bbox-iou-threshold {{dfire_iou}} \
        --out {{dfire_sweep_out}}-bbox

eval-dfire-prompt-sweep: _require-dfire-dataset
    uv run --frozen python scripts/sweep_dfire_prompts.py \
        --dataset {{dfire_dataset}} \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --models {{dfire_models}} \
        --thresholds {{dfire_thresholds}} \
        --out {{dfire_sweep_out}}-prompts

eval-dfire-prompt-sweep-bbox: _require-dfire-dataset
    uv run --frozen python scripts/sweep_dfire_prompts.py \
        --dataset {{dfire_dataset}} \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --models {{dfire_models}} \
        --thresholds {{dfire_thresholds}} \
        --bbox-iou-threshold {{dfire_iou}} \
        --out {{dfire_sweep_out}}-prompts-bbox

eval-dfire-model-refresh: _require-dfire-dataset
    uv run --frozen python scripts/eval_detector_models.py \
        --dataset {{dfire_dataset}} \
        --dataset-format dfire \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --models {{dfire_refresh_models}} \
        --baseline-model yoloe-11l-seg.pt \
        --out {{dfire_model_refresh_out}}

eval-dfire-model-refresh-bbox: _require-dfire-dataset
    uv run --frozen python scripts/eval_detector_models.py \
        --dataset {{dfire_dataset}} \
        --dataset-format dfire \
        --config {{eval_config}} \
        --policy {{eval_policy}} \
        --models {{dfire_refresh_models}} \
        --baseline-model yoloe-11l-seg.pt \
        --bbox-iou-threshold {{dfire_iou}} \
        --out {{dfire_model_refresh_out}}-bbox

eval-verifier-bakeoff:
    @test -d "{{verifier_dataset}}" || { \
        echo "missing verifier eval dataset: {{verifier_dataset}}" >&2; \
        echo "expected labeled_dir layout: media file plus sidecar JSON labels" >&2; \
        echo "override with: just verifier_dataset=/path/to/labeled-clips eval-verifier-bakeoff" >&2; \
        exit 1; \
    }
    @set -euo pipefail; \
        args=(); \
        IFS=',' read -ra candidates <<< "{{verifier_candidates}}"; \
        for candidate in "${candidates[@]}"; do \
            args+=(--candidate "$candidate"); \
        done; \
        set -a; [ ! -f .env ] || source .env; set +a; \
        uv run --frozen python scripts/eval_verifier_backends.py \
            --dataset "{{verifier_dataset}}" \
            --dataset-format "{{verifier_dataset_format}}" \
            --policy "{{verifier_policy}}" \
            --out "{{verifier_out}}" \
            "${args[@]}"

prepare-verifier-eval:
    @test -d "{{verifier_positive_root}}" || { \
        echo "missing positive video root: {{verifier_positive_root}}" >&2; \
        echo "override with: just verifier_positive_root=/path/to/positive-videos prepare-verifier-eval" >&2; \
        exit 1; \
    }
    @negative_args=(); \
        if [ -d "{{verifier_negative_root}}" ]; then \
            negative_args=(--negative-root "{{verifier_negative_root}}" --negative-limit "{{verifier_negative_limit}}"); \
        else \
            echo "negative video root not found, preparing positives only: {{verifier_negative_root}}" >&2; \
        fi; \
        uv run --frozen python scripts/prepare_verifier_eval_dataset.py \
            --out "{{verifier_dataset}}" \
            --positive-root "{{verifier_positive_root}}" \
            --positive-class "{{verifier_positive_class}}" \
            --positive-limit "{{verifier_positive_limit}}" \
            "${negative_args[@]}" \
            --copy-mode symlink \
            --overwrite

smoke-verifier:
    @set -a; [ ! -f .env ] || source .env; set +a; \
        uv run --frozen python scripts/smoke_verifier_backend.py \
            --config "{{verifier_smoke_config}}" \
            --policy "{{verifier_smoke_policy}}" \
            --class-name "{{verifier_smoke_class}}" \
            --out "{{verifier_smoke_out}}"

smoke-vllm:
    @set -a; [ ! -f .env ] || source .env; set +a; \
        uv run --frozen python scripts/validate_vllm_backend.py \
            --config "{{vllm_smoke_config}}" \
            --policy "{{vllm_smoke_policy}}" \
            --class-name "{{vllm_smoke_class}}" \
            --out "{{vllm_smoke_out}}"

eval-verifier-vllm-bakeoff:
    @test -d "{{verifier_dataset}}" || { \
        echo "missing verifier eval dataset: {{verifier_dataset}}" >&2; \
        echo "create it with: just prepare-verifier-eval" >&2; \
        exit 1; \
    }
    @set -a; [ ! -f .env ] || source .env; set +a; \
        uv run --frozen python scripts/eval_verifier_backends.py \
            --dataset "{{verifier_dataset}}" \
            --dataset-format "{{verifier_dataset_format}}" \
            --policy "{{verifier_policy}}" \
            --out "{{vllm_bakeoff_out}}" \
            --candidate transformers="{{vllm_bakeoff_baseline_config}}" \
            --candidate vllm="{{vllm_smoke_config}}"

_require-mivia-fire:
    @test -d "{{mivia_root}}" || { \
        echo "missing Kaggle fire dataset root: {{mivia_root}}" >&2; \
        echo "download with: kaggle datasets download -d arfintanim/fire-detection -p {{mivia_root}} --unzip" >&2; \
        echo "override with: just mivia_root=/path/to/kaggle-fire-detection mivia-fire" >&2; \
        exit 1; \
    }
    @test -f "{{mivia_fire_video}}" || { \
        echo "missing MIVIA positive fire video: {{mivia_fire_video}}" >&2; \
        echo "override with: just mivia_fire_video=/path/to/fire.avi mivia-fire" >&2; \
        exit 1; \
    }
    @test -f "{{mivia_negative_video}}" || { \
        echo "missing MIVIA negative/challenging video: {{mivia_negative_video}}" >&2; \
        echo "override with: just mivia_negative_video=/path/to/nonfire.avi mivia-fire-negative" >&2; \
        exit 1; \
    }
    @test -f "{{mivia_config}}" || { \
        echo "missing MIVIA eval config: {{mivia_config}}" >&2; \
        echo "override with: just mivia_config=/path/to/config.yaml mivia-fire" >&2; \
        exit 1; \
    }
    @test -f "{{mivia_policy}}" || { \
        echo "missing MIVIA eval policy: {{mivia_policy}}" >&2; \
        echo "override with: just mivia_policy=/path/to/policy.yaml mivia-fire" >&2; \
        exit 1; \
    }

mivia-find:
    /usr/bin/find "{{mivia_root}}" \( -iname '*mivia*' -o -iname '*.avi' -o -iname '*.mp4' \) -print | sort

mivia-fire: _require-mivia-fire
    uv run --frozen python scripts/run_mp4.py \
        --video {{mivia_fire_video}} \
        --config {{mivia_config}} \
        --policy {{mivia_policy}} \
        --out {{mivia_fire_out}}

mivia-fire-negative: _require-mivia-fire
    uv run --frozen python scripts/run_mp4.py \
        --video {{mivia_negative_video}} \
        --config {{mivia_config}} \
        --policy {{mivia_policy}} \
        --out {{mivia_negative_out}}

mivia-fire-rtsp: _require-mivia-fire
    @command -v ffmpeg >/dev/null || { echo "ffmpeg is required for mivia-fire-rtsp" >&2; exit 1; }
    @command -v timeout >/dev/null || { echo "timeout is required for mivia-fire-rtsp" >&2; exit 1; }
    {{compose}} up -d rtsp
    @set -euo pipefail; \
        mkdir -p "{{mivia_rtsp_out}}"; \
        log="{{mivia_rtsp_out}}/ffmpeg-publisher.log"; \
        echo "publishing {{mivia_fire_video}} to {{mivia_rtsp_url}}"; \
        ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -fflags +genpts \
            -i "{{mivia_fire_video}}" \
            -map 0:v:0 -an \
            -vf "fps=30,format=yuv420p" \
            -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
            -g 30 -keyint_min 30 -sc_threshold 0 \
            -f rtsp -rtsp_transport tcp "{{mivia_rtsp_url}}" > "$log" 2>&1 & \
        publisher_pid=$!; \
        cleanup() { kill "$publisher_pid" >/dev/null 2>&1 || true; wait "$publisher_pid" >/dev/null 2>&1 || true; }; \
        trap cleanup EXIT; \
        sleep 3; \
        if ! kill -0 "$publisher_pid" >/dev/null 2>&1; then \
            echo "ffmpeg publisher exited before VRS could connect; see $log" >&2; \
            exit 1; \
        fi; \
        echo "running VRS on {{mivia_rtsp_url}} for {{mivia_rtsp_runtime_s}}s"; \
        set +e; \
        set -a; [ ! -f .env ] || source .env; set +a; \
        timeout "{{mivia_rtsp_runtime_s}}" uv run --frozen python scripts/run_rtsp.py \
            --rtsp "{{mivia_rtsp_url}}" \
            --config "{{mivia_config}}" \
            --policy "{{mivia_policy}}" \
            --out "{{mivia_rtsp_out}}"; \
        status=$?; \
        set -e; \
        if [ "$status" -eq 124 ]; then \
            echo "bounded RTSP run reached {{mivia_rtsp_runtime_s}}s; treating timeout as success"; \
            exit 0; \
        fi; \
        exit "$status"

mivia-rtsp-publish: _require-mivia-fire
    @command -v ffmpeg >/dev/null || { echo "ffmpeg is required for mivia-rtsp-publish" >&2; exit 1; }
    {{compose}} up -d rtsp
    ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -fflags +genpts \
        -i "{{mivia_fire_video}}" \
        -map 0:v:0 -an \
        -vf "fps=30,format=yuv420p" \
        -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
        -g 30 -keyint_min 30 -sc_threshold 0 \
        -f rtsp -rtsp_transport tcp "{{mivia_rtsp_url}}"

mivia-rtsp-run: _require-mivia-fire
    @set -a; [ ! -f .env ] || source .env; set +a; \
        uv run --frozen python scripts/run_rtsp.py \
            --rtsp "{{mivia_rtsp_url}}" \
            --config "{{mivia_config}}" \
            --policy "{{mivia_policy}}" \
            --out "{{mivia_rtsp_out}}"

local-stream-clips:
    uv run --frozen python scripts/make_test_clips.py \
        --out runs/pr-integration/clips \
        --which all \
        --size 640x360 \
        --fps 30 \
        --seconds 10

local-rtsp-publish-all: local-stream-clips
    @command -v ffmpeg >/dev/null || { echo "ffmpeg is required for local-rtsp-publish-all" >&2; exit 1; }
    {{compose}} up -d rtsp
    uv run --frozen python scripts/publish_rtsp_streams.py --streams "{{stream_manifest}}"

local-multistream-run:
    @set -a; [ ! -f .env ] || source .env; set +a; \
        uv run --frozen python scripts/run_multistream.py \
            --config "{{stream_config}}" \
            --policy "{{stream_policy}}" \
            --streams "{{stream_manifest}}" \
            --out "{{stream_out}}"

stop-local:
    @pkill -f '[p]ublish_rtsp_streams.py' || true
    @pkill -f '[r]un_multistream.py' || true
    @pkill -f '[r]un_rtsp.py' || true
    @pkill -f '[u]vicorn vrs.web.api:app' || true
    @pkill -f '[p]ython -m http.server .*5173' || true
    @pkill -f '[f]fmpeg .*rtsp://127.0.0.1:8554' || true
    {{compose}} down

web-api:
    VRS_RUNS_ROOT="{{web_runs_root}}" VRS_POLICY_PATH="{{web_policy}}" VRS_STREAMS_PATH="{{stream_manifest}}" \
        uv run --frozen uvicorn vrs.web.api:app --host "{{web_host}}" --port "{{web_api_port}}"

web-ui:
    python -m http.server "{{web_ui_port}}" --bind "{{web_host}}" --directory web

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
