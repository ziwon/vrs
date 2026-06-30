# Local Console Workflow

This milestone is a local filesystem dashboard for VRS run artifacts. It uses
`runs/` as the source of truth, with no database, broker, object store, or
enterprise service. The API and UI stay CPU-light; real model inference runs in
the separate Docker Compose `inference` profile.

## RTX 5080 Environment

RTX 5080 / Blackwell needs CUDA 12.8+ compatible PyTorch wheels. Use Python 3.11
and the `cu128` extra for GPU pipeline work:

```bash
uv python install 3.11
uv sync --python 3.11 --extra cu128
```

The API itself is CPU-light. Importing `vrs.api.api` does not import
`torch`, `ultralytics`, `transformers`, or `cv2`; it only reads JSONL and image
files that already exist under `runs/`.

## Docker Compose Workflow

Start the local RTSP/API/console stack:

```bash
docker compose up --build
```

Open <http://127.0.0.1:5173>.

The stack starts:

- `rtsp` — MediaMTX RTSP server.
- `clip-init` — one-shot fallback generator for `clips/falldown_test.mp4`
  when the file is absent.
- `rtsp-falldown` — FFmpeg publisher for the selected MP4 under `clips/`.
- `backend` — FastAPI filesystem API over `runs/`.
- `console` — the VRS Console dashboard served by nginx.

The fallback clip created by `clip-init` is only an RTSP plumbing fixture. It is
useful for validating that MediaMTX, FFmpeg, the backend, and the dashboard can
move video and artifacts through the local stack; it is not a fall-detection
accuracy clip or benchmark.

The default console run is `local-rtsp-demo`. It is a deterministic demo artifact
set aligned with the default RTSP sample, not fresh model inference. Fresh
model-backed output appears as the `live` run only after the inference profile
or a manual pipeline writes `/app/runs/live`.

The falldown stream is available as RTSP and HLS:

```text
rtsp://127.0.0.1:8554/falldown
http://127.0.0.1:8888/falldown/index.m3u8
```

To try another MP4, put it under `clips/` and select it with
`VRS_SAMPLE_CLIP`:

```bash
cp /path/to/site-camera-sample.mp4 clips/site-camera-sample.mp4
VRS_SAMPLE_CLIP=site-camera-sample.mp4 docker compose up --build
```

The publisher loops the selected MP4, normalizes timestamps/FPS, and drops
audio. The host-side equivalent command is:

```bash
ffmpeg -re -stream_loop -1 -fflags +genpts \
  -i "clips/${VRS_SAMPLE_CLIP:-falldown_test.mp4}" \
  -map 0:v:0 -an \
  -vf "fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/falldown
```

Inside Docker Compose, the input path is `/clips/...` and the RTSP target is
`rtsp://rtsp:8554/falldown`.

Visually verify the stream:

```bash
ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/falldown
```

For download examples using `curl` or `yt-dlp`, see
[`clips/README.md`](../clips/README.md).

On a GPU host with NVIDIA Container Toolkit, start the inference worker:

```bash
docker compose --profile inference up --build
```

The default `configs/tiny.yaml` verifier uses
`embedl/Cosmos-Reason2-2B-W4A16`, which is a gated Hugging Face repo. For full
two-stage verification, authenticate before starting the profile:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN / HUGGING_FACE_HUB_TOKEN
docker compose --profile inference up --build
```

To reuse a host-side Hugging Face cache, enable
`docker-compose.hf-cache.yaml`. The override maps `VRS_HF_CACHE_DIR` into the
inference container:

```text
${VRS_HF_CACHE_DIR:-./.cache/huggingface}:/models/huggingface
```

Run the full local override stack with:

```bash
VRS_HF_CACHE_DIR=/data/models/huggingface \
docker compose -f docker-compose.yaml -f docker-compose.hf-cache.yaml \
  --profile inference up --build
```

Equivalent Just helpers:

```bash
just local-up
just local-logs
just local-down
```

If no token is present, the worker still validates the real-video RTSP decode,
CUDA container, YOLOE bootstrap, and dependency path before failing at the
gated Cosmos download boundary.

The worker consumes `rtsp://rtsp:8554/falldown` and runs:

```bash
/app/.venv/bin/python scripts/run_rtsp.py \
  --rtsp rtsp://rtsp:8554/falldown \
  --config configs/tiny.yaml \
  --policy configs/policies/falldown_orin_smoke.yaml \
  --out /app/runs/live
```

Useful checks:

```bash
curl http://127.0.0.1:5173/api/health
curl http://127.0.0.1:5173/api/runs
curl http://127.0.0.1:5173/api/streams
curl http://127.0.0.1:5173/api/policy
curl 'http://127.0.0.1:5173/api/runs/local-rtsp-demo/tail?mode=latest&limit=20'
curl http://127.0.0.1:5173/api/runs/live/alerts
curl 'http://127.0.0.1:5173/api/runs/live/tail?mode=latest&limit=20'
```

## CPU-Light Manual Workflow

Generate deterministic fixture runs:

```bash
uv run scripts/make_fixture_runs.py --out runs
```

Start the backend:

```bash
uv run uvicorn vrs.api.api:app --host 127.0.0.1 --port 5445 --reload
```

Serve the static console in another terminal:

```bash
cd console
python -m http.server 5173
```

Open <http://127.0.0.1:5173>. For same-origin API proxying use Docker Compose;
for manual static serving, set `window.VRS_CONFIG.apiBaseUrl` in `console/config.js`
to `http://127.0.0.1:5445`.

Useful API checks:

```bash
curl http://127.0.0.1:5445/api/health
curl http://127.0.0.1:5445/api/runs
curl http://127.0.0.1:5445/api/policy
curl 'http://127.0.0.1:5445/api/runs/fixture/alerts?limit=10'
curl 'http://127.0.0.1:5445/api/runs/fixture_multi/tail?mode=latest&limit=10'
```

`/api/policy` reads `configs/policies/safety.yaml` by default, or the path set
with `VRS_POLICY_PATH`. `/api/runs/{run}/tail` returns `next_cursor`; pass that
cursor back on the next poll to fetch only newly appended alerts:

```bash
curl 'http://127.0.0.1:5445/api/runs/fixture_multi/tail?mode=latest&limit=10'
curl 'http://127.0.0.1:5445/api/runs/fixture_multi/tail?cursor=<next_cursor>'
```

## Optional GPU Plumbing Check

Synthetic clips are useful for validating plumbing, artifact writing, and UI
display. They do not prove model accuracy.

```bash
uv run scripts/make_test_clips.py --out runs/test_clips
uv run scripts/bench.py --clips runs/test_clips --out runs/bench
```

If YOLOE does not fire on a synthetic clip, use the fixture runs above for UI
validation. Real datasets are required for precision/recall claims.

For the Compose falldown stream, the default source clip is short, deterministic,
and looped. It is suitable for local RTSP decode, HLS preview, and
artifact-display tests; it is not a precision/recall benchmark by itself. Use
`VRS_SAMPLE_CLIP` with your own licensed MP4 when you need a realistic camera
sample.

## Real Pipeline Workflow

Single MP4:

```bash
uv run scripts/run_mp4.py \
  --video /path/to/cctv.mp4 \
  --config configs/default.yaml \
  --policy configs/policies/safety.yaml \
  --out runs/demo
```

Multi-stream:

```bash
uv run scripts/run_multistream.py \
  --config  configs/default.yaml \
  --policy  configs/policies/safety.yaml \
  --streams configs/multistream.yaml \
  --out     runs/live
```

Single-stream runs write `runs/<run_name>/alerts.jsonl` and
`runs/<run_name>/thumbnails/*`. Multi-stream runs write
`runs/<run_name>/<stream_id>/alerts.jsonl` and
`runs/<run_name>/<stream_id>/thumbnails/*`.

Cosmos-Reason2-2B is a baseline verifier only, not the final production
verifier. BF16 may not fit comfortably in 16GB VRAM; use a quantized path or a
served verifier backend when needed.

## Troubleshooting

- `no kernel image available`: the Torch/CUDA wheel does not match Blackwell.
  Re-sync with `uv sync --python 3.11 --extra cu128` and verify CUDA 12.8+
  drivers/runtime.
- No runs found: generate fixtures with
  `uv run scripts/make_fixture_runs.py --out runs` or run a VRS pipeline command
  that writes under `runs/`.
- Demo alerts do not match a custom clip: `local-rtsp-demo` is a fixture aligned
  with the default sample only. Use the inference profile or `scripts/run_rtsp.py`
  to generate fresh alerts for a custom `VRS_SAMPLE_CLIP`.
- No `live` run: start the inference profile on a GPU host with
  `docker compose --profile inference up --build`, set `HF_TOKEN` if using the
  gated Cosmos verifier, or run `scripts/run_rtsp.py` manually against
  `rtsp://127.0.0.1:8554/falldown`.
- Thumbnails not loading: check that each alert `thumbnail_path` is relative to
  the run directory, for example `thumbnails/event.png`, and that the backend
  `VRS_RUNS_ROOT` points at the expected `runs/` root.
- CORS or backend URL issues: Docker Compose serves the console and API from the same
  origin at <http://127.0.0.1:5173>. For manual serving, point `console/config.js`
  at the backend URL.
