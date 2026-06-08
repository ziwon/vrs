# VRS Console Web UI

This is the `feat/web` VSS-themed dashboard integrated with the local FastAPI
backend. It is still zero-build: nginx or any static file server can serve it.

When the backend is available, `app.js` reads:

- `GET /api/health`
- `GET /api/runs`
- `GET /api/streams`
- `GET /api/runs/{run}/alerts`
- `GET /api/runs/{run}/thumbnails/{path}`

If the backend is unavailable, it falls back to embedded sample alerts.

## What's here

| File | Purpose |
| --- | --- |
| `index.html` | Page shell (Tailwind via CDN, VSS-style header + sidebar + main). |
| `app.js` | Tabs, filters, sample data, synthetic-keyframe SVGs, detail drawer. |
| `config.js` | Runtime API configuration. |
| `sample_alerts.jsonl` | Legacy fallback data. |

## Tabs

- **Live Alerts** — stat cards (candidates / confirmed / suppressed / flip-rate /
  avg confidence / streams) plus a filterable table with per-event keyframe,
  severity chip, VLM verdict, confidence bar, track id, and rationale. Click a row
  for the full record (raw verifier JSON, bbox, trajectory, thumbnail path).
- **Cascade** — the two-stage pipeline (Decode → YOLOE → Event-state → VLM verifier
  → Sink), the strict-JSON verifier contract, and the multi-stream threading model.
- **Streams** — per-camera latest keyframe with live status.
- **Watch Policy** — the `safety.yaml` classes: detector prompts, verifier sentence,
  severity, `min_score`, `min_persist_frames`.

## Run the full stack

From the repository root:

```bash
docker compose up --build
```

Open <http://127.0.0.1:5173>. The RTSP sample stream is published at
`rtsp://127.0.0.1:8554/falldown`.

For real inference on a GPU host:

```bash
export HF_TOKEN=hf_...  # required for the default gated Cosmos verifier
docker compose --profile inference up --build
```

The UI defaults to the `live` run when it exists. Fixture runs remain available
from the Run dropdown as demo/fallback data.

## Manual static serving

```bash
cd web
python -m http.server 5173
```

For manual serving, update `web/config.js` if the backend is not same-origin.

## Notes

- The NVIDIA wordmark is **not** bundled; the header uses a generic VRS glyph.
- The "Live updates" toggle polls the backend every ~3s when connected.
- Theme preference persists in `localStorage`.
- Deep-linkable: `?tab=cascade` / `?theme=light` (e.g. `index.html?tab=policy&theme=light`).
