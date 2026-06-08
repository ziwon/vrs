# VRS Console — sample web UI

A self-contained, zero-build dashboard that visualizes VRS cascade output, themed
after the **NVIDIA VSS Blueprint UI** (gray palette + `#76b900` green accent, 75px
header, 260px left sidebar, heavy-shadow main area, class-based dark/light mode).

It is a *demo skin*, not a wired service: it renders sample alerts shaped exactly
like `vrs/schemas.py::VerifiedAlert.to_json()` and the classes in
`configs/policies/safety.yaml`. No backend, no API, no build step.

## What's here

| File | Purpose |
| --- | --- |
| `index.html` | Page shell (Tailwind via CDN, VSS-style header + sidebar + main). |
| `app.js` | Tabs, filters, sample data, synthetic-keyframe SVGs, detail drawer. |
| `sample_alerts.jsonl` | Eight records in real `alerts.jsonl` format (loaded when served over HTTP). |

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

## Run it

Open directly:

```bash
open web/index.html          # macOS — uses the embedded sample data
```

Or serve it (also loads `sample_alerts.jsonl`, avoids `file://` fetch limits):

```bash
cd web && python -m http.server 8000
# → http://localhost:8000
```

## Point it at a real run

`app.js` fetches `./sample_alerts.jsonl` on load. To view a real run, copy (or
symlink) its alerts file next to `index.html`:

```bash
cp runs/<name>/alerts.jsonl web/sample_alerts.jsonl
# multi-stream: cat runs/<name>/*/alerts.jsonl > web/sample_alerts.jsonl
```

Each line must be one `VerifiedAlert.to_json()` record; `stream_id` and `ts` are
optional and default to `cam-01` / now if absent. Thumbnails are drawn synthetically
from `bbox_xywh_norm`, so no image files are required for the demo.

## Notes

- The NVIDIA wordmark is **not** bundled; the header uses a generic VRS glyph.
- The "Live updates" toggle in the sidebar simulates an incoming feed every ~3s.
- Theme preference persists in `localStorage`.
- Deep-linkable: `?tab=cascade` / `?theme=light` (e.g. `index.html?tab=policy&theme=light`).
