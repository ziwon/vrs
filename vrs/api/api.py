"""FastAPI app for local VRS run artifact browsing."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .artifacts import RunArtifactStore, UnsafePathError


def create_app(runs_root: str | Path | None = None) -> FastAPI:
    root = Path(runs_root or os.environ.get("VRS_RUNS_ROOT", "runs"))
    policy_path = Path(os.environ.get("VRS_POLICY_PATH", "configs/policies/safety.yaml"))
    rtsp_url = os.environ.get("VRS_SAMPLE_RTSP_URL", "rtsp://127.0.0.1:8554/sample")
    streams_path_raw = os.environ.get("VRS_STREAMS_PATH")
    streams_path = Path(streams_path_raw) if streams_path_raw else None
    store = RunArtifactStore(root)
    app = FastAPI(title="VRS Local Run Browser", version="0.1.0")
    app.state.store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "runs_root": str(store.runs_root)}

    @app.get("/api/runs")
    def list_runs() -> dict[str, object]:
        return {"runs": [asdict(run) for run in store.list_runs()]}

    @app.get("/api/streams")
    def list_streams() -> dict[str, object]:
        streams = _read_stream_manifest(streams_path, fallback_rtsp_url=rtsp_url)
        known_ids = {str(stream["id"]) for stream in streams}
        for run in store.list_runs():
            for stream_id in run.streams:
                if stream_id not in known_ids:
                    streams.append(
                        {
                            "id": stream_id,
                            "name": stream_id,
                            "location": run.name,
                            "status": "online",
                            "fps": 0,
                            "rtsp_url": None,
                        }
                    )
                    known_ids.add(stream_id)
        primary = streams[0] if streams else _default_stream(rtsp_url)
        return {
            "rtsp_sample": {
                "url": primary["rtsp_url"],
                "path": primary["id"],
                "description": "Primary local RTSP stream published through MediaMTX",
            },
            "streams": streams,
        }

    @app.get("/api/policy")
    def read_policy() -> dict[str, object]:
        try:
            policy = load_watch_policy(policy_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="policy file not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return policy

    @app.get("/api/runs/{run_name}/alerts")
    def read_alerts(
        run_name: str,
        class_name: str | None = None,
        severity: str | None = None,
        true_alert: Annotated[bool | None, Query()] = None,
        stream_id: str | None = None,
        limit: int = Query(default=100, ge=0, le=5000),
        offset: int = Query(default=0, ge=0),
        since_line: int | None = Query(default=None, ge=0),
    ) -> dict[str, object]:
        try:
            info = store.describe_run(run_name)
            result = store.read_alerts(
                run_name,
                class_name=class_name,
                severity=severity,
                true_alert=true_alert,
                stream_id=stream_id,
                limit=limit,
                offset=offset,
                since_line=since_line,
            )
        except UnsafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "run": asdict(info),
            "alerts": result.alerts,
            "errors": [asdict(error) for error in result.errors],
            "total": result.total,
            "limit": limit,
            "offset": offset,
            "since_line": since_line,
        }

    @app.get("/api/runs/{run_name}/tail")
    def tail_alerts(
        run_name: str,
        cursor: str | None = None,
        since_line: int | None = Query(default=None, ge=0),
        stream_id: str | None = None,
        limit: int = Query(default=100, ge=0, le=5000),
        mode: str = Query(default="poll", pattern="^(poll|latest)$"),
    ) -> dict[str, object]:
        try:
            info = store.describe_run(run_name)
            result = store.tail_alerts(
                run_name,
                cursor=cursor,
                since_line=since_line,
                stream_id=stream_id,
                limit=limit,
                mode=mode,
            )
        except UnsafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "run": asdict(info),
            "alerts": result.alerts,
            "errors": [asdict(error) for error in result.errors],
            "total": result.total,
            "limit": limit,
            "cursor": cursor,
            "mode": mode,
            "next_cursor": result.next_cursor,
        }

    @app.get("/api/runs/{run_name}/thumbnails/{thumbnail_path:path}")
    def read_thumbnail(
        run_name: str,
        thumbnail_path: str,
        stream_id: str | None = None,
    ) -> FileResponse:
        try:
            path = store.thumbnail_path(run_name, thumbnail_path, stream_id=stream_id)
        except UnsafePathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="thumbnail not found")
        return FileResponse(path)

    return app


app = create_app()


def load_watch_policy(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required to read watch policy files") from exc

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("policy file must contain a mapping")
    watch = raw.get("watch", [])
    if not isinstance(watch, list):
        raise ValueError("policy watch must be a list")
    policies: list[dict[str, Any]] = []
    for item in watch:
        if not isinstance(item, dict):
            raise ValueError("policy watch entries must be mappings")
        policies.append(dict(item))
    return {"path": str(path), "watch": policies}


def _default_stream(rtsp_url: str) -> dict[str, object]:
    return {
        "id": "falldown",
        "name": "Falldown RTSP",
        "location": "local",
        "status": "online",
        "fps": 30,
        "rtsp_url": rtsp_url,
    }


def _read_stream_manifest(path: Path | None, *, fallback_rtsp_url: str) -> list[dict[str, object]]:
    if path is None or not path.is_file():
        return [_default_stream(fallback_rtsp_url)]
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise ValueError("PyYAML is required to read stream manifest files") from exc

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    manifest_streams = raw.get("streams") if isinstance(raw, dict) else None
    if not isinstance(manifest_streams, list):
        return [_default_stream(fallback_rtsp_url)]

    streams: list[dict[str, object]] = []
    for item in manifest_streams:
        if not isinstance(item, dict) or "id" not in item:
            continue
        stream_id = str(item["id"])
        rtsp = item.get("rtsp") or item.get("rtsp_url")
        streams.append(
            {
                "id": stream_id,
                "name": str(item.get("name") or stream_id),
                "location": str(item.get("location") or "stream-manifest"),
                "status": "online",
                "fps": int(item.get("fps") or 0),
                "rtsp_url": str(rtsp) if rtsp else None,
            }
        )
    return streams or [_default_stream(fallback_rtsp_url)]
