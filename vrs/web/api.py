"""FastAPI app for local VRS run artifact browsing."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .artifacts import RunArtifactStore, UnsafePathError


def create_app(runs_root: str | Path | None = None) -> FastAPI:
    root = Path(runs_root or os.environ.get("VRS_RUNS_ROOT", "runs"))
    rtsp_url = os.environ.get("VRS_SAMPLE_RTSP_URL", "rtsp://127.0.0.1:8554/sample")
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
        streams = [
            {
                "id": "falldown",
                "name": "Falldown RTSP",
                "location": "docker-compose",
                "status": "online",
                "fps": 30,
                "rtsp_url": rtsp_url,
            }
        ]
        for run in store.list_runs():
            for stream_id in run.streams:
                if stream_id != "falldown":
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
        return {
            "rtsp_sample": {
                "url": rtsp_url,
                "path": "falldown",
                "description": "Local falldown MP4 published through MediaMTX with normalized timestamps",
            },
            "streams": streams,
        }

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
        since_line: int = Query(default=0, ge=0),
        stream_id: str | None = None,
        limit: int = Query(default=100, ge=0, le=5000),
    ) -> dict[str, object]:
        return read_alerts(
            run_name, stream_id=stream_id, since_line=since_line, limit=limit, offset=0
        )

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
