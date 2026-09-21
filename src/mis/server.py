"""Local web server for the live signal strip.

Interviewer-only by construction: it binds to localhost and is meant for a
second monitor or a corner of your screen. Nothing here is reachable by the
candidate, which is the constraint the whole product rests on.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from .live.session import LiveSession

STATIC = Path(__file__).parent / "static"


def create_app(session: LiveSession, worker: threading.Thread | None = None) -> FastAPI:
    app = FastAPI(title="mock-interview-signals")
    history: list[dict] = []

    @app.on_event("startup")
    async def _start() -> None:
        if worker is not None and not worker.is_alive():
            worker.start()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "strip.html")

    @app.get("/events")
    async def events() -> StreamingResponse:
        async def stream():
            # Replay what already happened so a late-opened tab is not blank.
            for past in list(history):
                yield f"data: {json.dumps(past)}\n\n"
            while True:
                try:
                    payload = session.events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.25)
                    yield ": keepalive\n\n"
                    continue
                history.append(payload)
                if len(history) > 400:
                    del history[:100]
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/stop")
    async def stop() -> dict:
        session.stop()
        return {"stopped": True}

    return app
