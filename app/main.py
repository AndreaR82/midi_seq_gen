"""FastAPI app: generate sequences via Claude, play them over MIDI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import generator, player
from .models import GenerateRequest, PlayRequest, Sequence

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="MIDI Sequence Generator")


@app.get("/ports")
def get_ports() -> dict:
    return {"ports": player.list_ports()}


@app.post("/generate", response_model=Sequence)
def generate(req: GenerateRequest) -> Sequence:
    try:
        seq = generator.generate_sequence(req)
    except Exception as exc:  # surface generation/parse failures to the UI
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}")
    player.player.set_sequence(seq)
    return seq


@app.post("/play")
def play(req: PlayRequest) -> dict:
    try:
        player.player.play(req.channel, req.port_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"playing": True}


@app.post("/stop")
def stop() -> dict:
    player.player.stop()
    return {"playing": False}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
