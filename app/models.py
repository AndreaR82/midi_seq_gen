"""Pydantic models: the note-event schema and request bodies."""

from pydantic import BaseModel, Field


class Note(BaseModel):
    pitch: int = Field(ge=0, le=127, description="MIDI note number")
    start_beat: float = Field(ge=0, description="0-based start, in quarter-note beats")
    duration_beats: float = Field(gt=0, description="length in quarter-note beats")
    velocity: int = Field(ge=1, le=127, description="MIDI velocity")


class Sequence(BaseModel):
    loop_bars: int = Field(ge=1, le=16, description="length of the loop in 4/4 bars")
    bpm: int = Field(ge=20, le=300)
    notes: list[Note]


class GenerateRequest(BaseModel):
    query: str
    bpm: int = Field(ge=20, le=300, default=130)
    key: str = "C minor"


class PlayRequest(BaseModel):
    channel: int = Field(ge=1, le=16, description="MIDI channel 1-16 (UI-facing)")
    port_name: str


class LoadMelodyRequest(BaseModel):
    path: str = Field(description="path relative to the melody root, from /melodies")
