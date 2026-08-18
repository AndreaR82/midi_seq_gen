"""The only place this package talks to concrete MIDI I/O or to app/: writing
a generated melody out as a standalone .mid file, or converting it into the
live app's beat-based Sequence schema so it can be looped straight through
app/player.py onto real hardware, same as a Claude-generated sequence.
"""

from __future__ import annotations

from pathlib import Path

import mido

from .notes import NoteEvent

DEFAULT_TICKS_PER_BEAT = 480


def notes_to_midi_file(
    notes: list[NoteEvent],
    steps_per_beat: int,
    bpm: int,
    path: str | Path,
    velocity: int = 96,
) -> None:
    mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    events: list[tuple[int, mido.Message]] = []
    for n in notes:
        start_tick = round(n.start_step / steps_per_beat * DEFAULT_TICKS_PER_BEAT)
        end_tick = round((n.start_step + n.dur_step) / steps_per_beat * DEFAULT_TICKS_PER_BEAT)
        events.append((start_tick, mido.Message("note_on", note=n.pitch, velocity=velocity)))
        events.append((max(end_tick, start_tick + 1), mido.Message("note_off", note=n.pitch, velocity=0)))
    events.sort(key=lambda e: e[0])

    last_tick = 0
    for tick, msg in events:
        track.append(msg.copy(time=tick - last_tick))
        last_tick = tick

    mid.save(str(path))


def notes_to_sequence(notes: list[NoteEvent], steps_per_beat: int, bpm: int, bars: int, velocity: int = 96):
    """Convert to app.models.Sequence. Imported lazily so melody_model has
    no hard dependency on the app/ package (e.g. fastapi) just to train or
    write plain .mid files."""
    from app.models import Note, Sequence

    return Sequence(
        loop_bars=bars,
        bpm=bpm,
        notes=[
            Note(
                pitch=n.pitch,
                start_beat=n.start_step / steps_per_beat,
                duration_beats=n.dur_step / steps_per_beat,
                velocity=velocity,
            )
            for n in notes
        ],
    )
