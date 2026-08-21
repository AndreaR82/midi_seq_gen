"""The only place this package talks to concrete MIDI I/O or to app/: writing
a generated melody out as a standalone .mid file, or converting it into the
live app's beat-based Sequence schema so it can be looped straight through
app/player.py onto real hardware, same as a Claude-generated sequence.
"""

from __future__ import annotations

import math
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


MAX_LOOP_BARS = 16  # matches app.models.Sequence.loop_bars
BEATS_PER_BAR = 4  # 4/4 only, same assumption as app/player.py
DEFAULT_BPM = 120  # used when a file carries no set_tempo meta message


def midi_file_to_sequence(path: str | Path):
    """Read a .mid file back into the live app's beat-based Sequence schema.

    The inverse of notes_to_midi_file, but deliberately more tolerant: it also
    has to cope with files this package didn't write (a POP909 original dropped
    into generated/ to A/B against model output). All tracks are merged and the
    per-note channel is discarded — Player assigns the channel at play time.
    """
    from app.models import Note, Sequence

    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat or DEFAULT_TICKS_PER_BEAT

    bpm = None
    pending: dict[int, list[tuple[int, int]]] = {}  # pitch -> [(start_tick, velocity)]
    notes: list[tuple[int, int, int, int]] = []  # (start_tick, end_tick, pitch, velocity)

    tick = 0
    for msg in mido.merge_tracks(mid.tracks):
        tick += msg.time
        if msg.type == "set_tempo" and bpm is None:
            bpm = round(mido.tempo2bpm(msg.tempo))
        elif msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault(msg.note, []).append((tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            # A note_on with velocity 0 is the running-status idiom for note_off.
            stack = pending.get(msg.note)
            if stack:
                start_tick, velocity = stack.pop(0)  # FIFO: oldest open note wins
                notes.append((start_tick, tick, msg.note, velocity))

    # Notes left hanging (no matching off) end where the file does.
    for pitch, stack in pending.items():
        for start_tick, velocity in stack:
            notes.append((start_tick, tick, pitch, velocity))

    if not notes:
        raise ValueError(f"{Path(path).name} contains no notes")

    notes.sort(key=lambda n: (n[0], n[2]))
    min_dur = 1.0 / tpb  # a zero-length note still needs duration_beats > 0

    seq_notes = [
        Note(
            pitch=pitch,
            start_beat=round(start_tick / tpb, 6),
            duration_beats=round(max((end_tick - start_tick) / tpb, min_dur), 6),
            velocity=velocity,
        )
        for start_tick, end_tick, pitch, velocity in notes
    ]

    # Loop length is derived from the last note *onset*, not the last note end:
    # a final note that rings past the barline is overhang, not a whole extra
    # bar of music, and rounding it up would inject a bar of silence into every
    # loop. Player already truncates a note at the loop boundary, same as a
    # hardware sequencer. A note that genuinely *starts* in bar 5 does extend
    # the loop -- which is how an under-trained model overrunning its LEN_
    # token shows up here honestly, rather than being silently cropped.
    last_onset = max(n.start_beat for n in seq_notes)
    bars = max(1, math.floor(last_onset / BEATS_PER_BAR) + 1)
    if bars > MAX_LOOP_BARS:
        raise ValueError(
            f"{Path(path).name} is {bars} bars long; the loop schema caps at "
            f"{MAX_LOOP_BARS} bars"
        )

    # Sequence.bpm is bounded 20-300; a pathological tempo gets clamped rather
    # than blowing up on validation.
    bpm = min(300, max(20, bpm if bpm is not None else DEFAULT_BPM))

    return Sequence(loop_bars=bars, bpm=bpm, notes=seq_notes)
