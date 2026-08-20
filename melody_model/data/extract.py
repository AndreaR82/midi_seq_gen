"""Pull monophonic melodic lines out of arbitrary MIDI files.

Two-step approach, both classic and deliberately simple (the literature on
melody extraction finds the simple methods hold up well against fancier
ones):

1. Group notes by channel and score each channel's "monophony ratio" — a
   channel that's already mostly one-note-at-a-time is more likely to be
   a lead/bass/arp line than a channel full of stacked chords.
2. Run a skyline pass on every channel that clears the bar, to force full
   monophony (keep the highest note whenever two overlap) — this both
   cleans up near-monophonic channels and lets genuinely polyphonic
   channels (e.g. a piano comping chords with an occasional top-line
   melody) still contribute a usable line.

Percussion (GM channel 10, i.e. index 9) is always skipped. Time signature
is ignored and 4/4 is assumed throughout the pipeline, matching app/player.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mido

from ..notes import NoteEvent

PERCUSSION_CHANNEL = 9  # GM channel 10, 0-indexed


def load_notes_by_channel(path: str | Path, steps_per_beat: int) -> dict[int, list[NoteEvent]]:
    """Read a MIDI file and bucket its notes by channel, on the step grid.
    A file's tracks are read independently (delta-times restart per track,
    but all tracks in a Standard MIDI File play back simultaneously from
    tick 0), then merged by channel."""
    mid = mido.MidiFile(str(path))
    ticks_per_beat = mid.ticks_per_beat or 480

    by_channel: dict[int, list[NoteEvent]] = defaultdict(list)
    for track in mid.tracks:
        abs_tick = 0
        open_notes: dict[tuple[int, int], int] = {}  # (channel, pitch) -> start_tick
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)] = abs_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                start_tick = open_notes.pop(key, None)
                if start_tick is None:
                    continue  # note_off with no matching note_on — malformed file, skip
                start_step = round(start_tick / ticks_per_beat * steps_per_beat)
                end_step = round(abs_tick / ticks_per_beat * steps_per_beat)
                if end_step <= start_step:
                    continue  # zero-length note, e.g. rounding collision
                by_channel[msg.channel].append(
                    NoteEvent(pitch=msg.note, start_step=start_step, dur_step=end_step - start_step)
                )

    for notes in by_channel.values():
        notes.sort(key=lambda n: n.start_step)
    return dict(by_channel)


def monophony_ratio(notes: list[NoteEvent]) -> float:
    """Fraction of adjacent-in-time note pairs that don't overlap. 1.0 for
    an already-monophonic line, lower for chord-heavy channels."""
    if len(notes) < 2:
        return 1.0
    overlaps = sum(1 for a, b in zip(notes, notes[1:]) if b.start_step < a.end_step)
    return 1 - overlaps / (len(notes) - 1)


def skyline(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Force monophony: sweep left to right, and whenever a new note starts
    before the current one ends, keep whichever is higher-pitched (trimming
    the lower one out of the overlap, or dropping the new note entirely if
    it's the lower one)."""
    ordered = sorted(notes, key=lambda n: (n.start_step, -n.pitch))
    out: list[NoteEvent] = []
    for n in ordered:
        if not out or n.start_step >= out[-1].end_step:
            out.append(NoteEvent(n.pitch, n.start_step, n.dur_step))
        elif n.pitch > out[-1].pitch:
            out[-1] = NoteEvent(out[-1].pitch, out[-1].start_step, n.start_step - out[-1].start_step)
            out.append(NoteEvent(n.pitch, n.start_step, n.dur_step))
        # else: lower/equal pitch hidden under a note already kept — drop it
    return [n for n in out if n.dur_step > 0]


def extract_melodic_lines(
    path: str | Path,
    steps_per_beat: int = 4,
    min_notes: int = 8,
    min_monophony: float = 0.5,
) -> list[list[NoteEvent]]:
    """Return one skylined, monophonic note list per channel that looks
    plausibly melodic (enough notes, not too chordal, not percussion).
    A single file can yield several lines (e.g. lead + bass), which is
    useful — more usable training fragments per song."""
    by_channel = load_notes_by_channel(path, steps_per_beat)
    lines = []
    for channel, notes in by_channel.items():
        if channel == PERCUSSION_CHANNEL or len(notes) < min_notes:
            continue
        if monophony_ratio(notes) < min_monophony:
            continue
        lines.append(skyline(notes))
    return lines
