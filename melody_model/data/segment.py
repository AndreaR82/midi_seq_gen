"""Slice extracted melodic lines into short, filtered, deduplicated,
key-augmented fragments ready for tokenizing.
"""

from __future__ import annotations

from ..notes import NoteEvent


def segment_into_fragments(
    notes: list[NoteEvent],
    bar_options: tuple[int, ...] = (2, 4, 8),
    steps_per_bar: int = 16,
) -> list[tuple[int, list[NoteEvent]]]:
    """Non-overlapping, bar-aligned windows at each length in bar_options.
    Notes are re-based to start at 0 within each window; a note straddling
    a window boundary is cut short in the earlier window and dropped from
    the next (its onset is what matters musically here, not a tied-over
    fragment)."""
    if not notes:
        return []
    span_end = max(n.end_step for n in notes)

    fragments: list[tuple[int, list[NoteEvent]]] = []
    for bars in bar_options:
        window = bars * steps_per_bar
        start = 0
        while start < span_end:
            frag = [
                NoteEvent(n.pitch, n.start_step - start, min(n.dur_step, start + window - n.start_step))
                for n in notes
                if start <= n.start_step < start + window
            ]
            if frag:
                fragments.append((bars, frag))
            start += window
    return fragments


def passes_filters(
    notes: list[NoteEvent],
    bars: int,
    steps_per_bar: int,
    min_notes: int = 4,
    min_distinct_pitches: int = 2,
    max_rest_ratio: float = 0.6,
) -> bool:
    """Cheap sanity filters — cuts fragments that are too sparse to be a
    real phrase, or so static they're one repeated note."""
    if len(notes) < min_notes:
        return False
    if len({n.pitch for n in notes}) < min_distinct_pitches:
        return False
    total_steps = bars * steps_per_bar
    covered = sum(n.dur_step for n in notes)
    rest_ratio = 1 - covered / total_steps
    return rest_ratio <= max_rest_ratio


def contour_signature(notes: list[NoteEvent]) -> tuple:
    """A transposition-invariant fingerprint (pitch relative to the first
    note, plus timing) used to dedupe fragments that are the same phrase
    played in a different key or found via an overlapping window."""
    if not notes:
        return ()
    base = notes[0].pitch
    return tuple((n.pitch - base, n.start_step, n.dur_step) for n in notes)


def transpose(notes: list[NoteEvent], semitones: int, pitch_min: int, pitch_max: int) -> list[NoteEvent] | None:
    """Returns None (rather than clamping) if the shift would push any note
    outside the tokenizer's pitch range — clamping would silently distort
    the melody's contour, which defeats the point of augmentation."""
    if semitones == 0:
        return list(notes)
    out = []
    for n in notes:
        pitch = n.pitch + semitones
        if pitch < pitch_min or pitch > pitch_max:
            return None
        out.append(NoteEvent(pitch, n.start_step, n.dur_step))
    return out
