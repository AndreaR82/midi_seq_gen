"""The internal note representation shared by extraction, segmentation,
tokenization, and sampling.

Everything here lives on an integer step grid (16th notes by default — see
TokenizerConfig.steps_per_beat) rather than float beats, because the whole
pipeline from MIDI-in to tokens-out is grid-quantized. Conversion to the
live app's beat-based `Sequence`/`Note` schema (app/models.py) happens only
at the edges, in midi_io.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoteEvent:
    """A single monophonic note on the step grid."""

    pitch: int  # MIDI note number
    start_step: int  # 0-based, in grid steps
    dur_step: int  # length in grid steps, >= 1

    @property
    def end_step(self) -> int:
        return self.start_step + self.dur_step


def is_monophonic(notes: list[NoteEvent]) -> bool:
    """True if no two notes overlap in time (once sorted by start)."""
    ordered = sorted(notes, key=lambda n: n.start_step)
    for a, b in zip(ordered, ordered[1:]):
        if b.start_step < a.end_step:
            return False
    return True
