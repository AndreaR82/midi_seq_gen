from pathlib import Path

import mido
import pytest

from melody_model.config import TokenizerConfig
from melody_model.data.extract import extract_melodic_lines, monophony_ratio, skyline
from melody_model.data.prepare_dataset import build_dataset
from melody_model.notes import NoteEvent


def test_skyline_keeps_higher_overlapping_note():
    notes = [
        NoteEvent(pitch=60, start_step=0, dur_step=8),
        NoteEvent(pitch=67, start_step=4, dur_step=4),  # overlaps, higher -> should win from step 4
    ]
    out = skyline(notes)
    assert out == [
        NoteEvent(pitch=60, start_step=0, dur_step=4),
        NoteEvent(pitch=67, start_step=4, dur_step=4),
    ]


def test_skyline_drops_lower_overlapping_note():
    notes = [
        NoteEvent(pitch=67, start_step=0, dur_step=8),
        NoteEvent(pitch=60, start_step=2, dur_step=4),  # fully hidden under the higher note
    ]
    out = skyline(notes)
    assert out == [NoteEvent(pitch=67, start_step=0, dur_step=8)]


def test_monophony_ratio_of_clean_line_is_one():
    notes = [NoteEvent(60, 0, 4), NoteEvent(62, 4, 4), NoteEvent(64, 8, 4)]
    assert monophony_ratio(notes) == 1.0


def _write_test_midi(path: Path) -> None:
    """One melodic channel (0) and one chordal/percussion-ish channel (9,
    skipped as percussion) so extraction has something to select between."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    pitches = [60, 62, 64, 65, 67, 69, 71, 72] * 2  # 16 notes, clearly melodic
    step_ticks = 480 // 4  # 16th note
    for p in pitches:
        track.append(mido.Message("note_on", channel=0, note=p, velocity=90, time=0))
        track.append(mido.Message("note_off", channel=0, note=p, velocity=0, time=step_ticks))

    # a percussion channel that should be ignored entirely
    for _ in range(8):
        track.append(mido.Message("note_on", channel=9, note=36, velocity=100, time=0))
        track.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=step_ticks))

    mid.save(str(path))


def test_extract_melodic_lines_skips_percussion_and_finds_melody(tmp_path):
    midi_path = tmp_path / "test.mid"
    _write_test_midi(midi_path)

    lines = extract_melodic_lines(midi_path, steps_per_beat=4, min_notes=8)
    assert len(lines) == 1  # only the melodic channel qualifies
    assert len(lines[0]) == 16
    assert lines[0][0].pitch == 60


def test_prepare_dataset_end_to_end(tmp_path):
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    _write_test_midi(midi_dir / "a.mid")
    _write_test_midi(midi_dir / "b.mid")

    cfg = TokenizerConfig()
    records, tokenizer = build_dataset(midi_dir, cfg, bar_options=(2,), transpose_range=(0, 0))

    assert len(records) > 0
    for r in records:
        notes, bars = tokenizer.decode(r["tokens"])
        assert bars == r["bars"]
        assert all(cfg.pitch_min <= n.pitch <= cfg.pitch_max for n in notes)


def test_transposition_augmentation_is_not_deduped_away(tmp_path):
    """Regression test: contour_signature is transposition-invariant, so
    deduping after transposing would collapse every augmented copy of a
    phrase into one. Each in-range transposition must survive."""
    _write_test_midi(tmp_path / "test.mid")
    cfg = TokenizerConfig()

    base_records, _ = build_dataset(tmp_path, cfg, bar_options=(2,), transpose_range=(0, 0))
    augmented_records, _ = build_dataset(tmp_path, cfg, bar_options=(2,), transpose_range=(-1, 1))

    assert len(augmented_records) == 3 * len(base_records)
