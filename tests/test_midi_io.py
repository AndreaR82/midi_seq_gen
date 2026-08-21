"""Round-trip tests for the .mid reader.

The write path and the read path have to agree about ticks-vs-beats and about
the note_on-velocity-0 idiom; an off-by-one there silently distorts rhythm in a
way that is easy to blame on the model instead of on the plumbing.
"""

import math

import mido
import pytest

from melody_model.midi_io import (
    DEFAULT_TICKS_PER_BEAT,
    midi_file_to_sequence,
    notes_to_midi_file,
)
from melody_model.notes import NoteEvent

STEPS_PER_BEAT = 4  # 16th-note grid, the TokenizerConfig default


def test_round_trip_preserves_pitch_timing_and_tempo(tmp_path):
    notes = [
        NoteEvent(pitch=60, start_step=0, dur_step=4),  # beat 0, one beat long
        NoteEvent(pitch=67, start_step=4, dur_step=2),  # beat 1, half a beat
        NoteEvent(pitch=72, start_step=8, dur_step=8),  # beat 2, two beats
    ]
    path = tmp_path / "m.mid"
    notes_to_midi_file(notes, STEPS_PER_BEAT, bpm=140, path=path, velocity=96)

    seq = midi_file_to_sequence(path)

    assert seq.bpm == 140
    assert [n.pitch for n in seq.notes] == [60, 67, 72]
    assert [n.start_beat for n in seq.notes] == [0.0, 1.0, 2.0]
    assert [n.duration_beats for n in seq.notes] == [1.0, 0.5, 2.0]
    assert all(n.velocity == 96 for n in seq.notes)


def test_loop_bars_come_from_last_onset_not_overhang(tmp_path):
    """A final note ringing past the barline is overhang, not another bar --
    otherwise every loop gains a bar of silence."""
    notes = [NoteEvent(pitch=60, start_step=60, dur_step=8)]  # starts beat 15, ends 17
    path = tmp_path / "m.mid"
    notes_to_midi_file(notes, STEPS_PER_BEAT, bpm=120, path=path)

    assert midi_file_to_sequence(path).loop_bars == 4


def test_note_starting_in_a_later_bar_extends_the_loop(tmp_path):
    notes = [NoteEvent(pitch=60, start_step=68, dur_step=4)]  # starts beat 17 -> bar 5
    path = tmp_path / "m.mid"
    notes_to_midi_file(notes, STEPS_PER_BEAT, bpm=120, path=path)

    assert midi_file_to_sequence(path).loop_bars == 5


def test_missing_set_tempo_defaults_to_120(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=DEFAULT_TICKS_PER_BEAT))
    path = tmp_path / "no_tempo.mid"
    mid.save(str(path))

    assert midi_file_to_sequence(path).bpm == 120


def test_note_on_velocity_zero_is_treated_as_note_off(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=64, velocity=90, time=0))
    track.append(mido.Message("note_on", note=64, velocity=0, time=DEFAULT_TICKS_PER_BEAT * 2))
    path = tmp_path / "running_status.mid"
    mid.save(str(path))

    seq = midi_file_to_sequence(path)
    assert len(seq.notes) == 1
    assert seq.notes[0].duration_beats == 2.0


def test_multiple_tracks_and_channels_are_merged(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    for channel, pitch in ((0, 60), (3, 67)):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=pitch, velocity=80, channel=channel, time=0))
        track.append(
            mido.Message("note_off", note=pitch, velocity=0, channel=channel,
                         time=DEFAULT_TICKS_PER_BEAT)
        )
    path = tmp_path / "multi.mid"
    mid.save(str(path))

    seq = midi_file_to_sequence(path)
    # Channel is dropped -- Player assigns it at play time.
    assert sorted(n.pitch for n in seq.notes) == [60, 67]
    assert all(n.start_beat == 0.0 for n in seq.notes)


def test_over_long_file_raises_with_its_actual_length(tmp_path):
    notes = [NoteEvent(pitch=60, start_step=0, dur_step=4),
             NoteEvent(pitch=60, start_step=17 * 16, dur_step=4)]  # onset in bar 18
    path = tmp_path / "long.mid"
    notes_to_midi_file(notes, STEPS_PER_BEAT, bpm=120, path=path)

    with pytest.raises(ValueError, match="18 bars"):
        midi_file_to_sequence(path)


def test_empty_file_raises(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    mid.tracks.append(mido.MidiTrack())
    path = tmp_path / "empty.mid"
    mid.save(str(path))

    with pytest.raises(ValueError, match="no notes"):
        midi_file_to_sequence(path)


def test_zero_length_note_still_has_positive_duration(tmp_path):
    """duration_beats has a gt=0 constraint; a degenerate note must not blow up
    schema validation for the whole file."""
    mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=0))
    path = tmp_path / "zero.mid"
    mid.save(str(path))

    seq = midi_file_to_sequence(path)
    assert seq.notes[0].duration_beats > 0
