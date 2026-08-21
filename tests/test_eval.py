"""Musical sanity metrics. Val loss cannot see any of these failure modes,
which is the whole reason melody_model/eval.py exists."""

from melody_model.eval import (
    actual_bars,
    aggregate,
    body_ngrams,
    build_ngram_index,
    copy_rate,
    in_key_ratio,
    metrics_for,
)
from melody_model.notes import NoteEvent

STEPS_PER_BAR = 16


def c_major_arpeggio() -> list[NoteEvent]:
    return [
        NoteEvent(pitch=60, start_step=0, dur_step=4),
        NoteEvent(pitch=64, start_step=4, dur_step=4),
        NoteEvent(pitch=67, start_step=8, dur_step=4),
        NoteEvent(pitch=72, start_step=12, dur_step=4),
    ]


def test_in_key_ratio_perfect_for_diatonic_material():
    assert in_key_ratio(c_major_arpeggio()) == 1.0


def test_in_key_ratio_bottoms_out_on_chromatic_material():
    chromatic = [NoteEvent(pitch=60 + i, start_step=i, dur_step=1) for i in range(12)]
    # Any 7-note scale covers exactly 7 of the 12 pitch classes.
    assert in_key_ratio(chromatic) == 7 / 12


def test_in_key_ratio_finds_the_key_it_is_actually_in():
    # F# major, which shares no accidentals-free spelling with C.
    fsharp = [NoteEvent(pitch=66 + i, start_step=i * 2, dur_step=2) for i in (0, 2, 4, 6, 7, 9, 11)]
    assert in_key_ratio(fsharp) == 1.0


def test_actual_bars_measured_from_last_onset_not_last_end():
    # A note starting in bar 1 but ringing into bar 2 is overhang, not a bar --
    # the same rule midi_file_to_sequence applies.
    overhang = [NoteEvent(pitch=60, start_step=12, dur_step=32)]
    assert actual_bars(overhang, STEPS_PER_BAR) == 1
    # A note that genuinely starts in bar 3 does extend the span.
    late = [NoteEvent(pitch=60, start_step=32, dur_step=4)]
    assert actual_bars(late, STEPS_PER_BAR) == 3
    assert actual_bars([], STEPS_PER_BAR) == 0


def test_metrics_on_a_well_formed_bar():
    m = metrics_for(c_major_arpeggio(), 1, STEPS_PER_BAR)
    assert m["len_respected"] == 1.0
    assert m["len_overrun"] == 0.0
    assert m["notes"] == 4
    assert m["density_per_bar"] == 4.0
    assert m["rest_fraction"] == 0.0
    assert m["pitch_range"] == 12
    assert m["mean_ioi_steps"] == 4.0
    assert m["degenerate_empty"] == 0.0


def test_metrics_flag_len_overrun():
    notes = c_major_arpeggio() + [NoteEvent(pitch=60, start_step=48, dur_step=4)]
    m = metrics_for(notes, 2, STEPS_PER_BAR)
    assert m["bars_actual"] == 4
    assert m["len_overrun"] == 1.0
    assert m["len_respected"] == 0.0


def test_metrics_flag_degenerate_output():
    empty = metrics_for([], 4, STEPS_PER_BAR)
    assert empty["degenerate_empty"] == 1.0
    assert empty["degenerate_sparse"] == 1.0
    assert empty["in_key"] == 0.0

    one_note = [NoteEvent(pitch=60, start_step=i * 4, dur_step=4) for i in range(8)]
    stuck = metrics_for(one_note, 2, STEPS_PER_BAR)
    assert stuck["degenerate_single_pitch"] == 1.0
    assert stuck["degenerate_empty"] == 0.0


def test_rest_fraction_reflects_silence():
    sparse = [NoteEvent(pitch=60, start_step=0, dur_step=4)]
    assert metrics_for(sparse, 1, STEPS_PER_BAR)["rest_fraction"] == 0.75


def test_ngrams_skip_the_shared_bos_len_prefix():
    # Two fragments differing only in their LEN_ token share no body n-grams.
    a = [1, 5, 60, 84, 64, 84, 67, 84]
    b = [1, 9, 60, 84, 64, 84, 67, 84]
    assert body_ngrams(a, 4) == body_ngrams(b, 4)
    assert body_ngrams([1, 5, 60], 4) == set()


def test_copy_rate_detects_verbatim_recall():
    train = [[1, 7, 60, 84, 64, 84, 67, 84, 72, 84, 2]]
    index = build_ngram_index(train, 4)
    assert copy_rate(train[0], index, 4) == 1.0
    unrelated = [1, 7, 50, 90, 51, 90, 52, 90, 53, 90, 2]
    assert copy_rate(unrelated, index, 4) == 0.0
    # Partial overlap lands in between.
    half = [1, 7, 60, 84, 64, 84, 40, 90, 41, 90, 2]
    assert 0.0 < copy_rate(half, index, 4) < 1.0


def test_aggregate_means_ignore_the_requested_bar_length():
    rows = [metrics_for(c_major_arpeggio(), 1, STEPS_PER_BAR), metrics_for([], 1, STEPS_PER_BAR)]
    agg = aggregate(rows)
    assert "bars_requested" not in agg
    assert agg["degenerate_empty"] == 0.5
    assert agg["notes"] == 2.0
