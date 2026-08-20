from melody_model.config import TokenizerConfig
from melody_model.notes import NoteEvent
from melody_model.tokenizer import MelodyTokenizer


def test_round_trip_simple_phrase():
    tok = MelodyTokenizer()
    notes = [
        NoteEvent(pitch=60, start_step=0, dur_step=4),
        NoteEvent(pitch=64, start_step=4, dur_step=4),
        NoteEvent(pitch=67, start_step=8, dur_step=8),
    ]
    ids = tok.encode(notes, num_bars=2)
    decoded, bars = tok.decode(ids)

    assert bars == 2
    assert decoded == notes


def test_round_trip_with_gaps_and_trailing_rest():
    tok = MelodyTokenizer()
    notes = [
        NoteEvent(pitch=48, start_step=2, dur_step=3),
        NoteEvent(pitch=50, start_step=10, dur_step=2),
    ]
    ids = tok.encode(notes, num_bars=1)
    decoded, bars = tok.decode(ids)

    assert bars == 1
    assert decoded == notes


def test_long_note_splits_and_remerges_on_decode():
    cfg = TokenizerConfig(max_event_steps=8)
    tok = MelodyTokenizer(cfg)
    notes = [NoteEvent(pitch=60, start_step=0, dur_step=20)]

    ids = tok.encode(notes, num_bars=2)
    # the 20-step note must have been split into steps <= max_event_steps
    dur_tokens = [tok.id_to_token[i] for i in ids if tok.id_to_token[i].startswith("DUR_")]
    assert all(int(d.split("_")[1]) <= 8 for d in dur_tokens)

    decoded, _ = tok.decode(ids)
    assert decoded == notes  # re-merged back into one note on decode


def test_pitch_out_of_range_is_clamped_not_dropped():
    tok = MelodyTokenizer()
    ids = tok.encode([NoteEvent(pitch=200, start_step=0, dur_step=4)], num_bars=1)
    decoded, _ = tok.decode(ids)
    assert decoded[0].pitch == tok.config.pitch_max


def test_sequence_shape_and_specials():
    tok = MelodyTokenizer()
    ids = tok.encode([NoteEvent(pitch=60, start_step=0, dur_step=16)], num_bars=1)
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.len_token_id(1)
    assert ids[-1] == tok.eos_id


def test_decode_tolerates_truncated_tail():
    tok = MelodyTokenizer()
    ids = tok.encode([NoteEvent(pitch=60, start_step=0, dur_step=4)], num_bars=1)
    truncated = ids[:-1] + [tok.pitch_token_id(62)]  # dangling event, no DUR, no EOS
    decoded, _ = tok.decode(truncated)  # must not raise
    assert decoded[0].pitch == 60


def test_vocab_size_matches_construction():
    cfg = TokenizerConfig(pitch_min=36, pitch_max=96, max_event_steps=32, max_bars=16)
    tok = MelodyTokenizer(cfg)
    expected = 4 + cfg.max_bars + (cfg.pitch_max - cfg.pitch_min + 1) + cfg.max_event_steps
    assert tok.vocab_size == expected
