import torch

from melody_model.config import ModelConfig, TokenizerConfig
from melody_model.model import MelodyTransformer
from melody_model.tokenizer import MelodyTokenizer


def _tiny_model():
    tok = MelodyTokenizer(TokenizerConfig())
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layer=2, n_head=2, d_ff=32, max_seq_len=64)
    model = MelodyTransformer(cfg, pad_id=tok.pad_id)
    return model, tok


def test_forward_shape_and_no_nans_with_padding():
    model, tok = _tiny_model()
    batch = torch.tensor([
        [tok.bos_id, tok.len_token_id(1), tok.pitch_token_id(60), tok.dur_token_id(4), tok.eos_id, tok.pad_id],
        [tok.bos_id, tok.len_token_id(2), tok.rest_id, tok.dur_token_id(16), tok.eos_id, tok.pad_id],
    ])
    logits = model(batch)
    assert logits.shape == (2, 6, tok.vocab_size)
    assert not torch.isnan(logits).any()


def test_generate_stops_at_eos_and_stays_in_vocab():
    model, tok = _tiny_model()
    gen = torch.Generator().manual_seed(0)
    ids = model.generate([tok.bos_id, tok.len_token_id(2)], max_new_tokens=80, eos_id=tok.eos_id, generator=gen)
    assert all(0 <= i < tok.vocab_size for i in ids)
    # either it terminated with EOS, or it ran out of the token budget
    assert ids[-1] == tok.eos_id or len(ids) == 2 + 80


def test_generate_never_emits_forbidden_ids():
    model, tok = _tiny_model()
    forbidden = [tok.len_token_id(n) for n in range(1, tok.config.max_bars + 1)]
    gen = torch.Generator().manual_seed(1)
    ids = model.generate(
        [tok.bos_id, tok.len_token_id(4)], max_new_tokens=80, eos_id=tok.eos_id,
        generator=gen, forbidden_ids=forbidden,
    )
    generated_body = ids[2:]  # exclude the prefix, which legitimately contains a LEN_ token
    assert not any(i in forbidden for i in generated_body)


def test_same_seed_is_deterministic_different_seed_usually_differs():
    model, tok = _tiny_model()
    prefix = [tok.bos_id, tok.len_token_id(2)]

    gen_a = torch.Generator().manual_seed(7)
    out_a = model.generate(prefix, 40, tok.eos_id, temperature=1.0, generator=gen_a)
    gen_a2 = torch.Generator().manual_seed(7)
    out_a2 = model.generate(prefix, 40, tok.eos_id, temperature=1.0, generator=gen_a2)
    assert out_a == out_a2  # reproducible given the same seed

    gen_b = torch.Generator().manual_seed(8)
    out_b = model.generate(prefix, 40, tok.eos_id, temperature=1.0, generator=gen_b)
    assert out_a != out_b  # "press the button again" gives something different


def test_generate_is_robust_across_many_random_models_and_seeds():
    """Regression test: on a freshly (randomly) initialized model, top-p
    sampling occasionally hit a near-degenerate post-slice distribution
    (sums to ~0 / carries floating-point dust) and torch.multinomial raised.
    Hammer many independent models and seeds — including with forbidden_ids,
    which slices the distribution further — to guard against that class of
    crash recurring."""
    tok = MelodyTokenizer(TokenizerConfig())
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=16, n_layer=2, n_head=2, d_ff=32, max_seq_len=64)
    forbidden = [tok.len_token_id(n) for n in range(1, tok.config.max_bars + 1)]

    for seed in range(25):
        torch.manual_seed(seed)  # vary weight initialization
        model = MelodyTransformer(cfg, pad_id=tok.pad_id)
        gen = torch.Generator().manual_seed(seed)
        ids = model.generate(
            [tok.bos_id, tok.len_token_id(4)], max_new_tokens=60, eos_id=tok.eos_id,
            temperature=1.0, top_p=0.95, generator=gen, forbidden_ids=forbidden,
        )
        assert all(0 <= i < tok.vocab_size for i in ids)
