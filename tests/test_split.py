"""The train/val split is the reason Wave 1's validation number was unusable:
records were shuffled and sliced, but prepare_dataset emits up to twelve
transposed copies of each phrase, so near-identical material landed on both
sides. These tests pin the fix down."""

import subprocess
import sys

from melody_model.split import corpus_of, split_records, split_sources


def _records(corpus: str, n_files: int, per_file: int = 12) -> list[dict]:
    """`per_file` stands in for the transposed/overlapping copies that
    prepare_dataset emits from a single source file."""
    return [
        {"source": f"data/raw/{corpus}/{i}.mid", "tokens": [1, 7, 60 + j, 84, 2], "bars": 2}
        for i in range(n_files)
        for j in range(per_file)
    ]


def test_no_source_file_appears_on_both_sides():
    records = _records("pop909", 100) + _records("nottingham", 40)
    train, val, _ = split_records(records, 0.1)

    train_sources = {r["source"] for r in train}
    val_sources = {r["source"] for r in val}
    assert train_sources & val_sources == set()
    # ...and nothing was silently dropped on the way through.
    assert len(train) + len(val) == len(records)


def test_every_copy_of_a_phrase_travels_with_its_file():
    records = _records("pop909", 20, per_file=12)
    train, val, _ = split_records(records, 0.25)

    for subset in (train, val):
        by_source = {}
        for r in subset:
            by_source.setdefault(r["source"], 0)
            by_source[r["source"]] += 1
        # A file is either wholly present with all 12 copies, or wholly absent.
        assert set(by_source.values()) == {12} or not by_source


def test_split_is_stratified_by_corpus():
    records = _records("pop909", 100) + _records("nottingham", 40)
    _, _, stats = split_records(records, 0.1)

    assert stats["val_files"] == 14  # 10 of 100 + 4 of 40, not 14 drawn from the pool at large
    per_corpus = stats["records_per_corpus"]
    assert per_corpus["pop909"]["val"] == 10 * 12
    assert per_corpus["nottingham"]["val"] == 4 * 12


def test_split_is_deterministic_across_processes():
    """Guards the use of hashlib over the builtin hash(), which is salted per
    process — with hash() two runs would silently get different splits and be
    quietly incomparable."""
    script = (
        "from melody_model.split import split_sources;"
        "s=[f'data/raw/pop909/{i}.mid' for i in range(50)];"
        "print(sorted(split_sources(s, 0.2)[1]))"
    )
    runs = [
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_salt_changes_the_split():
    sources = [f"data/raw/pop909/{i}.mid" for i in range(50)]
    assert split_sources(sources, 0.2, salt="0")[1] != split_sources(sources, 0.2, salt="1")[1]


def test_small_corpus_still_contributes_validation():
    # A three-file corpus would round to zero val files; it should still give one.
    _, val, stats = split_records(_records("tiny", 3) + _records("pop909", 100), 0.1)
    assert stats["records_per_corpus"]["tiny"]["val"] == 12
    # ...but a single-file corpus must not be taken entirely for validation.
    _, _, stats = split_records(_records("solo", 1) + _records("pop909", 100), 0.1)
    assert stats["records_per_corpus"]["solo"]["val"] == 0


def test_corpus_of_handles_nesting_and_odd_layouts():
    assert corpus_of("data/raw/pop909/211.mid") == "pop909"
    assert corpus_of("data/raw/lakh/a/b/c.mid") == "lakh"
    assert corpus_of("somewhere/else/tune.mid") == "else"


def test_token_budget_caps_batch_memory():
    """A fixed batch size OOMs on long fragments and wastes memory on short
    ones, because attention costs B x H x T x T. Batching to a token budget
    makes B fall out of T instead."""
    from melody_model.train import LengthBucketBatchSampler

    lengths = [20] * 500 + [259] * 500
    sampler = LengthBucketBatchSampler(lengths, batch_size=128, max_tokens=8192)

    for batch in sampler:
        width = max(lengths[i] for i in batch)
        assert len(batch) * width <= 8192 or len(batch) == 1
        assert len(batch) <= 128
    # Every fragment is used exactly once per epoch.
    assert sorted(i for b in sampler for i in b) == list(range(1000))


def test_token_budget_gives_long_fragments_smaller_batches():
    from melody_model.train import LengthBucketBatchSampler

    lengths = [20] * 500 + [250] * 500
    sampler = LengthBucketBatchSampler(lengths, batch_size=128, max_tokens=8192)
    short = [len(b) for b in sampler if max(lengths[i] for i in b) == 20]
    long_ = [len(b) for b in sampler if max(lengths[i] for i in b) == 250]
    assert min(short) > max(long_)


def test_sampler_length_matches_what_it_yields():
    from melody_model.train import LengthBucketBatchSampler

    sampler = LengthBucketBatchSampler([37] * 1000, batch_size=128, max_tokens=8192)
    assert len(sampler) == len(list(sampler))
    # set_epoch reshuffles but still covers everything exactly once.
    sampler.set_epoch(1)
    assert sorted(i for b in sampler for i in b) == list(range(1000))
