"""Group-aware train/validation splitting.

`prepare_dataset` emits up to twelve transposed copies of every phrase as
separate records, and several overlapping fragments per source file. Splitting
records at random therefore puts near-identical material on both sides: a val
phrase whose rhythm and interval pattern also appear in training. Val loss then
measures recall, not generalization, and model selection on it is meaningless.

So the split is by **source file**. Every fragment and every transposition of a
file lands on one side. That is the strictest option available from the data we
keep, and it makes val loss "music this model has never seen in any form".

Deliberately torch-free so the split rule can be unit-tested on its own.
"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

RAW_DIR_NAME = "raw"


def corpus_of(source: str) -> str:
    """Which corpus a source file came from, for stratification.

    Paths look like `data/raw/pop909/211.mid`, so the component just after
    `raw/` is the corpus. Falls back to the containing directory's name for
    anything laid out differently (a one-off folder, a test fixture)."""
    parts = PurePosixPath(source).parts
    if RAW_DIR_NAME in parts:
        idx = parts.index(RAW_DIR_NAME)
        if idx + 1 < len(parts) - 1:  # must leave at least the filename behind
            return parts[idx + 1]
    return PurePosixPath(source).parent.name or "unknown"


def _stable_rank(key: str, salt: str) -> str:
    """A deterministic pseudo-random ordering key.

    Uses hashlib rather than the builtin `hash()`, which is salted per process
    (PYTHONHASHSEED) and would silently reshuffle the split on every run --
    making two training runs incomparable for reasons nobody would think to
    look for."""
    return hashlib.sha1(f"{salt}\x00{key}".encode()).hexdigest()


def split_sources(sources: list[str], val_fraction: float, salt: str = "") -> tuple[set[str], set[str]]:
    """Partition source paths into (train, val), holding out `val_fraction` of
    the files *within each corpus* so validation is not accidentally dominated
    by whichever corpus happens to be larger."""
    by_corpus: dict[str, list[str]] = {}
    for src in set(sources):
        by_corpus.setdefault(corpus_of(src), []).append(src)

    train: set[str] = set()
    val: set[str] = set()
    for corpus, files in sorted(by_corpus.items()):
        # Sorting by the hash gives an exact fraction per corpus, rather than
        # the approximate one a hash-threshold would give on small corpora.
        ordered = sorted(files, key=lambda s: _stable_rank(s, salt))
        n_val = int(len(ordered) * val_fraction)
        # Never let a corpus vanish from validation, but never take all of it
        # for validation either (a corpus of one file stays in training).
        if n_val == 0 and len(ordered) > 1:
            n_val = 1
        val.update(ordered[:n_val])
        train.update(ordered[n_val:])
    return train, val


def split_records(
    records: list[dict], val_fraction: float, salt: str = ""
) -> tuple[list[dict], list[dict], dict]:
    """Split records by their `source` file. Returns (train, val, stats)."""
    sources = [r["source"] for r in records]
    train_sources, val_sources = split_sources(sources, val_fraction, salt)

    train = [r for r in records if r["source"] in train_sources]
    val = [r for r in records if r["source"] in val_sources]

    per_corpus: dict[str, dict[str, int]] = {}
    for name, subset, srcs in (("train", train, train_sources), ("val", val, val_sources)):
        for r in subset:
            entry = per_corpus.setdefault(corpus_of(r["source"]), {"train": 0, "val": 0})
            entry[name] += 1

    stats = {
        "policy": "by_source_file",
        "salt": salt,
        "val_fraction": val_fraction,
        "train_files": len(train_sources),
        "val_files": len(val_sources),
        "train_records": len(train),
        "val_records": len(val),
        "records_per_corpus": per_corpus,
    }
    return train, val, stats
