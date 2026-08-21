"""Musical sanity metrics for a trained checkpoint.

Validation loss says how well the model predicts the next token. It says
nothing about whether the output is *music*, and nothing at all about whether
the model is simply reciting its training set -- which is the live risk here,
because the corpus holds only ~52k unique phrases and the model is far larger
than that supports.

So this module generates samples and measures them, and measures the same
things on held-out real music so every number has something to be compared
against. Run it on two checkpoints to compare them:

    python -m melody_model.eval --checkpoint checkpoints/wave2/best.pt \\
        --dataset data/fragments.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import TokenizerConfig
from .notes import NoteEvent
from .split import split_records
from .tokenizer import MelodyTokenizer

MAJOR = frozenset({0, 2, 4, 5, 7, 9, 11})
MINOR = frozenset({0, 2, 3, 5, 7, 8, 10})  # natural minor


# -- single-sample metrics --------------------------------------------------

def actual_bars(notes: list[NoteEvent], steps_per_bar: int) -> int:
    """Bars spanned, measured from the last note *onset* -- the same rule
    midi_file_to_sequence uses, so this number matches what the app will
    report when the file is loaded."""
    if not notes:
        return 0
    return max(n.start_step for n in notes) // steps_per_bar + 1


def in_key_ratio(notes: list[NoteEvent]) -> float:
    """Best-fit diatonic key over all 24 major/minor candidates, reported as
    the fraction of notes inside it. Real melodies sit high (chromatic passing
    tones aside); a model producing random pitches sits near 7/12 = 0.58."""
    if not notes:
        return 0.0
    pcs = [n.pitch % 12 for n in notes]
    best = 0.0
    for root in range(12):
        for scale in (MAJOR, MINOR):
            hits = sum(1 for pc in pcs if (pc - root) % 12 in scale)
            best = max(best, hits / len(pcs))
    return best


def metrics_for(notes: list[NoteEvent], requested_bars: int, steps_per_bar: int) -> dict:
    span_bars = actual_bars(notes, steps_per_bar)
    total_steps = max(1, requested_bars * steps_per_bar)
    sounding = sum(n.dur_step for n in notes)
    pitches = [n.pitch for n in notes]
    onsets = sorted(n.start_step for n in notes)
    iois = [b - a for a, b in zip(onsets, onsets[1:])]

    return {
        "notes": len(notes),
        "bars_requested": requested_bars,
        "bars_actual": span_bars,
        "len_respected": float(span_bars == requested_bars),
        "len_overrun": float(span_bars > requested_bars),
        "in_key": in_key_ratio(notes),
        "density_per_bar": len(notes) / requested_bars,
        "rest_fraction": max(0.0, 1.0 - sounding / total_steps),
        "mean_pitch": sum(pitches) / len(pitches) if pitches else 0.0,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else 0.0,
        "mean_duration_steps": sounding / len(notes) if notes else 0.0,
        "mean_ioi_steps": sum(iois) / len(iois) if iois else 0.0,
        # Degeneracy: the failure modes an undertrained model actually shows.
        "degenerate_empty": float(not notes),
        "degenerate_single_pitch": float(bool(notes) and len(set(pitches)) == 1),
        "degenerate_sparse": float(len(notes) < requested_bars),  # under one note per bar
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "bars_requested"]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


# -- memorization -----------------------------------------------------------

def body_ngrams(tokens: list[int], n: int, prefix_len: int = 2) -> set[tuple]:
    """N-grams of the token body, skipping the BOS/LEN prefix (which every
    sequence shares and which would otherwise inflate every overlap)."""
    body = tokens[prefix_len:]
    if len(body) < n:
        return set()
    return {tuple(body[i : i + n]) for i in range(len(body) - n + 1)}


def build_ngram_index(token_seqs: list[list[int]], n: int) -> set[int]:
    """Hashed n-gram index. Stores hashes rather than tuples because the full
    training set runs to tens of millions of n-grams and the tuples alone
    would not fit comfortably in memory."""
    index: set[int] = set()
    for tokens in token_seqs:
        index.update(hash(g) for g in body_ngrams(tokens, n))
    return index


def copy_rate(tokens: list[int], index: set[int], n: int) -> float:
    """Fraction of a sequence's n-grams that appear verbatim in the index."""
    grams = body_ngrams(tokens, n)
    if not grams:
        return 0.0
    return sum(1 for g in grams if hash(g) in index) / len(grams)


# -- report -----------------------------------------------------------------

FIELDS = [
    ("len_respected", "LEN_ respected", "{:.1%}"),
    ("len_overrun", "  ...overran", "{:.1%}"),
    ("in_key", "in-key ratio", "{:.1%}"),
    ("notes", "notes/sample", "{:.1f}"),
    ("density_per_bar", "notes/bar", "{:.2f}"),
    ("rest_fraction", "rest fraction", "{:.1%}"),
    ("mean_pitch", "mean pitch", "{:.1f}"),
    ("pitch_range", "pitch range", "{:.1f}"),
    ("mean_duration_steps", "mean note len", "{:.2f}"),
    ("mean_ioi_steps", "mean IOI", "{:.2f}"),
    ("degenerate_empty", "degenerate: empty", "{:.1%}"),
    ("degenerate_single_pitch", "degenerate: 1 pitch", "{:.1%}"),
    ("degenerate_sparse", "degenerate: sparse", "{:.1%}"),
    ("copy_rate", "copy rate vs train", "{:.1%}"),
]


def print_report(generated: dict, corpus: dict, n: int) -> None:
    print(f"\n{'metric':<24}{'generated':>12}{'held-out corpus':>18}")
    print("-" * 54)
    for key, label, fmt in FIELDS:
        gen = fmt.format(generated[key]) if key in generated else "-"
        cor = fmt.format(corpus[key]) if key in corpus else "-"
        print(f"{label:<24}{gen:>12}{cor:>18}")
    print(
        f"\ncopy rate = share of {n}-token windows found verbatim in the training split.\n"
        "The corpus column is real music the model never saw, so it is the floor:\n"
        "generated output well above it is reciting rather than composing."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100, help="samples per bar length")
    parser.add_argument("--bars", default="2,4,8", help="comma-separated bar lengths to test")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ngram", type=int, default=16, help="window size for the copy-rate check")
    parser.add_argument("--index-size", type=int, default=200_000,
                        help="training records sampled to build the copy-rate index")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0,
                        help="must match the training run's --seed to reproduce its split")
    parser.add_argument("--json", type=Path, default=None, help="also write the raw numbers here")
    args = parser.parse_args()

    # Imported here so the metric functions above stay importable without torch.
    from .sample import generate_one, load_checkpoint

    model, tokenizer = load_checkpoint(args.checkpoint)
    steps_per_bar = tokenizer.config.steps_per_bar
    bar_options = [int(b) for b in args.bars.split(",")]

    ckpt_meta = _checkpoint_meta(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    for key in ("num_params", "val_loss", "epoch", "git_commit", "dataset_path"):
        if key in ckpt_meta:
            print(f"  {key}: {ckpt_meta[key]}")
    if "val_loss" not in ckpt_meta:
        print("  (no provenance recorded — pre-Wave-2 checkpoint)")

    records = _load_records(args.dataset)
    train_records, val_records, split_stats = split_records(records, args.val_fraction, salt=str(args.split_seed))
    print(
        f"\nsplit: {split_stats['train_files']} train files / {split_stats['val_files']} val files, "
        f"{len(train_records):,} / {len(val_records):,} records"
    )

    rng = random.Random(args.seed)
    index_sample = train_records if len(train_records) <= args.index_size else rng.sample(train_records, args.index_size)
    print(f"building {args.ngram}-gram index over {len(index_sample):,} training records ...")
    index = build_ngram_index([r["tokens"] for r in index_sample], args.ngram)
    print(f"  {len(index):,} distinct windows")

    # Generated samples.
    gen_rows = []
    for bars in bar_options:
        print(f"generating {args.count} x {bars}-bar samples ...")
        for i in range(args.count):
            notes, _ = generate_one(model, tokenizer, bars, args.temperature, args.top_p, args.seed + i)
            row = metrics_for(notes, bars, steps_per_bar)
            row["copy_rate"] = copy_rate(tokenizer.encode(notes, bars), index, args.ngram)
            gen_rows.append(row)

    # The same measurements on held-out real music, as the reference column.
    cor_rows = []
    for record in (val_records if len(val_records) <= 2000 else rng.sample(val_records, 2000)):
        notes, bars = tokenizer.decode(record["tokens"])
        row = metrics_for(notes, max(1, bars), steps_per_bar)
        row["copy_rate"] = copy_rate(record["tokens"], index, args.ngram)
        cor_rows.append(row)

    generated, corpus = aggregate(gen_rows), aggregate(cor_rows)
    print_report(generated, corpus, args.ngram)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "checkpoint_meta": ckpt_meta,
            "settings": vars(args) | {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
                                      "json": str(args.json)},
            "generated": generated,
            "corpus": corpus,
        }, indent=2, default=str))
        print(f"\nwrote {args.json}")


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _checkpoint_meta(path: Path) -> dict:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in ckpt.items() if k != "model_state"}


if __name__ == "__main__":
    main()
