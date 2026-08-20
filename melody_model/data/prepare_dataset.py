"""CLI: turn a directory of raw MIDI files into tokenized training fragments.

    python -m melody_model.data.prepare_dataset \\
        --input-dir /path/to/midi_collection \\
        --output data/fragments.jsonl

Recurses through --input-dir for .mid/.midi files, extracts monophonic
melodic lines from each, slices them into fragments, filters and
transposition-augments them, dedupes by melodic contour, and writes one
JSON object per line: {"tokens": [...], "bars": N, "source": "path"}.

Writes a matching <output>.meta.json with the TokenizerConfig used, so
train.py can rebuild the exact same tokenizer later without guessing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..config import TokenizerConfig
from ..tokenizer import MelodyTokenizer
from .extract import extract_melodic_lines
from .segment import contour_signature, passes_filters, segment_into_fragments, transpose

MIDI_EXTENSIONS = (".mid", ".midi")


def find_midi_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in MIDI_EXTENSIONS)


def build_dataset(
    input_dir: Path,
    cfg: TokenizerConfig,
    bar_options: tuple[int, ...] = (2, 4, 8),
    transpose_range: tuple[int, int] = (-5, 6),
    seed: int = 0,
) -> tuple[list[dict], MelodyTokenizer]:
    tokenizer = MelodyTokenizer(cfg)
    seen_contours: set[tuple] = set()
    records: list[dict] = []
    skipped = 0

    for path in find_midi_files(input_dir):
        try:
            lines = extract_melodic_lines(path, steps_per_beat=cfg.steps_per_beat)
        except Exception as exc:  # malformed MIDI files are common in the wild
            print(f"  skip {path}: {exc}")
            skipped += 1
            continue

        for line in lines:
            for bars, frag in segment_into_fragments(line, bar_options, cfg.steps_per_bar):
                if not passes_filters(frag, bars, cfg.steps_per_bar):
                    continue
                # Dedupe on the untransposed phrase — contour_signature is
                # itself transposition-invariant, so checking it *after*
                # transposing would treat every augmented copy of the same
                # phrase as a duplicate of the first and throw the
                # augmentation away. Once a phrase is accepted, all its
                # in-range transpositions are kept.
                sig = contour_signature(frag)
                if sig in seen_contours:
                    continue
                seen_contours.add(sig)

                for semitones in range(transpose_range[0], transpose_range[1] + 1):
                    shifted = transpose(frag, semitones, cfg.pitch_min, cfg.pitch_max)
                    if shifted is None:
                        continue
                    records.append({
                        "tokens": tokenizer.encode(shifted, bars),
                        "bars": bars,
                        "source": str(path),
                    })

    random.Random(seed).shuffle(records)
    if skipped:
        print(f"  ({skipped} file(s) failed to parse and were skipped)")
    return records, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bars", default="2,4,8", help="comma-separated fragment lengths in bars")
    parser.add_argument("--transpose-min", type=int, default=-5)
    parser.add_argument("--transpose-max", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bar_options = tuple(int(b) for b in args.bars.split(","))
    cfg = TokenizerConfig()

    print(f"scanning {args.input_dir} ...")
    records, tokenizer = build_dataset(
        args.input_dir, cfg, bar_options=bar_options,
        transpose_range=(args.transpose_min, args.transpose_max), seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with meta_path.open("w") as f:
        json.dump({
            "tokenizer_config": cfg.to_dict(),
            "vocab_size": tokenizer.vocab_size,
            "num_fragments": len(records),
            "bar_options": bar_options,
        }, f, indent=2)

    print(f"wrote {len(records)} fragments -> {args.output}")
    print(f"wrote tokenizer config -> {meta_path}")
    print(f"vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
