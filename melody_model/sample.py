"""Generate new melodies from a trained checkpoint — the "press the
button" step. Each call with a different seed gives a different melody at
the same length/settings; --count simulates pressing the button N times
in a row.

    python -m melody_model.sample --checkpoint checkpoints/run1/best.pt \\
        --bars 4 --count 5 --out-dir generated/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import ModelConfig, TokenizerConfig
from .midi_io import notes_to_midi_file
from .model import MelodyTransformer
from .notes import NoteEvent
from .tokenizer import MelodyTokenizer


def load_checkpoint(path: Path) -> tuple[MelodyTransformer, MelodyTokenizer]:
    ckpt = torch.load(path, map_location="cpu")
    tok_cfg = TokenizerConfig.from_dict(ckpt["tokenizer_config"])
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    tokenizer = MelodyTokenizer(tok_cfg)
    model = MelodyTransformer(model_cfg, pad_id=tokenizer.pad_id)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


def generate_one(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    bars: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[NoteEvent], int]:
    prefix = [tokenizer.bos_id, tokenizer.len_token_id(bars)]
    # Generous upper bound: worst case every grid step is its own (event, DUR)
    # pair, plus the closing EOS.
    max_new_tokens = 2 * bars * tokenizer.config.steps_per_bar + 2
    # LEN_ tokens belong only in the prefix (position 1) — forbid the model
    # from sampling a fresh one and silently overriding the requested length.
    forbidden = [tokenizer.len_token_id(n) for n in range(1, tokenizer.config.max_bars + 1)]
    generator = torch.Generator().manual_seed(seed)
    ids = model.generate(prefix, max_new_tokens, tokenizer.eos_id, temperature, top_p, generator, forbidden)
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--count", type=int, default=1, help="how many melodies to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="higher = wilder/less predictable")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling cutoff")
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None, help="omit for a fresh random seed each run")
    parser.add_argument("--out-dir", type=Path, default=Path("generated"))
    args = parser.parse_args()

    model, tokenizer = load_checkpoint(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else time.time_ns() % (2**31)
    for i in range(args.count):
        notes, bars = generate_one(model, tokenizer, args.bars, args.temperature, args.top_p, base_seed + i)
        out_path = args.out_dir / f"melody_{i:03d}.mid"
        notes_to_midi_file(notes, tokenizer.config.steps_per_beat, args.bpm, out_path)
        print(f"[{i}] seed={base_seed + i} {len(notes)} notes over {bars} bar(s) -> {out_path}")


if __name__ == "__main__":
    main()
