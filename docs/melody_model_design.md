# Melody model — design notes

**Goal:** a small, from-scratch generative model for monophonic melodic
fragments (bass riffs, arpeggios, leads) — trained on short phrases pulled
from many genres, musical enough to beat a scale-random or Euclidean
generator, and eventually deployable on cheap hardware behind a
"set length + press button → get a new MIDI melody" interaction. This
first pass targets a laptop; the hardware/embedded target is deliberately
deferred (see Status below).

It lives in `melody_model/`, separate from `app/` (the live Claude-in-the-loop
MVP, untouched). The two only meet at `melody_model/midi_io.py`, which can
convert a generated melody into `app.models.Sequence` so it can be looped
out over MIDI through `app/player.py` exactly like a Claude-generated one.

## Data pipeline (`melody_model/data/`)

1. **Extract** (`extract.py`) — pull monophonic melodic lines out of
   arbitrary MIDI files: group notes by channel, score each channel's
   monophony ratio, keep channels that are dense enough and not too
   chordal (and never GM channel 10/percussion), then run a **skyline**
   pass (keep the higher note whenever two overlap) to force full
   monophony. A single file can yield several lines (e.g. a lead and a
   bass channel both qualify) — deliberately, for more usable fragments
   per song. Time signature is ignored; 4/4 is assumed throughout, same
   as `app/player.py`.

2. **Segment** (`segment.py`) — slice each line into non-overlapping,
   bar-aligned fragments at a few target lengths (2/4/8 bars by default),
   drop fragments that are too sparse or too static, dedupe by a
   transposition-invariant contour fingerprint (relative pitch + rhythm),
   and augment survivors by transposing into nearby keys (dropped, not
   clamped, if a transposition would push a note out of the tokenizer's
   pitch range — clamping would distort the melody's shape).

   **Dedupe happens before transposing**, deliberately: the fingerprint is
   transposition-invariant by construction, so deduping after transposing
   would treat every augmented copy of the same phrase as a duplicate of
   the first one generated and throw the augmentation away. There's a
   regression test for this (`test_transposition_augmentation_is_not_deduped_away`).

3. **`prepare_dataset.py`** ties it together into a CLI: a directory of raw
   `.mid`/`.midi` files in, a `.jsonl` of tokenized fragments out, plus a
   `.meta.json` recording the exact `TokenizerConfig` used (so `train.py`
   never has to guess/assume it matches).

## Tokenizer (`melody_model/tokenizer.py`)

Hand-rolled, no external tokenization library (MidiTok, REMI, etc.) —
that was an explicit goal, and a monophonic-only, loop-length domain is
small enough that a custom scheme is both simpler and easy to reason about.

```
[BOS, LEN_<bars>, EVENT_1, DUR_1, EVENT_2, DUR_2, ..., EOS]
```

- One control token up front, `LEN_<bars>` — the fragment length. This is
  how "set the sequence length" becomes a training signal rather than
  post-hoc truncation.
- The body is a flat stream of `(event, duration)` pairs. An event is
  either `REST` or `PITCH_<midi note number>`; its paired `DUR_<n>` gives
  its length in grid steps (16th notes by default — `steps_per_beat=4`).
- A note or rest longer than `max_event_steps` (default 32 = 2 bars) is
  split into consecutive same-type pairs and re-merged on decode, so every
  `DUR` token stays small no matter how long a single note runs.

Default vocab is ~113 tokens (4 specials + 16 LEN + 61 pitches [C2–C7] + 32
durations). Pitch is **absolute**, not relative/interval — simpler to
implement and debug first; the tradeoff (no built-in transposition
invariance) is covered by the transpose-augmentation step in the data
pipeline instead of by the token scheme itself.

## Model (`melody_model/model.py`)

A small decoder-only causal Transformer, hand-written (causal self-attention
implemented directly rather than via `nn.MultiheadAttention`/
`nn.TransformerDecoder`) so the architecture stays in scope for "built from
scratch" — only tensor ops/autograd come from PyTorch. Defaults
(`ModelConfig`): 4 layers, d_model 128, 4 heads, ~1-2M params at the default
vocab size. Small enough to train on a laptop CPU in minutes; a real
training run will want a GPU for iteration speed, but nothing about the
architecture requires one.

`generate()` does temperature + nucleus (top-p) sampling from an explicit
`torch.Generator`, so a different seed on an otherwise-identical call is
the entire mechanism behind "press the button again for a different
melody" — no extra machinery needed. It also **masks out `LEN_*` token ids
during generation** (`forbidden_ids`): nothing architecturally stops an
under-trained model from sampling a stray `LEN_` token mid-body and
silently overriding the requested length, so it's blocked explicitly
rather than left for training to (eventually) discourage.

## Status / what's scaffolded vs. what's still open

**Working now** (all covered by `tests/`, plus a manual end-to-end smoke
test: synthetic MIDI → `prepare_dataset` → `train` → `sample` → valid,
varied `.mid` output):
- tokenizer (encode/decode round-trip, long-note splitting, out-of-range
  clamping, truncated-sequence tolerance)
- extraction (skyline, monophony scoring, percussion skipping)
- segmentation/filtering/dedup/augmentation
- model (forward pass, generation, seeded reproducibility, LEN masking)
- the three CLIs: `melody_model.data.prepare_dataset`, `melody_model.train`,
  `melody_model.sample`

**Not done / explicitly deferred:**
- **No real dataset yet.** Nothing has been downloaded — `prepare_dataset`
  has only been run against synthetic test fixtures. Sourcing real
  multi-genre MIDI (classical/jazz/pop/rock via Lakh + POP909 + Nottingham
  + a jazz solo corpus; techno/house is the weak spot — no good public
  monophonic bass/arp corpus exists, may need hand-curated or
  LLM-bootstrapped supplementing) is the next real step.
- **No real training run.** The checkpoints produced so far are a 30K-param
  model trained for 3 epochs on ~175 synthetic fragments, purely to prove
  the pipeline doesn't crash — not evidence of melody quality.
  Hyperparameters (`ModelConfig`/`TrainConfig` defaults) are untuned.
- **Hardware target undecided** (per earlier discussion — building for a
  laptop first). Quantization/export/embedded runtime is unaddressed.
- **Only length is a control token today.** Register, note-density, and
  genre/style conditioning were discussed as v2 features — same
  mechanism (more control tokens), not yet implemented.
- The skyline algorithm is the simple classic version (highest note wins
  on overlap) — good baseline per the literature, but doesn't handle
  every edge case (e.g. a genuine countermelody crossing above the main
  line will get eaten).
