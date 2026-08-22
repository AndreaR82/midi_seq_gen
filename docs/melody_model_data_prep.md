# Melody model — data preparation

How raw MIDI files become tokenized training fragments for `melody_model/`.
This documents the pipeline in `melody_model/data/` (plus the tokenizer), the
knobs that govern it, and the concrete results of the first real run (Wave 1).

For the *why* behind the design choices, see
[`melody_model_design.md`](melody_model_design.md). This doc is the operational
companion: what each stage does, what it emits, and how to reproduce it.

---

## 1. The shape of the problem

The model trains on **short monophonic melodic fragments** — one note at a time,
1–8 bars long (every length in that range, one-bar riffs included). Raw MIDI is
none of those things: it's polyphonic, multi-track, arbitrary-length, and mixed
with drums and chords. Data prep's whole job is to
mine playable monophonic lines out of that mess and slice them into uniform,
tokenized training examples.

**The internal unit** (`melody_model/notes.py`) is `NoteEvent(pitch, start_step,
dur_step)` — everything lives on an integer **step grid** (16th notes by default),
not float beats, because the entire pipeline from MIDI-in to tokens-out is
grid-quantized. `1 beat = 4 steps`, `1 bar = 16 steps` (4/4 assumed throughout).

**End-to-end flow:**

```
raw .mid/.midi files
      │
      ▼  extract.py        group by MIDI channel → score monophony → skyline
monophonic NoteEvent lines (one or more per file)
      │
      ▼  segment.py        bar-aligned windows (1–8) → filter → dedup → transpose
clean, augmented fragments
      │
      ▼  tokenizer.py      [BOS, LEN_n, (event,dur)…, EOS]  → list[int]
      │
      ▼  prepare_dataset.py  ties it together as a CLI
fragments.jsonl  +  fragments.jsonl.meta.json
```

---

## 2. Stage 0 — Acquisition (Wave 1)

Data prep consumes **any directory tree of `.mid`/`.midi` files** (recursed via
`rglob`), so acquisition is just "get MIDI onto disk." Genre tags are *not*
consumed — genre conditioning is a future feature — so no metadata wrangling is
needed.

Wave 1 uses two clean, permissive, already-melodic sources (a deliberately
phased start before noisier/larger corpora):

| Source | Genre | What we took | Count | Why this subset |
|---|---|---|---|---|
| [POP909](https://github.com/music-x-lab/POP909-Dataset) | Pop | canonical `POP909/NNN/NNN.mid` only | 909 | Skipped the 3× `versions/` near-duplicates — same songs, alternate alignments; they'd just triple-weight pop and get deduped anyway |
| [Nottingham](https://github.com/jukedeck/nottingham-dataset) | Folk | `MIDI/melody/` only | 1,034 | Pure monophonic melody lines — the best-fit input; skipped the melody+chord and chords-only variants |

Both land under `data/raw/<source>/` (git-ignored). Total: **1,943 files, ~17 MB.**

Later waves (deferred): Lakh (multi-genre), MAESTRO (classical), Weimar Jazz DB.
Techno/house remains a known gap.

---

## 3. Stage 1 — Extraction (`melody_model/data/extract.py`)

Pulls monophonic melodic lines out of arbitrary MIDI. A single file can yield
**several** lines (e.g. a lead channel and a bass channel both qualify) — that's
intentional, more usable fragments per song.

**Step 1 — bucket notes by channel** (`load_notes_by_channel`). All tracks are
read (delta-times restart per track but all play from tick 0), notes are matched
`note_on`→`note_off`, quantized onto the step grid, and merged by MIDI channel.
Robustness details: unmatched `note_off`s are skipped (malformed files are common
in the wild), and zero-length notes from rounding collisions are dropped.

**Step 2 — keep the plausibly-melodic channels** (`extract_melodic_lines`). For
each channel:
- **Skip percussion** — GM channel 10 (index 9) is always dropped.
- **Density gate** — need at least `min_notes` (default **8**) notes.
- **Monophony gate** — `monophony_ratio` (fraction of adjacent note pairs that
  *don't* overlap) must be ≥ `min_monophony` (default **0.5**). A channel full of
  stacked chords scores low and is rejected; a mostly-single-note lead scores high.

**Step 3 — skyline** (`skyline`). Force full monophony: sweep left→right, and
whenever a new note starts before the current one ends, keep whichever is
**higher-pitched** (trimming the lower one, or dropping the new note if it's the
lower). This both cleans up near-monophonic channels and lets a genuinely
polyphonic channel (e.g. piano comping with a top-line melody) still contribute a
usable line.

> **Note on POP909:** its melody is organized by *track*, not necessarily a
> dedicated channel. Because extraction groups by channel and then skylines,
> the melody is still recovered as the top line — verified in the Wave 1
> sanity-check (decoded POP909 fragments are genuinely melodic, not
> chord-skyline artifacts).

**Known limitation:** classic skyline (highest note wins) will eat a genuine
countermelody that crosses *above* the main line. Accepted as a good baseline.

---

## 4. Stage 2 — Segmentation & augmentation (`melody_model/data/segment.py`)

Turns each extracted line into many short, clean, deduplicated, key-augmented
fragments.

**Windowing** (`segment_into_fragments`). Non-overlapping, bar-aligned windows at
each length in `bar_options` (default **1–8** bars, i.e. every length from a
single bar to eight). Notes are re-based to
start at 0 within each window; a note straddling a window boundary is cut short in
the earlier window and dropped from the next (onset is what matters musically, not
a tied-over tail).

**Filtering** (`passes_filters`) — cheap sanity cuts:
- at least `min_notes` = **4** notes,
- at least `min_distinct_pitches` = **2** distinct pitches (kills one-note drones),
- rest ratio ≤ `max_rest_ratio` = **0.6** (kills fragments that are mostly silence).

The note-count floor is absolute, so it bites hardest on the shortest window —
but not hard enough to matter: measured over a 120-file sample, **66 %** of
one-bar windows pass (4 notes in a bar is ordinary melodic density), and after
dedup one-bar fragments are the *largest* length bucket in the corpus. No
length-relative loosening was needed.

**Deduplication** (`contour_signature`). A **transposition-invariant** fingerprint
— each note as `(pitch − first_pitch, start_step, dur_step)` — collapses the same
phrase found in different keys or via overlapping windows into one.

**Transpose augmentation** (`transpose`). Each surviving fragment is shifted across
`transpose_range` = **−5..+6 semitones** (12 copies including the original). Since
pitch is *absolute* in the tokenizer, this is how the model learns transposition
invariance. A shift that would push any note outside the tokenizer's pitch range
returns `None` (the copy is **dropped, not clamped** — clamping would distort the
melody's contour).

> **Order matters — dedup happens *before* transpose** (`prepare_dataset.build_dataset`).
> The fingerprint is transposition-invariant, so deduping *after* transposing would
> treat every augmented copy as a duplicate of the first and throw the augmentation
> away. There's a regression test for this
> (`test_transposition_augmentation_is_not_deduped_away`).

---

## 5. Stage 3 — Tokenization (`melody_model/tokenizer.py`)

Each fragment becomes a flat token-id sequence:

```
[BOS, LEN_<bars>, EVENT_1, DUR_1, EVENT_2, DUR_2, …, EOS]
```

- One control token up front: **`LEN_<bars>`** — how "set the length" becomes a
  training signal instead of post-hoc truncation.
- The body is `(event, duration)` pairs. An **event** is `REST` or
  `PITCH_<n>`; its paired **`DUR_<n>`** gives length in grid steps.
- Gaps between notes become `REST` events; a trailing rest pads the fragment out
  to the full bar length.
- A note/rest longer than `max_event_steps` (**32** steps = 2 bars) is split into
  consecutive same-type pairs (`_span`) and re-merged on decode (`_merge_adjacent`),
  so every `DUR` token stays small regardless of how long a note runs.

Pitch is **absolute** (not interval-relative) — simpler to implement; the lack of
built-in transposition invariance is covered by the augmentation step above.

---

## 6. The vocabulary & config (`melody_model/config.py` → `TokenizerConfig`)

The vocabulary is derived entirely from `TokenizerConfig`. It's saved alongside
every dataset and checkpoint so training/sampling never have to guess it.

| Field | Default | Meaning |
|---|---|---|
| `steps_per_beat` | 4 | 16th-note grid |
| `beats_per_bar` | 4 | 4/4 only |
| `pitch_min` / `pitch_max` | 36 / 96 | C2–C7 (5 octaves: bass → leads) |
| `max_event_steps` | 32 | longest single note/rest (= 2 bars) |
| `max_bars` | 16 | largest `LEN_<n>` control token |

**Vocab = 113 tokens:** 4 specials (`PAD BOS EOS REST`) + 16 `LEN_` + 61 `PITCH_`
(36–96 inclusive) + 32 `DUR_` (1–32).

---

## 7. The CLI (`melody_model/data/prepare_dataset.py`)

Ties every stage together:

```sh
python -m melody_model.data.prepare_dataset \
    --input-dir data/raw \
    --output data/fragments.jsonl
```

| Flag | Default | Effect |
|---|---|---|
| `--input-dir` | (required) | root scanned recursively for `.mid`/`.midi` |
| `--output` | (required) | `.jsonl` fragment file to write |
| `--bars` | `1,2,3,4,5,6,7,8` | fragment lengths |
| `--transpose-min` / `--transpose-max` | `-5` / `6` | augmentation range (semitones) |
| `--seed` | `0` | shuffle seed for the output ordering |

**Robustness:** files that fail to parse (malformed MIDI is common) are caught,
counted, and skipped rather than aborting the run.

**Outputs:**
- `fragments.jsonl` — one JSON object per line:
  `{"tokens": [int, …], "bars": N, "source": "path/to/file.mid"}`.
  Records are **shuffled before writing** (seeded), so any `head -n K` prefix is
  already a uniform random subsample.
- `fragments.jsonl.meta.json` — the exact `TokenizerConfig`, `vocab_size`,
  `num_fragments`, and `bar_options`. `train.py` requires this file so it can
  rebuild the identical tokenizer.

---

## 8. Corpus results (actual run)

`--input-dir data/raw` over the 1,943 Wave 1 files, defaults, ~3 min:

- **2,602,037 fragments**, vocab **113**.
- **Per source:** POP909 1,974,545 · Nottingham 627,492 (≈ **3.1 : 1**, pop-heavy —
  a genre-balance consideration for later waves).
- **Coverage:** 908 / 909 POP909 files and 1,027 / 1,034 Nottingham files yielded
  fragments — extraction rejected almost nothing.
- **Length mix** (shorter windows tile more times per line, so they dominate):

  | bars | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|---|
  | fragments | 504,741 | 480,837 | 422,111 | 302,819 | 286,161 | 226,630 | 210,381 | 168,357 |
  | share | 19.4% | 18.5% | 16.2% | 11.6% | 11.0% | 8.7% | 8.1% | 6.5% |

  One-bar fragments are the biggest bucket but not a majority; the distribution
  tapers smoothly, so no length is starved.

**Interpreting the multiplier:** dedup runs *before* the 12× transpose, so
~2.6M fragments correspond to only **≈ 219k distinct underlying phrases** (each in
up to 12 keys). Real musical diversity is bounded by the phrase count, not the
fragment count — worth remembering when reasoning about overfitting and about how
much genuinely new signal Wave 2 would add.

> **History:** the earlier `--bars 2,4,8` build of the same corpus produced
> 964,525 fragments from ≈ 80k phrases. Widening to every length from 1 to 8 bars
> is a **2.7×** increase in both — same music, ~2.7× more windows onto it, and
> 612 MB instead of 227 MB on disk.

---

## 9. Reproducing from scratch

```sh
# deps (torch, numpy, mido)
.venv/bin/pip install -r requirements-train.txt

# acquire Wave 1 (shallow-clone to a temp dir, copy only the .mid we want)
git clone --depth 1 https://github.com/music-x-lab/POP909-Dataset.git /tmp/pop909_src
git clone --depth 1 https://github.com/jukedeck/nottingham-dataset.git  /tmp/nott_src
mkdir -p data/raw/pop909 data/raw/nottingham
cp /tmp/pop909_src/POP909/*/*.mid       data/raw/pop909/       # canonical only, not versions/
cp /tmp/nott_src/MIDI/melody/*.mid      data/raw/nottingham/   # melody/ only

# build the dataset
PYTHONPATH=. .venv/bin/python -m melody_model.data.prepare_dataset \
    --input-dir data/raw --output data/fragments.jsonl
```

`data/`, `checkpoints/`, and `generated/` are already git-ignored.

---

## 10. Caveats & where the knobs are

- **Genre imbalance** — Wave 1 is pop-heavy (3.3:1). Balance comes from Wave 2 data
  and, eventually, genre-conditioning tokens (a future feature).
- **Skyline eats countermelodies** that cross above the main line.
- **Decode-time note count can dip below the filter's `min_notes`** — `_merge_adjacent`
  rejoins back-to-back same-pitch notes after the filter already counted them
  separately. Negligible at corpus scale.
- **If fragment counts look wrong,** tune extraction (`min_notes`, `min_monophony`
  in `extract_melodic_lines`) or the filters (`passes_filters`) — *not* the
  tokenizer, whose config defines the vocabulary identity.
- **4/4 is assumed everywhere;** time signatures are ignored, matching `app/player.py`.
