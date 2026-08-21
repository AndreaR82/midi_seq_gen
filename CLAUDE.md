# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Text prompt → MIDI sequence (via an LLM API) → looped live over USB MIDI into
hardware (built for an Elektron Digitone 2). MVP scope: **one** sequence at a time,
generated → looped → played, driven from a browser UI.

A sequence can also come from disk: `.mid` files written by `melody_model.sample`
(the from-scratch generator in `melody_model/`, see `docs/melody_model_design.md`)
are listed and loaded into the same single slot, so model output can be auditioned
on the same hardware path — or through a browser synth when no MIDI port exists.

## Commands

```sh
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # paste OPENAI_API_KEY

# run (load the key into the shell first — the app reads it from the env)
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload   # serves http://localhost:8000
```

```sh
# audition model output: write .mid files, then Load them in the UI
python -m melody_model.sample --checkpoint checkpoints/wave1/best.pt \
    --bars 4 --count 5 --out-dir generated/preview
```

```sh
# train (validation holds out whole source files — see melody_model/split.py)
python -m melody_model.train --dataset data/fragments.jsonl --out-dir checkpoints/wave2 \
    --d-model 384 --n-layer 6 --n-head 6 --d-ff 1536 --dropout 0.2

# score a checkpoint on things val loss cannot see (LEN_ adherence, in-key,
# degeneracy, copy rate vs the training split), against held-out real music
python -m melody_model.eval --checkpoint checkpoints/wave2/best.pt \
    --dataset data/fragments.jsonl
```

Tests are pytest: `python -m pytest tests/ -q` (needs `requirements-train.txt`
for the `melody_model` tests). No linter or build step is configured. The
frontend has no toolchain — `static/app.js` is plain ES modules-free JS, so
there is nothing to typecheck or bundle.

## Key facts before you change anything

- **An API key is required for `/generate` only.** It is read from the process
  environment in `app/generator.py`, which is pinned to a cheap model — keep it cheap
  unless asked. The client is built **lazily** (`_client_or_die()`), not at import:
  a missing key must never stop the app from starting, because `/ports`, `/melodies`
  and all playback need no API at all.
- **Everything except `/generate` runs with no key and no network.** Loading and
  playing a `.mid` off disk is fully offline.
- **Sound needs a Digitone or, on macOS, the IAC Driver** virtual port (enable in
  *Audio MIDI Setup*) which then shows up in `/ports`. With no port at all, the
  browser preview is the only way to hear anything — that is what it exists for.

## Architecture

Five backend modules under `app/` plus a static frontend. Two sources feed one
sink — whichever ran last owns the single `player` slot:

    UI → /generate      → generator (LLM)              ┐
                                                       ├→ Sequence in singleton player
    UI → /melodies/load → midi_file_to_sequence (disk) ┘
                                                            │
                                    /play → scheduler thread → MIDI port
                                    Preview → WebAudio in the browser

- `app/models.py` — Pydantic schema. A `Sequence` is `{loop_bars, bpm, notes}` where
  each `Note` is `{pitch, start_beat, duration_beats, velocity}`. Times are in
  **quarter-note beats** (1 bar = 4 beats), 0-based. **Channel is deliberately NOT in
  the schema** — it comes from the UI at play time.
- `app/generator.py` — a single LLM call using structured output, so the model returns
  a `Sequence` directly rather than JSON to be parsed by hand. The `SYSTEM` prompt
  carries the musical rules (GM drum note numbers, key handling, loop length).
  User-requested BPM is **force-overwritten** onto the parsed result regardless of what
  the model returns. The provider and model id sit at the top of `generator.py` —
  read them there rather than assuming; they have been switched before.
- `app/player.py` — `Player` singleton (`player.player`). Holds one current sequence and
  one daemon thread. Playback uses an **internal monotonic clock**: notes are flattened
  to time-sorted `(beat, message)` events and looped; `stop()` joins the thread, sends
  CC 123 (All Notes Off), and closes the port. **UI channels are 1-16; mido is 0-15** —
  the `- 1` conversion happens in `Player.play`.
- `app/melodies.py` — discovery and loading of `.mid` files. `melody_root()` is
  `generated/` unless **`MELODY_DIR`** overrides it; `list_melodies()` walks it
  recursively (`**/*.mid`), newest mtime first. `load_melody()` resolves the
  client-supplied path and **rejects anything not under the root** — the path comes
  over HTTP, so it is never trusted as a string. The app deliberately does **not**
  import torch or run the model: `melody_model.sample` stays the only producer.
- `app/main.py` — FastAPI routes: `/generate`, `/ports`, `/melodies`, `/melodies/load`,
  `/play`, `/stop`, `/` (serves the UI), and mounts `/static`. Generation failures
  surface as HTTP 502; play and melody-load failures as 400.
- `melody_model/midi_io.py` — the bridge in both directions. `notes_to_midi_file()`
  writes; **`midi_file_to_sequence()`** is the only MIDI *reader* in the repo and
  the only thing `app/` imports from `melody_model` (safe: importing it pulls in
  `mido` but not torch). It merges all tracks, treats `note_on` velocity 0 as
  `note_off`, takes the first `set_tempo` (120 if absent), discards per-note channel,
  and clamps bpm into the schema's 20-300.
- `static/index.html` + `static/app.js` — UI only; the API key stays server-side.
  The **browser preview** is a plain oscillator synth scheduled against
  `AudioContext.currentTime` (not `setTimeout`, which would jitter the rhythm), with
  a lookahead pump that re-queues each loop ~0.3s before it is due. It previews
  whatever is armed, so it works for LLM-generated sequences too.

## Conventions / gotchas

- One sequence, one playback thread at a time by design. `play()` always calls `stop()`
  first to tear down any existing loop. `/generate` and `/melodies/load` write the
  **same** slot — loading a melody clobbers a generated sequence and vice versa. The
  UI's "Armed:" label exists so that is never ambiguous.
- **Loop length is derived from the last note _onset_, not the last note end**
  (`midi_file_to_sequence`). A final note ringing past the barline is overhang, not an
  extra bar — rounding it up would inject a bar of silence into every loop, and
  `Player` already truncates a note at the loop edge like a hardware sequencer. A note
  that genuinely *starts* in a later bar does extend the loop, which is how a model
  overrunning its `LEN_` control token stays visible instead of being silently cropped.
- Files whose onsets exceed 16 bars are **rejected** with their actual length, rather
  than truncated, because `Sequence.loop_bars` caps at 16.
- Playback of a loaded file uses the **file's own tempo**; the UI BPM box only affects
  `/generate`. To hear a melody at another tempo, re-run `sample` with `--bpm`.
- Key/scale is **prompt context only** — there is no backend scale quantization.
- Deferred and intentionally absent (do not assume these exist): multi-track layering,
  MIDI clock/transport sync to the Digitone, a visual piano-roll/editor, backend scale
  enforcement, and **running `melody_model` inside the app** — `/melodies` reads files
  off disk on purpose, which is what keeps torch out of the server process. If live
  sampling is ever wanted, it slots in behind the existing `/melodies` interface.
