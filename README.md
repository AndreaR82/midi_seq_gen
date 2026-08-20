# MIDI Sequence Generator

Two related projects sharing one repo:

- **`app/`** — the live MVP: generate MIDI sequences from text prompts using
  the Claude API, then play them live over USB MIDI into hardware (built
  for an Elektron Digitone 2). One sequence at a time, generated → looped →
  played, driven from a simple browser UI.
- **`melody_model/`** — a small, from-scratch generative model for
  monophonic melodic fragments (bass riffs, arps, leads), trained on short
  phrases pulled from many genres. See
  [`docs/melody_model_design.md`](docs/melody_model_design.md) for the full
  design and current status.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your ANTHROPIC_API_KEY
```

> **Note:** the Anthropic API is billed separately from a Claude Pro subscription.
> Get a key + credits at https://console.anthropic.com. This app uses the cheap
> `claude-haiku-4-5` model.

## Run

```sh
export $(grep -v '^#' .env | xargs)   # load ANTHROPIC_API_KEY into the shell
uvicorn app.main:app --reload
```

Open http://localhost:8000.

## Use

1. Type a prompt (e.g. *"write a drum sequence for a techno song"*).
2. Set BPM, key, and the MIDI channel that matches a Digitone track.
3. Pick the MIDI output port (refresh with ↻ after plugging in the Digitone).
4. **Generate** → see the note summary + raw JSON.
5. **Play** to loop it out the port; **Stop** to silence.

## Testing without hardware (macOS)

Enable the **IAC Driver** in *Audio MIDI Setup* and route it to any soft-synth;
it will appear in the port dropdown. You can also just confirm generation works
and that `/ports` lists ports before connecting the Digitone.

## Layout

- `app/models.py` — note-event schema + request bodies
- `app/generator.py` — Claude call (structured output → `Sequence`)
- `app/player.py` — MIDI port discovery + looping scheduler thread
- `app/main.py` — FastAPI routes, serves the frontend
- `static/` — HTML + JS UI
- `melody_model/` — from-scratch tokenizer, model, and training pipeline for
  the small melody generator (see below)
- `docs/melody_model_design.md` — its design doc and current status

## Melody model

A small decoder-only Transformer, trained from scratch (hand-written
tokenizer and attention, PyTorch for tensors/autograd), on short monophonic
fragments extracted from multi-genre MIDI. Full rationale in
[`docs/melody_model_design.md`](docs/melody_model_design.md); quickstart:

```sh
pip install -r requirements-train.txt

# 1. raw MIDI files -> tokenized fragments (point --input-dir at your own collection)
python -m melody_model.data.prepare_dataset \
    --input-dir /path/to/midi_collection --output data/fragments.jsonl

# 2. train
python -m melody_model.train --dataset data/fragments.jsonl --out-dir checkpoints/run1

# 3. generate — each run/seed is a new "button press"
python -m melody_model.sample --checkpoint checkpoints/run1/best.pt \
    --bars 4 --count 5 --out-dir generated/
```

Run the test suite (tokenizer, extraction, model, end-to-end data pipeline)
with:

```sh
pip install pytest
python -m pytest tests/
```

**Status:** the pipeline is scaffolded and tested end-to-end against
synthetic MIDI fixtures — no real dataset has been sourced yet and no real
training run has happened. See the design doc's Status section before
expecting musical output.
