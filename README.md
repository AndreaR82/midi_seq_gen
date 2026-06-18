# MIDI Sequence Generator

Generate MIDI sequences from text prompts using the Claude API, then play them
live over USB MIDI into hardware (built for an Elektron Digitone 2).

This is an MVP: one sequence at a time, generated → looped → played, driven from
a simple browser UI.

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
