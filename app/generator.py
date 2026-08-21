"""Turn a text query into a MIDI Sequence via the OpenAI API."""

from openai import OpenAI

from .models import GenerateRequest, Sequence

MODEL = "gpt-4o-mini"

SYSTEM = """You are a MIDI sequence generator for live electronic music.

Output a note-event list as structured JSON. Rules:
- start_beat and duration_beats are in quarter-note beats; bar = 4 beats.
- Keep all notes inside loop_bars * 4 beats.
- For drum/percussion queries use General MIDI drum note numbers:
  kick 36, snare 38, closed hat 42, open hat 46, clap 39, low tom 41,
  rimshot 37, ride 51, crash 49.
- For melodic/bass queries write notes in the requested key.
- Match velocity dynamics to the style (accents louder, ghost notes quieter).
- Pick a musically sensible loop_bars (1 or 2 for most loops)."""

_client = None


def _client_or_die() -> OpenAI:
    """Built on first use, not at import: a missing key must not stop the app
    from starting, since /ports, /melodies and playback need no API at all."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def generate_sequence(req: GenerateRequest) -> Sequence:
    user = (
        f"{req.query}\n\n"
        f"Tempo: {req.bpm} BPM. Key: {req.key}. "
        f"Set the sequence bpm to {req.bpm}."
    )
    completion = _client_or_die().beta.chat.completions.parse(
        model=MODEL,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=Sequence,
    )
    seq = completion.choices[0].message.parsed
    if seq is None:
        raise ValueError("OpenAI did not return a parseable sequence")
    # Force the bpm the user asked for, regardless of what the model wrote.
    seq.bpm = req.bpm
    return seq
