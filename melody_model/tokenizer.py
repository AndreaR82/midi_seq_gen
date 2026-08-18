"""A from-scratch tokenizer for monophonic melodic fragments.

Design (see docs/melody_model_design.md for the reasoning):

  [BOS, LEN_<bars>, EVENT_1, DUR_1, EVENT_2, DUR_2, ..., EOS]

- One control token up front: LEN_<bars>, the fragment's length in bars.
- The body is a flat stream of (event, duration) pairs. An event is either
  REST or PITCH_<midi note number>. Its paired DUR_<n> token gives the
  event's length in grid steps (16th notes by default).
- Rests and notes share one duration vocabulary, and a rest/note longer than
  `max_event_steps` is split into consecutive same-type pairs, so every
  DUR token stays small and the vocabulary stays tiny (on the order of
  100-150 tokens total at the default config) regardless of fragment length.

No external tokenization library is used — this is deliberately hand-rolled
per the project's "build it from scratch" goal.
"""

from __future__ import annotations

from .config import TokenizerConfig
from .notes import NoteEvent

PAD = "PAD"
BOS = "BOS"
EOS = "EOS"
REST = "REST"


class MelodyTokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self._build_vocab()

    # -- vocab construction -------------------------------------------------

    def _build_vocab(self) -> None:
        cfg = self.config
        tokens: list[str] = [PAD, BOS, EOS, REST]
        tokens += [f"LEN_{n}" for n in range(1, cfg.max_bars + 1)]
        tokens += [f"PITCH_{p}" for p in range(cfg.pitch_min, cfg.pitch_max + 1)]
        tokens += [f"DUR_{d}" for d in range(1, cfg.max_event_steps + 1)]

        self.id_to_token = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.pad_id = self.token_to_id[PAD]
        self.bos_id = self.token_to_id[BOS]
        self.eos_id = self.token_to_id[EOS]
        self.rest_id = self.token_to_id[REST]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def len_token_id(self, bars: int) -> int:
        bars = max(1, min(bars, self.config.max_bars))
        return self.token_to_id[f"LEN_{bars}"]

    def pitch_token_id(self, pitch: int) -> int:
        cfg = self.config
        clamped = max(cfg.pitch_min, min(pitch, cfg.pitch_max))
        return self.token_to_id[f"PITCH_{clamped}"]

    def dur_token_id(self, steps: int) -> int:
        steps = max(1, min(steps, self.config.max_event_steps))
        return self.token_to_id[f"DUR_{steps}"]

    # -- encode ---------------------------------------------------------------

    def encode(self, notes: list[NoteEvent], num_bars: int) -> list[int]:
        """Notes must be monophonic (non-overlapping) and sorted or not —
        they're sorted here. Gaps become REST events; the fragment is padded
        with a trailing rest out to num_bars if the last note ends early."""
        cfg = self.config
        total_steps = num_bars * cfg.steps_per_bar
        ids = [self.bos_id, self.len_token_id(num_bars)]

        cursor = 0
        for n in sorted(notes, key=lambda n: n.start_step):
            if n.start_step > cursor:
                ids += self._span(self.rest_id, n.start_step - cursor)
            dur = max(1, min(n.dur_step, total_steps - n.start_step))
            ids += self._span(self.pitch_token_id(n.pitch), dur)
            cursor = n.start_step + dur

        if cursor < total_steps:
            ids += self._span(self.rest_id, total_steps - cursor)

        ids.append(self.eos_id)
        return ids

    def _span(self, event_id: int, steps: int) -> list[int]:
        """Split a run of `steps` grid steps into consecutive (event, DUR)
        pairs, each capped at max_event_steps."""
        out: list[int] = []
        remaining = steps
        cap = self.config.max_event_steps
        while remaining > 0:
            chunk = min(remaining, cap)
            out += [event_id, self.dur_token_id(chunk)]
            remaining -= chunk
        return out

    # -- decode ---------------------------------------------------------------

    def decode(self, ids: list[int]) -> tuple[list[NoteEvent], int]:
        """Returns (notes, num_bars). Ignores PAD/BOS/EOS and stops at the
        first EOS if present. Tolerant of a malformed/truncated tail (e.g.
        a dangling event token with no duration) since generated sequences
        can be cut off mid-token by a max-length limit."""
        num_bars = 1
        notes: list[NoteEvent] = []
        cursor = 0

        i = 0
        n = len(ids)
        while i < n:
            tok = self.id_to_token[ids[i]] if 0 <= ids[i] < len(self.id_to_token) else PAD
            if tok in (PAD, BOS):
                i += 1
                continue
            if tok == EOS:
                break
            if tok.startswith("LEN_"):
                num_bars = int(tok.split("_", 1)[1])
                i += 1
                continue
            if tok == REST or tok.startswith("PITCH_"):
                if i + 1 >= n:
                    break  # dangling event with no duration token — drop it
                dur_tok = self.id_to_token[ids[i + 1]] if 0 <= ids[i + 1] < len(self.id_to_token) else PAD
                if not dur_tok.startswith("DUR_"):
                    i += 1
                    continue  # malformed pairing — skip and resync
                steps = int(dur_tok.split("_", 1)[1])
                if tok.startswith("PITCH_"):
                    pitch = int(tok.split("_", 1)[1])
                    notes.append(NoteEvent(pitch=pitch, start_step=cursor, dur_step=steps))
                cursor += steps
                i += 2
                continue
            i += 1  # unknown token — skip

        return self._merge_adjacent(notes), num_bars

    @staticmethod
    def _merge_adjacent(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Re-join notes of the same pitch that are directly back-to-back —
        the inverse of the splitting `_span` does for long notes."""
        if not notes:
            return notes
        merged = [notes[0]]
        for n in notes[1:]:
            last = merged[-1]
            if n.pitch == last.pitch and n.start_step == last.end_step:
                last.dur_step += n.dur_step
            else:
                merged.append(n)
        return merged
