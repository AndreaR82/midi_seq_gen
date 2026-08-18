"""Hyperparameter dataclasses shared by the tokenizer, model, and training/data scripts.

Keeping these in one place means a checkpoint can carry its own config and be
reloaded (for sampling) without guessing what it was trained with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TokenizerConfig:
    """Defines the token grid and vocabulary shape. Part of a model's identity —
    changing any of these values changes the vocabulary, so it's saved alongside
    every checkpoint rather than assumed to match the current code."""

    steps_per_beat: int = 4  # 16th-note grid (4 steps per quarter-note beat)
    beats_per_bar: int = 4  # 4/4 only, for now
    pitch_min: int = 36  # C2
    pitch_max: int = 96  # C7 (5 octaves — covers bass riffs through leads)
    max_event_steps: int = 32  # longest single note/rest, in grid steps (= 2 bars)
    max_bars: int = 16  # largest LEN_<n> control token

    @property
    def steps_per_bar(self) -> int:
        return self.steps_per_beat * self.beats_per_bar

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizerConfig":
        return cls(**d)


@dataclass
class ModelConfig:
    """A deliberately small decoder-only Transformer. Defaults land around
    1-2M parameters at the default vocab size — small enough to train on a
    laptop CPU in minutes and to run inference on cheap hardware later."""

    vocab_size: int = 0  # filled in from the tokenizer at build time
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    # Covers fragments up to 8 bars at the default 16th-note grid, worst case
    # (every step its own note): 2 * 8 bars * 16 steps/bar + BOS + LEN + EOS,
    # plus headroom. Training on longer fragments (LEN_ tokens > 8) needs a
    # bigger value here — the model's own positional embedding table.
    max_seq_len: int = 288

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    val_fraction: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 50
