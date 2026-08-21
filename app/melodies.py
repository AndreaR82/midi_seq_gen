"""Discovery and loading of .mid files produced by melody_model.sample.

The app never runs the model itself -- `python -m melody_model.sample` stays
the only producer, which keeps torch out of the server process. This module
is just the bridge: find the files it wrote, parse one into a Sequence.
"""

import os
from pathlib import Path

from melody_model.midi_io import midi_file_to_sequence

from .models import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def melody_root() -> Path:
    """Where to look for .mid files. MELODY_DIR overrides for auditioning a
    folder outside the repo (a corpus sample, a bounce off the Digitone)."""
    return Path(os.environ.get("MELODY_DIR") or REPO_ROOT / "generated").resolve()


def list_melodies() -> list[dict]:
    """Every .mid under the melody root, newest first -- whatever you just
    sampled lands at the top of the picker."""
    root = melody_root()
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*.mid") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"path": str(p.relative_to(root)), "size": p.stat().st_size}
        for p in files
    ]


def load_melody(rel_path: str) -> Sequence:
    root = melody_root()
    target = (root / rel_path).resolve()
    # The path comes from the client, so confirm it actually lives under the
    # root rather than trusting the string (../../etc/passwd).
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"No such melody: {rel_path}")
    return midi_file_to_sequence(target)
