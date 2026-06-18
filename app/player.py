"""MIDI port discovery + a scheduler thread that loops a Sequence to a port."""

import threading
import time

import mido

from .models import Sequence

BEATS_PER_BAR = 4


def list_ports() -> list[str]:
    return mido.get_output_names()


class Player:
    """Holds the current sequence and a single looping playback thread."""

    def __init__(self) -> None:
        self.sequence: Sequence | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._port = None
        self._channel = 0  # 0-based for mido
        self._lock = threading.Lock()

    def set_sequence(self, sequence: Sequence) -> None:
        self.sequence = sequence

    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, channel: int, port_name: str) -> None:
        if self.sequence is None:
            raise ValueError("No sequence generated yet")
        self.stop()  # tear down any existing loop first
        with self._lock:
            self._channel = channel - 1  # UI is 1-16, mido is 0-15
            self._port = mido.open_output(port_name)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        if self._port is not None:
            self._all_notes_off()
            self._port.close()
            self._port = None

    # --- internals -------------------------------------------------------

    def _all_notes_off(self) -> None:
        if self._port is None:
            return
        # CC 123 = All Notes Off; belt-and-suspenders for stuck notes.
        self._port.send(mido.Message("control_change", channel=self._channel,
                                     control=123, value=0))

    def _run(self) -> None:
        seq = self.sequence
        assert seq is not None and self._port is not None
        spb = 60.0 / seq.bpm
        loop_beats = seq.loop_bars * BEATS_PER_BAR

        # Build a flat, time-sorted list of (beat, message) events.
        events: list[tuple[float, mido.Message]] = []
        for n in seq.notes:
            on = mido.Message("note_on", channel=self._channel,
                              note=n.pitch, velocity=n.velocity)
            off = mido.Message("note_off", channel=self._channel,
                               note=n.pitch, velocity=0)
            events.append((n.start_beat, on))
            events.append((min(n.start_beat + n.duration_beats, loop_beats), off))
        events.sort(key=lambda e: e[0])

        while not self._stop.is_set():
            loop_start = time.monotonic()
            for beat, msg in events:
                if self._stop.is_set():
                    break
                target = loop_start + beat * spb
                self._sleep_until(target)
                if self._stop.is_set():
                    break
                self._port.send(msg)
            # hold until the end of the loop so timing stays steady
            self._sleep_until(loop_start + loop_beats * spb)

    def _sleep_until(self, target: float) -> None:
        while not self._stop.is_set():
            remaining = target - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.01))


player = Player()
