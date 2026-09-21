"""Dual-stream audio capture via ffmpeg + avfoundation.

Diarization for free. Rather than separating speakers from a mixed recording --
which ADR 007 says needs an audio model and which is the hardest part of live
transcription -- we capture two devices separately and label by source:

    microphone    -> interviewer (you)
    system audio  -> candidate   (everyone else in the call)

Perfect attribution, no model, no latency. The cost is one setup step: a loopback
device (BlackHole) so the call's output can be read as an input.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..models import Speaker

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class AudioChunk:
    speaker: Speaker
    audio: np.ndarray  # float32 mono at SAMPLE_RATE
    t_ms: int          # ms since session start


def list_devices() -> list[tuple[int, str]]:
    """Audio input devices as avfoundation indexes them."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it: brew install ffmpeg")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    out: list[tuple[int, str]] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        if "] [" in line:
            head, _, name = line.partition("] [")
            idx, _, rest = name.partition("]")
            if idx.strip().isdigit():
                out.append((int(idx), rest.strip()))
    return out


class _Stream(threading.Thread):
    """One ffmpeg process, read into fixed-length windows."""

    def __init__(self, device: int, speaker: Speaker, sink: queue.Queue,
                 started_at: float, window_s: float) -> None:
        super().__init__(daemon=True, name=f"capture-{speaker.value}")
        self.device, self.speaker, self.sink = device, speaker, sink
        self.started_at, self.window_s = started_at, window_s
        self._stop = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.error: str | None = None

    def run(self) -> None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-i", f":{self.device}",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:  # pragma: no cover - depends on host
            self.error = str(exc)
            return

        want = int(SAMPLE_RATE * self.window_s) * BYTES_PER_SAMPLE
        buf = bytearray()
        assert self.proc.stdout is not None
        while not self._stop.is_set():
            block = self.proc.stdout.read(4096)
            if not block:
                break
            buf.extend(block)
            while len(buf) >= want:
                raw, buf = bytes(buf[:want]), buf[want:]
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self.sink.put(AudioChunk(
                    speaker=self.speaker,
                    audio=audio,
                    t_ms=int((time.monotonic() - self.started_at) * 1000),
                ))
        if self.proc.stderr is not None and self.error is None:
            err = self.proc.stderr.read().decode(errors="replace").strip()
            if err:
                self.error = err

    def stop(self) -> None:
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class DualCapture:
    """Capture interviewer and candidate as separate labelled streams."""

    def __init__(self, mic_device: int, system_device: int | None,
                 window_s: float = 5.0) -> None:
        self.sink: queue.Queue[AudioChunk] = queue.Queue()
        self.started_at = time.monotonic()
        self.streams = [_Stream(mic_device, Speaker.INTERVIEWER, self.sink,
                                self.started_at, window_s)]
        if system_device is not None:
            self.streams.append(_Stream(system_device, Speaker.CANDIDATE, self.sink,
                                        self.started_at, window_s))

    def start(self) -> None:
        for s in self.streams:
            s.start()

    def stop(self) -> None:
        for s in self.streams:
            s.stop()

    @property
    def errors(self) -> list[str]:
        return [f"{s.speaker.value}: {s.error}" for s in self.streams if s.error]
