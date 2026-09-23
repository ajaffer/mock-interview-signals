"""Speech to text for live capture.

faster-whisper on the local machine. `base.en` keeps up comfortably on Apple
silicon; `small.en` is more accurate and still real-time on an M-series chip.
Speaker attribution does not come from here -- it comes from which device the
audio arrived on (see capture.py).
"""

from __future__ import annotations

import numpy as np


class Transcriber:
    def __init__(self, model_size: str = "base.en") -> None:
        from faster_whisper import WhisperModel

        # int8 keeps latency low; these are short windows, not archival transcription.
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0 or float(np.abs(audio).max()) < 0.005:
            return ""  # silence, don't pay for it
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
