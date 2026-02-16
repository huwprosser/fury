from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np
import soundfile as sf

AudioSource = Union[str, Path, BinaryIO]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers.")

    ratio = target_sr / orig_sr
    target_length = max(int(round(audio.shape[0] * ratio)), 1)
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio(
    source: AudioSource,
    *,
    sr: int | None = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(source, dtype="float32")

    if mono:
        audio = _to_mono(audio)

    if sr is not None and sample_rate != sr:
        audio = _resample_linear(audio, sample_rate, sr)
        sample_rate = sr

    return audio, sample_rate
