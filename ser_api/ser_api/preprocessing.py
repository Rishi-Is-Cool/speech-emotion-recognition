"""conform(): 16kHz mono, explicitly nothing else -- per the project's own audit finding
(no preprocessing was ever in the accuracy-measured path, so the API must not silently add
denoising/normalization/VAD that would change the numbers the model was actually validated on).

validate_audio() is the "too short, silent, unreadable, unsupported format" gate from
SER_Audit_and_Production_Architecture.md section 4 -- it raises SERInputError with a specific
code+message rather than letting a bad upload reach the model and crash as a raw 500."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from ser_api.config import Config
from ser_api.errors import SERInputError


def conform(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Mono, resampled to target_sr. Nothing else -- no denoising, no normalization, no VAD."""
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != target_sr:
        n_target = int(round(len(y) * target_sr / orig_sr))
        y = np.interp(
            np.linspace(0, len(y), n_target, endpoint=False), np.arange(len(y)), y
        ).astype(np.float32)
    return y.astype(np.float32)


def load_and_validate(audio_bytes: bytes, config: Config) -> tuple[np.ndarray, float]:
    """Read raw upload bytes -> validated, conformed mono/16kHz waveform + its duration in
    seconds. Raises SERInputError (never a bare exception) for anything the caller should be
    told about specifically: unreadable file, too short, too long, silent."""
    if not audio_bytes:
        raise SERInputError("empty_upload", "The uploaded file is empty.")

    try:
        y, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        raise SERInputError(
            "unreadable_audio",
            "Could not read this file as audio. Supported formats: wav, flac, ogg "
            "(anything libsndfile can decode). If this is an mp3/m4a, convert it to wav first.",
        ) from e

    if y.size == 0:
        raise SERInputError("empty_audio", "The audio file contains no samples.")

    y = conform(y, orig_sr, config.sample_rate)
    duration_sec = len(y) / config.sample_rate

    if duration_sec < config.min_duration_sec:
        raise SERInputError(
            "too_short",
            f"Recording is {duration_sec:.1f}s, shorter than the {config.min_duration_sec:.1f}s "
            "minimum needed for a reliable prediction.",
        )
    if duration_sec > config.max_duration_sec:
        raise SERInputError(
            "too_long",
            f"Recording is {duration_sec:.1f}s, longer than the {config.max_duration_sec:.0f}s "
            "maximum this service accepts per request.",
        )

    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    if rms < config.silence_rms_threshold:
        raise SERInputError(
            "silent_audio",
            "Recording appears to be silent (no meaningful audio energy detected).",
        )

    return y, duration_sec
