"""Stage 5's timeline, ported into the service unchanged from notebook 23's verified logic:
non-overlapping windows, independent acoustic evidence (librosa, not the model), and per-window
linear-classifier attribution (`w_c . scaler(x_window)`).

IMPORTANT, from SER_Stage5_Explainability_Finding.md: this attribution is exact ONLY when a clip
is short enough to be a single window (nothing was split, so there's nothing to approximate).
For real multi-window clips, reconstruction error is large (measured 2.18 on a real 44s
recording) -- the underlying cause is that each window is re-embedded independently and a
transformer encoder's features depend on context the isolated crop doesn't have. So every
response with n_windows > 1 must be read as approximate supporting evidence, not exact math --
`attribution_exact` below is the caller-facing flag for this, and the API never claims exactness
it hasn't verified for that specific request."""

from __future__ import annotations

import dataclasses

import librosa
import numpy as np

from ser_api.inference import PredictionResult, embed_raw
from ser_api.model_package import ModelPackage

# Placeholder copy -- swap for notebook 10's real EMOTION_EXPLANATION_LANGUAGE dict if/when
# available, so wording stays consistent across the product rather than being redefined here.
EMOTION_EXPLANATION_LANGUAGE = {
    "angry": "raised pitch and energy consistent with an angry tone",
    "disgust": "vocal tension consistent with a disgusted tone",
    "fear": "unsteady pitch and energy consistent with a fearful tone",
    "happy": "raised pitch and energetic delivery consistent with a happy tone",
    "neutral": "flat pitch and steady energy consistent with a neutral tone",
    "sad": "low energy and falling pitch consistent with a sad tone",
}


def make_windows(n_samples: int, sr: int, window_sec: float) -> list[tuple[int, int]]:
    """Non-overlapping partition (last window shorter if it doesn't divide evenly). A clip no
    longer than one window returns a single window spanning the whole clip -- see module
    docstring for why that single-window case is the only case where attribution is exact."""
    win = int(window_sec * sr)
    if n_samples <= win:
        return [(0, n_samples)]
    spans = []
    start = 0
    while start < n_samples:
        end = min(start + win, n_samples)
        spans.append((start, end))
        start = end
    return spans


def acoustic_evidence(y: np.ndarray, sr: int) -> dict:
    """Independent DSP evidence (pitch, energy), not model output. Correlational, not causal --
    keep that framing wherever this is surfaced (SER_Audit_and_Production_Architecture.md §5)."""
    energy = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y.astype(np.float64), fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
        )
        voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced = voiced[~np.isnan(voiced)] if len(voiced) else voiced
        pitch_hz = float(np.mean(voiced)) if len(voiced) else None
    except Exception:
        pitch_hz = None
    return {"pitch_hz": pitch_hz, "energy_rms": energy}


@dataclasses.dataclass
class Timeline:
    window_sec: float
    n_windows: int
    attribution_exact: bool
    reconstruction_error: float
    windows: list[dict]
    top_window_idx: int
    explanation_text: str


def linear_windows(y, sr, embed_fn, scaler, clf, pred_idx, whole_embedding, window_sec):
    """Per-window attribution for any scaler + linear head on the encoder's embedding:
    `w_c . scaler(x_window)` for each window. Returns (windows, top_window_idx, attribution_exact,
    reconstruction_error). `embed_fn(window_samples)` returns that window's embedding."""
    w_c = clf.coef_[pred_idx]
    b_c = float(clf.intercept_[pred_idx])
    whole_scaled = scaler.transform(whole_embedding.reshape(1, -1))
    true_logit = float(clf.decision_function(whole_scaled)[0][pred_idx])

    spans = make_windows(len(y), sr, window_sec)

    windows = []
    contributions = []
    weights = []
    for k, (a, b) in enumerate(spans):
        wy = y[a:b]
        w_embed = embed_fn(wy)
        w_scaled = scaler.transform(w_embed.reshape(1, -1))[0]
        contribution = float(np.dot(w_c, w_scaled))
        contributions.append(contribution)
        weights.append(b - a)  # window length in samples -- the last window is usually shorter
        evidence = acoustic_evidence(wy, sr)
        windows.append({
            "window_idx": k,
            "start_s": round(a / sr, 2),
            "end_s": round(b / sr, 2),
            "contribution": contribution,
            **evidence,
        })

    weights_arr = np.array(weights, dtype=np.float64)
    recon_logit = float(np.average(contributions, weights=weights_arr) + b_c)
    reconstruction_error = abs(recon_logit - true_logit)

    total_abs = sum(abs(w["contribution"]) for w in windows) or 1.0
    for w in windows:
        w["contribution_pct"] = round(100 * w["contribution"] / total_abs, 1)
    top_window = max(windows, key=lambda w: w["contribution"])

    # Exact by construction only when nothing was split -- see module docstring. This is a
    # structural fact (n_windows == 1), not something that needs re-measuring per request.
    attribution_exact = len(windows) == 1
    return windows, top_window["window_idx"], attribution_exact, reconstruction_error


def build_timeline(y: np.ndarray, sr: int, model: ModelPackage, prediction: PredictionResult) -> Timeline:
    pred_idx = model.classes.index(prediction.emotion)
    windows, top_idx, attribution_exact, reconstruction_error = linear_windows(
        y, sr, lambda wy: embed_raw(wy, sr, model), model.scaler, model.clf, pred_idx,
        prediction.embedding, model.window_sec,
    )
    top_window = windows[top_idx]

    explanation_text = (
        f"Predicted '{prediction.emotion}' ({prediction.confidence * 100:.0f}% confidence). "
        f"The strongest evidence is at {top_window['start_s']}s-{top_window['end_s']}s, where "
        f"the audio shows {EMOTION_EXPLANATION_LANGUAGE.get(prediction.emotion, 'a distinct tone')}. "
        "This explains the classifier's decision, not what the model 'heard' internally."
        + ("" if attribution_exact else " Per-window contributions are approximate for "
           "recordings long enough to span more than one window -- see attribution_exact.")
    )

    return Timeline(
        window_sec=model.window_sec,
        n_windows=len(windows),
        attribution_exact=attribution_exact,
        reconstruction_error=reconstruction_error,
        windows=windows,
        top_window_idx=top_idx,
        explanation_text=explanation_text,
    )
