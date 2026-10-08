"""Adaptive routing: a cheap MFCC+MLP "light" branch runs on every clip, a calibrated router
decides whether the clip is worth paying the ~24x cost of the Emotion2Vec+ "heavy" branch, and
the heavy branch runs ONLY when it escalates.

This is additive and sits beside the product's locked classifier -- it does not replace it and
`tier1b_best_head.joblib` is never touched here.

THE TRAP (serving_spec.json's own warning, repeated because it fails silently):
the light branch and the router consume DIFFERENT audio from the same recording. Swapping them
raises no error -- the models just return confident, wrong numbers.

    light branch : FULL-length audio, MFCC with n_fft=400  hop=160
    router (ASCA): FIRST 3.00s only,  librosa DEFAULTS n_fft=2048 hop=512

Both branches' scalers are applied after raw extraction and before the model. The router's three
confidence features are derived live from the light branch's probabilities and are NOT scaled.

Verified against Router/demo_clips_manifest.json on all three demo clips: light confidence to
1.8e-07, router score to 8.6e-08, heavy confidence to 1.1e-06.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np

from ser_api.config import Config
from ser_api.inference import embed_raw
from ser_api.model_package import ModelPackage

logger = logging.getLogger("ser_api.adaptive")

ASCA_FEATURES = [
    "snr", "rms_energy", "spectral_bandwidth", "spectral_flatness", "spectral_centroid",
    "zero_crossing_rate", "spectral_rolloff", "spectral_contrast", "spectral_flux",
    "energy_entropy", "mfcc_mean", "mfcc_std",
]
CONFIDENCE_FEATURES = ["max_probability", "prediction_margin", "entropy"]
ROUTER_FEATURES = ASCA_FEATURES + CONFIDENCE_FEATURES

ROUTER_WINDOW_SEC = 3.0


# --------------------------------------------------------------------------------------------
# feature extraction -- exact ports of notebook 04's extractors, per serving_spec.json
# --------------------------------------------------------------------------------------------
def mfcc_72(y: np.ndarray, sr: int) -> np.ndarray:
    """Light branch input. FULL-length audio, MFCC n_fft=400 / hop=160, coefficient 0 dropped,
    mean+std pooled -> 72-d."""
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
    d1 = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)
    stacked = np.vstack([m[1:], d1[1:], d2[1:]])  # drop coefficient 0 -> (36, T)
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])


def prepare_router_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Exactly 3.00s for the ASCA path: zero-pad the end if short, take the FIRST 3s if long.

    OPEN ITEM (notebook 29 §10): the system was only ever evaluated on clips whose first 3
    seconds WERE the whole clip. Taking the opening of a 30-60s recording is a placeholder, not
    a validated choice -- whether early-clip routing behaves like whole-clip routing is unchecked.
    The UI says so too. Revisit before treating routing decisions on long clips as trustworthy.
    """
    n = int(round(ROUTER_WINDOW_SEC * sr))
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def asca_12(y: np.ndarray, sr: int) -> np.ndarray:
    """Router's first 12 features. librosa DEFAULT n_fft/hop (2048/512) throughout -- NOT the
    light branch's 400/160. Order matches ASCA_FEATURES."""
    # Note: this SNR is the notebook's own definition. Because `noise` is the signal minus its
    # own mean, it is ~0 for any zero-mean audio -- the training range is [-7.8e-07, 1.5e-02],
    # which confirms it. Kept exactly as calibrated rather than "fixed" into something the
    # router was never fitted on.
    signal_power = np.mean(y ** 2)
    noise = y - np.mean(y)
    snr = 10 * np.log10((signal_power + 1e-10) / (np.mean(noise ** 2) + 1e-10))

    stft = np.abs(librosa.stft(y))
    flux = float(np.mean(np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))))

    rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    prob = rms_frames / (np.sum(rms_frames) + 1e-10)
    energy_entropy = float(-np.sum(prob * np.log2(prob + 1e-10)))

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # all 13 coefficients, c0 included

    return np.array([
        float(snr),
        float(np.sqrt(np.mean(y ** 2))),
        float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        float(np.mean(librosa.feature.spectral_flatness(y=y))),
        float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        float(np.mean(librosa.feature.zero_crossing_rate(y))),
        float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))),
        flux,
        energy_entropy,
        float(np.mean(mf)),
        float(np.std(mf)),
    ])


def confidence_features(probs: np.ndarray) -> np.ndarray:
    """The router's last 3 columns, derived from the light branch's own probabilities.
    Deliberately NOT scaled -- the router was calibrated on these raw."""
    srt = np.sort(probs)[::-1]
    return np.array([
        float(srt[0]),                                      # max_probability
        float(srt[0] - srt[1]),                             # prediction_margin
        float(-np.sum(probs * np.log(probs + 1e-12))),      # entropy (nats)
    ])


# --------------------------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class AdaptivePackage:
    mfcc_scaler: object
    asca_scaler: object
    light: object
    router: object
    heavy: object
    threshold: float
    classes: list[str]
    spec: dict
    demo_clips: list[dict]
    artifacts_dir: Path

    @classmethod
    def load(cls, config: Config) -> "AdaptivePackage":
        d = config.adaptive_artifacts_dir
        spec = json.loads((d / "serving_spec.json").read_text(encoding="utf-8"))

        # The threshold is a Learn-then-Test certified value, not a tuning knob. It comes from
        # the spec, and config.yaml may only agree with it -- a mismatch means someone changed
        # one without re-running the calibration, so refuse to start rather than serve decisions
        # against an uncertified bound.
        spec_threshold = float(spec["router_input"]["threshold"])
        if config.adaptive_threshold is not None and abs(config.adaptive_threshold - spec_threshold) > 1e-12:
            raise RuntimeError(
                f"adaptive.threshold in config.yaml ({config.adaptive_threshold}) does not match "
                f"serving_spec.json's certified threshold ({spec_threshold}). That number was "
                "fixed by a Learn-then-Test calibration run; changing it needs a new calibration, "
                "not an edit. Refusing to start."
            )

        classes = list(spec["emotions_in_order"])
        manifest_path = d / "demo_clips_manifest.json"
        demo = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"] if manifest_path.exists() else []

        pkg = cls(
            mfcc_scaler=joblib.load(d / "mfcc_scaler.joblib"),
            asca_scaler=joblib.load(d / "asca_minmax_scaler.joblib"),
            light=joblib.load(d / "light_mfcc_mlp.joblib"),
            router=joblib.load(d / "router_recalibrated.joblib"),
            heavy=joblib.load(d / "heavy_e2vplus_lr.joblib"),
            threshold=spec_threshold,
            classes=classes,
            spec=spec,
            demo_clips=demo,
            artifacts_dir=d,
        )

        # Cheap structural gate -- 72 vs 12 catches a scaler wired to the wrong branch, which is
        # otherwise silent.
        if pkg.mfcc_scaler.n_features_in_ != 72 or pkg.asca_scaler.n_features_in_ != 12:
            raise RuntimeError(
                f"Scaler shapes are wrong (mfcc={pkg.mfcc_scaler.n_features_in_} expected 72, "
                f"asca={pkg.asca_scaler.n_features_in_} expected 12) -- the two scalers are "
                "probably swapped."
            )
        if list(getattr(pkg.router, "feature_names_in_", ROUTER_FEATURES)) != ROUTER_FEATURES:
            raise RuntimeError("Router feature order does not match serving_spec.json.")

        logger.info("AdaptivePackage ready: threshold=%.3f classes=%s demo_clips=%d",
                    pkg.threshold, classes, len(demo))
        return pkg


# --------------------------------------------------------------------------------------------
# the routed decision
# --------------------------------------------------------------------------------------------
def _probs_dict(classes: list[str], probs: np.ndarray) -> dict[str, float]:
    return {c: float(p) for c, p in zip(classes, probs)}


def route(y: np.ndarray, sr: int, adaptive: AdaptivePackage, model: ModelPackage) -> dict:
    """Run the light branch, score the router, and run the heavy branch only if it escalates.

    `model` supplies the already-loaded Emotion2Vec+ encoder -- the heavy branch reuses it rather
    than loading a second 94M-parameter copy.
    """
    # --- light branch: FULL audio -------------------------------------------------------------
    raw_mfcc = mfcc_72(y, sr)
    light_probs = adaptive.light.predict_proba(
        adaptive.mfcc_scaler.transform(raw_mfcc.reshape(1, -1))
    )[0]
    light_idx = int(np.argmax(light_probs))

    # --- router: FIRST 3s, different extractor ------------------------------------------------
    raw_asca = asca_12(prepare_router_audio(y, sr), sr)
    # The MinMaxScaler does not clip: a live clip outside the training range scales outside
    # [0, 1], which is what the router was calibrated on. Left as-is on purpose.
    scaled_asca = adaptive.asca_scaler.transform(raw_asca.reshape(1, -1))[0]
    row = np.concatenate([scaled_asca, confidence_features(light_probs)]).reshape(1, -1)

    with warnings.catch_warnings():
        # The router was fitted on a named DataFrame; passing a correctly-ordered array is
        # numerically identical (verified to 8.6e-08) and avoids a pandas dependency here.
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        router_score = float(adaptive.router.predict_proba(row)[0][1])

    escalate = router_score >= adaptive.threshold

    out = {
        "light_prediction": adaptive.classes[light_idx],
        "light_confidence": float(light_probs[light_idx]),
        "light_probabilities": _probs_dict(adaptive.classes, light_probs),
        "router_score": router_score,
        "router_threshold": adaptive.threshold,
        "branch_used": "heavy" if escalate else "light",
        "heavy_prediction": None,
        "heavy_confidence": None,
        "heavy_probabilities": None,
        "router_audio_sec": ROUTER_WINDOW_SEC,
        "router_audio_is_whole_clip": bool(len(y) <= int(round(ROUTER_WINDOW_SEC * sr))),
    }

    # --- heavy branch: only when escalated ----------------------------------------------------
    # Never run speculatively just to populate the fields -- the whole point is not paying for it.
    if escalate:
        embedding = embed_raw(y, sr, model)
        heavy_probs = adaptive.heavy.predict_proba(embedding.reshape(1, -1))[0]
        heavy_idx = int(np.argmax(heavy_probs))
        out["heavy_prediction"] = adaptive.classes[heavy_idx]
        out["heavy_confidence"] = float(heavy_probs[heavy_idx])
        out["heavy_probabilities"] = _probs_dict(adaptive.classes, heavy_probs)

    out["final_prediction"] = out["heavy_prediction"] if escalate else out["light_prediction"]
    out["final_confidence"] = out["heavy_confidence"] if escalate else out["light_confidence"]
    return out
