"""Adaptive routing: a cheap MFCC+MLP "light" branch runs on every clip, a calibrated router
decides whether the clip is worth paying the ~24x cost of the Emotion2Vec+ "heavy" branch, and
the heavy branch runs ONLY when it escalates.

Which files make up the system is set by the serving spec config.yaml selects:
  serving_spec_v2.json (notebook 31): light + router retrained on real voices, heavy head = the
      product's tier1b classifier, so the site gives one answer per recording.
  serving_spec.json (notebooks 28/29): the original acted-speech system with its own heavy head.
No model file is modified here.

THE TRAP (serving_spec.json's own warning, repeated because it fails silently):
the light branch and the router consume DIFFERENT audio from the same recording. Swapping them
raises no error -- the models just return confident, wrong numbers.

    light branch : FULL-length audio, MFCC with n_fft=400  hop=160
    router (ASCA): FIRST 3.00s only,  librosa DEFAULTS n_fft=2048 hop=512

Both branches' scalers are applied after raw extraction and before the model. The router's three
confidence features are derived live from the light branch's probabilities and are NOT scaled.

The v1 system is verified against Router/demo_clips_manifest.json on all three demo clips: light
confidence to 1.8e-07, router score to 8.6e-08, heavy confidence to 1.1e-06. Notebook 31 uses the
same MFCC and ASCA recipes verbatim, so v2 shares this serving path.
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
from ser_api.explainability import acoustic_evidence, linear_windows, make_windows
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
def _check_classes(est, classes: list[str], what: str) -> None:
    """Probabilities are mapped to emotions by position, so the model's class order must be the
    spec's emotion order -- either the names themselves or their indices 0..5."""
    got = [str(c) for c in getattr(est, "classes_", [])]
    if got not in (classes, [str(i) for i in range(len(classes))]):
        raise RuntimeError(f"{what} class order {got} does not match the spec's emotions {classes}.")


def _load_heavy(config: Config, d: Path, spec: dict) -> tuple[object, object, str]:
    """(scaler, linear classifier, file name) of the heavy head the spec names. v2 names the
    product's main model (tier1b) so the site gives one answer; v1 used its own Emotion2Vec+ head."""
    head = (spec.get("heavy_branch") or {}).get("head") or "heavy_e2vplus_lr.joblib"
    if head == config.classifier_artifact_path.name:
        bundle = joblib.load(config.classifier_artifact_path)
        if config.classifier_expected_config and bundle.get("config") != config.classifier_expected_config:
            raise RuntimeError(f"Heavy head {head} is not the expected {config.classifier_expected_config!r}.")
        return bundle["scaler"], bundle["clf"], head
    obj = joblib.load(d / head)
    if isinstance(obj, dict):
        return obj["scaler"], obj["clf"], head
    return obj.named_steps["scaler"], obj.named_steps["clf"], head


@dataclasses.dataclass
class AdaptivePackage:
    mfcc_scaler: object
    asca_scaler: object
    light: object
    router: object
    heavy_scaler: object
    heavy_clf: object
    heavy_name: str
    threshold: float
    classes: list[str]
    spec: dict
    spec_name: str
    demo_clips: list[dict]
    artifacts_dir: Path

    def heavy_proba(self, embedding: np.ndarray) -> np.ndarray:
        return self.heavy_clf.predict_proba(self.heavy_scaler.transform(embedding.reshape(1, -1)))[0]

    @classmethod
    def load(cls, config: Config) -> "AdaptivePackage":
        d = config.adaptive_artifacts_dir
        spec_name = config.adaptive_spec
        spec = json.loads((d / spec_name).read_text(encoding="utf-8"))

        # The threshold comes from a Learn-then-Test calibration run, not a tuning knob. It comes
        # from the spec, and config.yaml may only agree with it -- a mismatch means someone changed
        # one without re-running the calibration, so refuse to start rather than serve decisions
        # against an uncalibrated bound.
        spec_threshold = float(spec["router_input"]["threshold"])
        if config.adaptive_threshold is not None and abs(config.adaptive_threshold - spec_threshold) > 1e-12:
            raise RuntimeError(
                f"adaptive.threshold in config.yaml ({config.adaptive_threshold}) does not match "
                f"{spec_name}'s certified threshold ({spec_threshold}). That number was "
                "fixed by a Learn-then-Test calibration run; changing it needs a new calibration, "
                "not an edit. Refusing to start."
            )

        classes = list(spec["emotions_in_order"])
        manifest_path = d / "demo_clips_manifest.json"
        demo = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"] if manifest_path.exists() else []

        mfcc_path, asca_path, router_input = spec["mfcc_path"], spec["asca_path"], spec["router_input"]
        if not router_input.get("artifact"):
            raise RuntimeError(f"{spec_name} has no router file (constant router); not supported.")
        heavy_scaler, heavy_clf, heavy_name = _load_heavy(config, d, spec)

        pkg = cls(
            mfcc_scaler=joblib.load(d / mfcc_path["scaler"]),
            asca_scaler=joblib.load(d / asca_path["scaler"]),
            light=joblib.load(d / mfcc_path.get("model", "light_mfcc_mlp.joblib")),
            router=joblib.load(d / router_input["artifact"]),
            heavy_scaler=heavy_scaler,
            heavy_clf=heavy_clf,
            heavy_name=heavy_name,
            threshold=spec_threshold,
            classes=classes,
            spec=spec,
            spec_name=spec_name,
            demo_clips=demo,
            artifacts_dir=d,
        )
        _check_classes(pkg.light, classes, "Light model")
        _check_classes(pkg.heavy_clf, classes, f"Heavy head {heavy_name}")

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

        logger.info("AdaptivePackage ready: spec=%s threshold=%.3f heavy=%s demo_clips=%d",
                    spec_name, pkg.threshold, heavy_name, len(demo))
        return pkg


# --------------------------------------------------------------------------------------------
# the routed decision
# --------------------------------------------------------------------------------------------
def _probs_dict(classes: list[str], probs: np.ndarray) -> dict[str, float]:
    return {c: float(p) for c, p in zip(classes, probs)}


def route(y: np.ndarray, sr: int, adaptive: AdaptivePackage, model: ModelPackage) -> dict:
    """Run the light branch, score the router, and run the heavy branch only if it escalates.

    `model` supplies the already-loaded Emotion2Vec+ encoder -- the heavy branch reuses it rather
    than loading a second 94M-parameter copy. Its classifier head is whichever the spec names.
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
        heavy_probs = adaptive.heavy_proba(embedding)
        heavy_idx = int(np.argmax(heavy_probs))
        out["heavy_prediction"] = adaptive.classes[heavy_idx]
        out["heavy_confidence"] = float(heavy_probs[heavy_idx])
        out["heavy_probabilities"] = _probs_dict(adaptive.classes, heavy_probs)

    out["final_prediction"] = out["heavy_prediction"] if escalate else out["light_prediction"]
    out["final_confidence"] = out["heavy_confidence"] if escalate else out["light_confidence"]
    return out


# --------------------------------------------------------------------------------------------
# one answer + its timeline: what the website shows
# --------------------------------------------------------------------------------------------
MODES = ("gie", "heavy")
LIGHT_MIN_WINDOW_SEC = 1.0  # MFCC deltas need ~9 frames; a sliver of a last window joins the one before


def light_windows(y: np.ndarray, sr: int, adaptive: AdaptivePackage, pred_idx: int,
                  window_sec: float) -> tuple[list[dict], int]:
    """Timeline for an answer the light model gave. The MLP is not linear, so there is no exact
    per-window split of its score; each window is instead run through the light model on its own,
    and its contribution is how far above (or below) a six-way coin toss it rates the answer."""
    spans = make_windows(len(y), sr, window_sec)
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < LIGHT_MIN_WINDOW_SEC * sr:
        spans = spans[:-2] + [(spans[-2][0], spans[-1][1])]
    chance = 1.0 / len(adaptive.classes)
    windows = []
    for k, (a, b) in enumerate(spans):
        wy = y[a:b]
        p = adaptive.light.predict_proba(adaptive.mfcc_scaler.transform(mfcc_72(wy, sr).reshape(1, -1)))[0]
        windows.append({
            "window_idx": k, "start_s": round(a / sr, 2), "end_s": round(b / sr, 2),
            "contribution": float(p[pred_idx] - chance), **acoustic_evidence(wy, sr),
        })
    total_abs = sum(abs(w["contribution"]) for w in windows) or 1.0
    for w in windows:
        w["contribution_pct"] = round(100 * w["contribution"] / total_abs, 1)
    return windows, max(windows, key=lambda w: w["contribution"])["window_idx"]


def analyze(y: np.ndarray, sr: int, adaptive: AdaptivePackage, model: ModelPackage,
            mode: str = "gie", window_sec: float = 5.0) -> dict:
    """The website's single answer for a recording.

    gie   : Green Inference Engine -- light model, router, and the heavy model only if escalated.
    heavy : heavy model on every clip; the light model and router are never run.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    routing = None
    if mode == "heavy":
        embedding = embed_raw(y, sr, model)
        probs = adaptive.heavy_proba(embedding)
        branch = "heavy"
    else:
        routing = route(y, sr, adaptive, model)
        branch = routing["branch_used"]
        source = routing["heavy_probabilities"] if branch == "heavy" else routing["light_probabilities"]
        probs = np.array([source[c] for c in adaptive.classes])
        embedding = embed_raw(y, sr, model) if branch == "heavy" else None  # cached by route()

    idx = int(np.argmax(probs))
    if branch == "heavy":
        windows, top_idx, exact, recon = linear_windows(
            y, sr, lambda wy: embed_raw(wy, sr, model), adaptive.heavy_scaler, adaptive.heavy_clf,
            idx, embedding, window_sec,
        )
        method = "linear"
    else:
        windows, top_idx = light_windows(y, sr, adaptive, idx, window_sec)
        exact, recon, method = False, None, "light_window_probability"

    return {
        "mode": mode,
        "branch_used": branch,
        "emotion": adaptive.classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(adaptive.classes, probs)},
        "routing": routing,
        "timeline": {
            "window_sec": window_sec,
            "n_windows": len(windows),
            "method": method,
            "attribution_exact": exact,
            "reconstruction_error": None if recon is None else round(recon, 6),
            "top_window_idx": top_idx,
            "windows": windows,
        },
    }
