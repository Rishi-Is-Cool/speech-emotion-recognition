"""Tests for adaptive routing.

Unlike tests/test_api.py these run against the REAL Router artifacts rather than a mocked
encoder -- the light branch and the router need no encoder at all, so the single most valuable
assertion available (does the serving path reproduce the numbers notebook 28 recorded?) is cheap
to make. They skip themselves when the Router folder isn't present.

The heavy branch is NOT exercised here: it needs the 94M-parameter encoder. What IS tested is the
guarantee that matters for cost -- that it is never invoked when the router says stay light.

Run with: pytest tests/test_adaptive.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ser_api.config import Config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "ser_api" / "config.yaml"


@pytest.fixture(scope="module")
def config() -> Config:
    # Loaded explicitly rather than via SER_API_CONFIG so a sibling test module's env var
    # cannot redirect these at a synthetic fixture config.
    if not CONFIG_PATH.exists():
        pytest.skip("config.yaml not found")
    return Config.load(CONFIG_PATH)


@pytest.fixture(scope="module")
def adaptive(config):
    d = config.adaptive_artifacts_dir
    if not d or not (d / "serving_spec.json").exists():
        pytest.skip("Router artifacts not present")
    from ser_api.adaptive import AdaptivePackage

    return AdaptivePackage.load(config)


@pytest.fixture(scope="module")
def manifest(config):
    p = config.adaptive_artifacts_dir / "demo_clips_manifest.json"
    if not p.exists():
        pytest.skip("demo manifest not present")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_clip(adaptive, clip):
    import librosa

    y, _ = librosa.load(adaptive.artifacts_dir / clip["exported_filename"], sr=16000, mono=True)
    return y


# ---------------------------------------------------------------------------------------------
# the assertion that actually matters: the serving path reproduces the evaluated system
# ---------------------------------------------------------------------------------------------
def test_light_and_router_reproduce_the_manifest(adaptive, manifest):
    """Every recorded light prediction, light confidence, router score and escalate/keep decision
    must come back out of this serving path. A drift here means the feature assembly is wrong --
    which is silent, because the models still return confident-looking numbers."""
    from ser_api.adaptive import asca_12, confidence_features, mfcc_72, prepare_router_audio

    for clip in manifest["clips"]:
        y = _load_clip(adaptive, clip)

        probs = adaptive.light.predict_proba(
            adaptive.mfcc_scaler.transform(mfcc_72(y, 16000).reshape(1, -1))
        )[0]
        idx = int(np.argmax(probs))

        assert adaptive.classes[idx] == clip["light_prediction"], clip["case"]
        assert probs[idx] == pytest.approx(clip["light_confidence"], abs=1e-5), clip["case"]

        scaled = adaptive.asca_scaler.transform(
            asca_12(prepare_router_audio(y, 16000), 16000).reshape(1, -1)
        )[0]
        row = np.concatenate([scaled, confidence_features(probs)]).reshape(1, -1)
        score = float(adaptive.router.predict_proba(row)[0][1])

        assert score == pytest.approx(clip["router_score_p_heavy"], abs=1e-5), clip["case"]
        assert (score >= adaptive.threshold) == clip["escalated"], clip["case"]


def test_borderline_clip_stays_light_by_a_hair(adaptive, manifest):
    """The borderline clip sits ~0.0002 under the threshold and is kept on the light branch,
    where it is wrong. Guards against anyone 'tidying' the threshold comparison into `>` vs `>=`
    or rounding the score -- either would flip this case and quietly change the certified
    operating point."""
    clip = next(c for c in manifest["clips"] if c["case"] == "borderline")
    assert clip["router_score_p_heavy"] < adaptive.threshold
    assert adaptive.threshold - clip["router_score_p_heavy"] < 1e-3
    assert clip["escalated"] is False
    assert clip["final_prediction"] != clip["true_label"]


# ---------------------------------------------------------------------------------------------
# the trap: two audio preparations, two extractors
# ---------------------------------------------------------------------------------------------
def test_router_audio_is_exactly_three_seconds():
    from ser_api.adaptive import prepare_router_audio

    sr = 16000
    assert len(prepare_router_audio(np.zeros(sr), sr)) == 3 * sr            # short -> padded
    assert len(prepare_router_audio(np.zeros(10 * sr), sr)) == 3 * sr       # long  -> truncated
    assert len(prepare_router_audio(np.zeros(3 * sr), sr)) == 3 * sr        # exact -> unchanged

    long = np.linspace(-1, 1, 10 * sr).astype(np.float32)
    assert np.array_equal(prepare_router_audio(long, sr), long[: 3 * sr])   # the FIRST 3s


def test_the_two_paths_see_different_audio():
    """On a clip longer than the router window the two branches must not be fed the same thing.
    Swapping them raises no error, so this is the cheap structural guard."""
    from ser_api.adaptive import asca_12, mfcc_72, prepare_router_audio

    sr = 16000
    rng = np.random.default_rng(0)
    y = (0.3 * rng.standard_normal(8 * sr)).astype(np.float32)

    assert mfcc_72(y, sr).shape == (72,)
    assert asca_12(prepare_router_audio(y, sr), sr).shape == (12,)
    # the router's view is a strict prefix, not the whole clip
    assert not np.array_equal(prepare_router_audio(y, sr), y)


def test_scaler_shapes_are_gated(adaptive):
    """72 vs 12 is the cheap sanity gate against a scaler wired to the wrong branch."""
    assert adaptive.mfcc_scaler.n_features_in_ == 72
    assert adaptive.asca_scaler.n_features_in_ == 12
    assert list(adaptive.router.feature_names_in_)[:12] == [
        "snr", "rms_energy", "spectral_bandwidth", "spectral_flatness", "spectral_centroid",
        "zero_crossing_rate", "spectral_rolloff", "spectral_contrast", "spectral_flux",
        "energy_entropy", "mfcc_mean", "mfcc_std",
    ]


def test_confidence_features_are_not_scaled():
    """They are appended raw -- the router was calibrated on them that way."""
    from ser_api.adaptive import confidence_features

    p = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
    mx, margin, ent = confidence_features(p)
    assert mx == pytest.approx(0.7)
    assert margin == pytest.approx(0.6)
    assert ent == pytest.approx(float(-np.sum(p * np.log(p + 1e-12))))


# ---------------------------------------------------------------------------------------------
# the cost guarantee
# ---------------------------------------------------------------------------------------------
def test_heavy_branch_is_not_run_when_the_router_keeps_it_light(adaptive, manifest, monkeypatch):
    """The whole point of routing is not paying for the encoder. If `route()` ever calls it
    speculatively the saving disappears, so make that an explicit failure."""
    import ser_api.adaptive as adaptive_mod

    def explode(*a, **k):
        raise AssertionError("the heavy encoder was invoked on a clip that stayed light")

    monkeypatch.setattr(adaptive_mod, "embed_raw", explode)

    easy = next(c for c in manifest["clips"] if c["case"] == "easy")
    out = adaptive_mod.route(_load_clip(adaptive, easy), 16000, adaptive, model=None)

    assert out["branch_used"] == "light"
    assert out["heavy_prediction"] is None
    assert out["heavy_probabilities"] is None
    assert out["final_prediction"] == out["light_prediction"]


def test_threshold_must_match_the_certified_value(config, tmp_path):
    """0.375 is a Learn-then-Test certified value. Disagreeing with serving_spec.json must stop
    the service, not quietly serve decisions against an uncertified operating point."""
    from ser_api.adaptive import AdaptivePackage

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["adaptive"]["threshold"] = 0.5
    tampered = Config.from_dict(raw, base_dir=CONFIG_PATH.parent)

    if not tampered.adaptive_artifacts_dir or not (
        tampered.adaptive_artifacts_dir / "serving_spec.json"
    ).exists():
        pytest.skip("Router artifacts not present")

    with pytest.raises(RuntimeError, match="certified threshold"):
        AdaptivePackage.load(tampered)
