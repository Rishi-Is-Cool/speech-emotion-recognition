"""End-to-end tests for the SER API, against a mocked funasr encoder (no GPU, no network,
CI-safe) -- mirrors the mocking discipline used throughout this project's notebook tests
(nb22/nb23): the mock is seeded on the audio's actual CONTENT (not filename/offset) so identical
samples always produce identical embeddings, matching real model determinism. This is what makes
the single-window reconstruction-exactness assertion meaningful rather than accidental.

Run with: pytest tests/ -v   (from the ser_api/ directory, after `pip install -r requirements.txt`)
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import types

import joblib
import numpy as np
import pytest
import soundfile as sf
import yaml

EM = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
FREQS = {e: 200 + 150 * i for i, e in enumerate(EM)}
SR = 16000


# ---------------------------------------------------------------------------------------------
# Mock funasr -- installed into sys.modules BEFORE ser_api.main / ser_api.model_package are
# imported anywhere, since ModelPackage.load() does `import funasr` lazily inside the method.
# ---------------------------------------------------------------------------------------------
def _class_from_freq(f0):
    return min(EM, key=lambda e: abs(FREQS[e] - f0))


_FRAME_SEC = 0.25


def _install_mock_funasr():
    fa = types.ModuleType("funasr")

    class AutoModel:
        def __init__(self, **kwargs):
            pass

        def generate(self, input, **kwargs):
            y, sr = sf.read(input, dtype="float32")
            n = len(y)
            frame_n = max(1, int(_FRAME_SEC * sr))
            frame_vecs = []
            for start in range(0, n, frame_n):
                chunk = y[start:start + frame_n]
                if len(chunk) < 8:
                    continue
                spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) + 1e-12
                freqs = np.fft.rfftfreq(len(chunk), d=1 / sr)
                peak_freq = float(freqs[np.argmax(spec)])
                cls = _class_from_freq(peak_freq)
                idx = EM.index(cls)
                seed = int(hashlib.sha256(chunk.tobytes()).hexdigest(), 16) % (2**32)
                local_rng = np.random.default_rng(seed)
                vec = local_rng.normal(0, 0.15, 8).astype(np.float32)
                vec += 3.0 * idx
                frame_vecs.append(vec)
            if not frame_vecs:
                frame_vecs = [np.zeros(8, dtype=np.float32)]
            pooled = np.mean(frame_vecs, axis=0)
            return [{"feats": pooled.reshape(1, -1)}]

    fa.AutoModel = AutoModel
    sys.modules["funasr"] = fa


_install_mock_funasr()


# ---------------------------------------------------------------------------------------------
# Fixtures: a real classifier bundle trained on synthetic embeddings compatible with the mock
# above, plus a config.yaml pointing at it -- SER_API_CONFIG is set before ser_api is imported
# so Config's module-level default path picks it up.
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ser_api_fixtures")

    import sklearn.linear_model
    import sklearn.preprocessing

    rng = np.random.default_rng(0)
    Xtr, ytr = [], []
    for i, e in enumerate(EM):
        Xtr.append(rng.normal(loc=3.0 * i, scale=1.0, size=(20, 8)))
        ytr += [e] * 20
    Xtr = np.concatenate(Xtr).astype(np.float32)
    scaler = sklearn.preprocessing.StandardScaler().fit(Xtr)
    clf = sklearn.linear_model.LogisticRegression(max_iter=500).fit(scaler.transform(Xtr), np.array(ytr))

    artifact_path = d / "tier1b_best_head.joblib"
    joblib.dump(
        {"scaler": scaler, "clf": clf, "config": "C: acted+MELD+IEMOCAP", "emotions": EM},
        artifact_path,
    )

    config = {
        "encoder": {
            "model_id": "iic/emotion2vec_plus_base",
            "hub": "ms",
            "generate_kwargs": {"granularity": "utterance", "extract_embedding": True},
        },
        "classifier": {
            "artifact_path": str(artifact_path),
            "expected_config": "C: acted+MELD+IEMOCAP",
        },
        "audio": {
            "sample_rate": SR,
            "min_duration_sec": 1.0,
            "max_duration_sec": 20.0,
            "silence_rms_threshold": 0.001,
        },
        "explainability": {"window_sec": 5.0},
        "server": {"max_upload_mb": 1, "log_level": "info"},
    }
    config_path = d / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    return {"dir": d, "config_path": config_path}


@pytest.fixture(scope="session")
def client(fixture_dir):
    os.environ["SER_API_CONFIG"] = str(fixture_dir["config_path"])
    # Import AFTER the env var and mock are both in place.
    from fastapi.testclient import TestClient
    from ser_api.main import app

    with TestClient(app) as c:
        yield c


def _wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _tone(duration_sec: float, emotion: str, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration_sec)) / sr
    f0 = FREQS[emotion]
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    return (0.6 * env * np.sin(2 * np.pi * f0 * t) + 0.05 * np.sin(2 * np.pi * f0 * 2 * t)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    # the fixture config has no adaptive section and doesn't switch YouTube off
    assert body["features"] == {"youtube": True, "adaptive": False}


def test_youtube_endpoint_refuses_when_disabled(client, monkeypatch):
    """A public deployment turns YouTube off; the endpoint must then refuse rather than download."""
    import ser_api.main as main_module

    monkeypatch.setattr(main_module._state["config"], "youtube_enabled", False)
    r = client.post("/predict/youtube", json={"url": "https://www.youtube.com/watch?v=abc123"})
    assert r.status_code == 404
    assert client.get("/health").json()["features"]["youtube"] is False


def test_model_version(client):
    r = client.get("/model/version")
    assert r.status_code == 200
    body = r.json()
    assert body["classes"] == EM
    assert body["window_sec"] == 5.0
    assert body["sample_rate"] == SR
    assert body["classifier_config"] == "C: acted+MELD+IEMOCAP"


def test_predict_short_clip_is_single_window_and_exact(client):
    """A clip shorter than window_sec must produce exactly one window with (near-)zero
    reconstruction error -- exact by construction, per the Stage 5 finding."""
    audio = _tone(3.0, "angry")
    r = client.post("/predict", files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["emotion"] in EM
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3

    tl = body["timeline"]
    assert tl["n_windows"] == 1
    assert tl["attribution_exact"] is True
    assert tl["reconstruction_error"] < 1e-6
    assert len(tl["windows"]) == 1
    assert body["warnings"] == []


def test_predict_long_clip_is_multi_window_and_flagged_approximate(client):
    """A 12s clip at a 5s window must split into 3 windows (5+5+2) and be flagged
    non-exact -- this is the actual case the Stage 5 bonus test proved matters."""
    audio = _tone(12.0, "sad")
    r = client.post("/predict", files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()

    tl = body["timeline"]
    assert tl["n_windows"] == 3
    assert tl["attribution_exact"] is False
    assert len(tl["windows"]) == 3
    starts_ends = [(w["start_s"], w["end_s"]) for w in tl["windows"]]
    for a, b in zip(starts_ends, starts_ends[1:]):
        assert a[1] == b[0], f"windows overlap or leave a gap: {starts_ends}"
    assert abs(starts_ends[-1][1] - 12.0) < 1e-6
    assert len(body["warnings"]) == 1
    assert "approximate" in body["warnings"][0]


def test_predict_correct_headline_on_a_clear_tone(client):
    """Sanity: the mock's classes are well-separated by frequency, so a clean single-emotion
    tone should be classified correctly -- catches a broken wiring between predict() and the
    classifier, not just structural correctness."""
    audio = _tone(2.0, "happy")
    r = client.post("/predict", files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["emotion"] == "happy"


def test_predict_rejects_too_short(client):
    audio = _tone(0.5, "angry")
    r = client.post("/predict", files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error_code"] == "too_short"


def test_predict_rejects_too_long(client):
    audio = _tone(25.0, "angry")  # fixture config caps max_duration_sec at 20
    r = client.post("/predict", files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error_code"] == "too_long"


def test_predict_rejects_silence(client):
    audio = np.zeros(int(SR * 3.0), dtype=np.float32)
    r = client.post("/predict", files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error_code"] == "silent_audio"


def test_predict_rejects_unreadable_file(client):
    r = client.post("/predict", files={"file": ("not_audio.wav", b"this is not a wav file", "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error_code"] == "unreadable_audio"


def test_predict_rejects_empty_upload(client):
    r = client.post("/predict", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error_code"] == "empty_upload"


def test_predict_rejects_oversized_upload(client):
    # fixture config sets max_upload_mb = 1
    big = b"0" * (2 * 1024 * 1024)
    r = client.post("/predict", files={"file": ("big.wav", big, "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error_code"] == "file_too_large"


def _install_mock_ytdlp(monkeypatch, *, duration=6.0, emotion="happy", title="a test video",
                         fail_with=None):
    """Installs a fake ser_api.youtube.fetch_audio -- the real one shells out to yt-dlp+ffmpeg,
    which this CI-safe test suite must not depend on. `fail_with` lets a test make the "download"
    raise a specific SERInputError, e.g. to check /predict/youtube's error mapping end to end."""
    import ser_api.main as main_module

    def fake_fetch_audio(url, config):
        if fail_with is not None:
            raise fail_with
        audio = _tone(duration, emotion)
        return _wav_bytes(audio), {"title": title, "duration_sec": duration}

    monkeypatch.setattr(main_module, "fetch_youtube_audio", fake_fetch_audio)


def test_predict_youtube_happy_path(client, monkeypatch):
    _install_mock_ytdlp(monkeypatch, duration=3.0, emotion="fear", title="a fear clip")
    r = client.post("/predict/youtube", json={"url": "https://www.youtube.com/watch?v=abc123"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["emotion"] == "fear"
    assert body["source"] == {
        "type": "youtube", "url": "https://www.youtube.com/watch?v=abc123", "title": "a fear clip",
    }
    assert body["timeline"]["n_windows"] == 1
    assert any("YouTube" in w for w in body["warnings"])
    assert "download_ms" in body["latency_ms"]

    # The extracted audio must be returned so the frontend can decode it for segment replay,
    # exactly like it does for an uploaded file.
    raw = base64.b64decode(body["audio_base64"])
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    assert sr == SR
    assert abs(len(y) / sr - 3.0) < 0.1


def test_youtube_audio_only_fetches(client, monkeypatch):
    """/youtube/audio hands back the WAV for the page to analyse like a recording -- no model runs."""
    _install_mock_ytdlp(monkeypatch, duration=2.0, emotion="sad", title="a sad clip")
    r = client.post("/youtube/audio", json={"url": "https://youtu.be/abc123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"]["title"] == "a sad clip"
    assert "emotion" not in body
    y, sr = sf.read(io.BytesIO(base64.b64decode(body["audio_base64"])), dtype="float32")
    assert sr == SR and abs(len(y) / sr - 2.0) < 0.1


def test_analyze_needs_the_routing_models(client):
    """The fixture config has no Router artifacts: /analyze must say so, not crash."""
    r = client.post("/analyze", files={"file": ("clip.wav", _wav_bytes(_tone(2.0, "happy")), "audio/wav")})
    assert r.status_code == 503


def test_predict_youtube_rejects_non_youtube_url(client):
    # No mock installed -- _check_url() must reject this before ever calling yt-dlp.
    r = client.post("/predict/youtube", json={"url": "https://example.com/video.mp4"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "invalid_youtube_url"


def test_predict_youtube_rejects_empty_url(client):
    r = client.post("/predict/youtube", json={"url": "   "})
    assert r.status_code == 400
    assert r.json()["error_code"] == "empty_url"


def test_predict_youtube_propagates_too_long(client, monkeypatch):
    from ser_api.errors import SERInputError

    _install_mock_ytdlp(monkeypatch, fail_with=SERInputError("too_long", "video is too long"))
    r = client.post("/predict/youtube", json={"url": "https://youtu.be/abc123"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "too_long"


def test_predict_youtube_maps_unavailable_video(client, monkeypatch):
    from ser_api.errors import SERInputError

    _install_mock_ytdlp(
        monkeypatch,
        fail_with=SERInputError("youtube_unavailable", "private or region-locked"),
    )
    r = client.post("/predict/youtube", json={"url": "https://youtu.be/abc123"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "youtube_unavailable"


def test_model_package_refuses_mismatched_classifier_config(fixture_dir, tmp_path):
    """A classifier bundle whose config doesn't match config.yaml's expected_config must fail
    loudly at startup, not silently serve mislabeled predictions."""
    import joblib
    import sklearn.linear_model
    import sklearn.preprocessing

    from ser_api.config import Config
    from ser_api.model_package import ModelPackage

    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 8))
    y = ["angry"] * 5 + ["sad"] * 5
    scaler = sklearn.preprocessing.StandardScaler().fit(X)
    clf = sklearn.linear_model.LogisticRegression(max_iter=200).fit(scaler.transform(X), y)
    bad_path = tmp_path / "wrong.joblib"
    joblib.dump({"scaler": scaler, "clf": clf, "config": "WRONG CONFIG", "emotions": ["angry", "sad"]}, bad_path)

    raw = yaml.safe_load(open(fixture_dir["config_path"]))
    raw["classifier"]["artifact_path"] = str(bad_path)
    config = Config.from_dict(raw)

    with pytest.raises(RuntimeError, match="does not match"):
        ModelPackage.load(config)
