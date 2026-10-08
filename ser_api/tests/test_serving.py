"""Serving-path helpers added for small hosts: embedding reuse and split encoder weights."""

from __future__ import annotations

import hashlib
import types

import numpy as np
import pytest


def _fake_model(calls):
    class Enc:
        def generate(self, input, **kw):
            calls.append(input)
            import soundfile as sf
            y, _ = sf.read(input, dtype="float32")
            return [{"feats": np.full((1, 768), float(np.abs(y).sum()), dtype=np.float32)}]

    return types.SimpleNamespace(encoder=Enc(), encoder_generate_kwargs={}, version_tag="test")


def test_embedding_is_reused_and_identical():
    from ser_api import inference

    inference._EMBED_CACHE.clear()
    calls = []
    model = _fake_model(calls)
    y = np.sin(np.linspace(0, 50, 16000)).astype(np.float32)

    a = inference.embed_raw(y, 16000, model)
    b = inference.embed_raw(y.copy(), 16000, model)
    assert np.array_equal(a, b)
    assert len(calls) == 1, "the second identical request should not run the encoder again"

    a[:] = 0  # callers get copies; mutating one must not poison the cache
    assert np.array_equal(inference.embed_raw(y, 16000, model), b)

    inference.embed_raw(y * 0.5, 16000, model)  # different audio -> a real encoder call
    assert len(calls) == 2


def test_split_weights_rejoin_exactly(tmp_path, monkeypatch):
    from ser_api import model_package

    payload = np.random.default_rng(0).bytes(250_000)
    src = tmp_path / "enc"
    src.mkdir()
    (src / "config.yaml").write_text("x: 1")
    for i in range(3):
        (src / f"model.pt.part{i}").write_bytes(payload[i * 100_000:(i + 1) * 100_000])
    (src / "model.pt.sha256").write_text(hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(model_package.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    out = model_package._materialize_local_encoder(src)
    assert (out / "model.pt").read_bytes() == payload
    assert (out / "config.yaml").read_text() == "x: 1"


def test_split_weights_with_wrong_checksum_refuse_to_load(tmp_path, monkeypatch):
    from ser_api import model_package

    src = tmp_path / "enc"
    src.mkdir()
    (src / "model.pt.part0").write_bytes(b"abc")
    (src / "model.pt.sha256").write_text("0" * 64)
    monkeypatch.setattr(model_package.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    with pytest.raises(RuntimeError, match="checksum"):
        model_package._materialize_local_encoder(src)
