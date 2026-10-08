"""embed_raw(): the one canonical embedding call, used identically for the whole clip and for
every explainability window (SER_Decision_Timeline_Architecture.md's canonical contract, same
call shape verified in notebook 11 / used throughout notebooks 12-23). predict(): whole-clip
embedding -> locked classifier -- this is the number that ships as the headline result and is
untouched by anything explainability does."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import tempfile
import threading
from collections import OrderedDict

import numpy as np
import soundfile as sf

from ser_api.model_package import ModelPackage

# The page asks for the headline result and the routing view separately, and both embed the same
# whole clip; on a 1-vCPU host that encoder pass is most of the cost. Remember the last few
# embeddings, keyed by a fingerprint of the exact samples, so the second request reuses the first's
# result. Stores 768 numbers and a hash per entry, never audio. Identical input -> identical output,
# so this cannot change any result.
_EMBED_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_EMBED_CACHE_SIZE = 8
_EMBED_CACHE_LOCK = threading.Lock()


def embed_raw(y: np.ndarray, sr: int, model: ModelPackage) -> np.ndarray:
    key = f"{hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()}:{y.dtype}:{sr}:{model.version_tag}"
    with _EMBED_CACHE_LOCK:
        hit = _EMBED_CACHE.get(key)
        if hit is not None:
            _EMBED_CACHE.move_to_end(key)
            return hit.copy()
    emb = _embed_uncached(y, sr, model)
    with _EMBED_CACHE_LOCK:
        _EMBED_CACHE[key] = emb.copy()
        while len(_EMBED_CACHE) > _EMBED_CACHE_SIZE:
            _EMBED_CACHE.popitem(last=False)
    return emb


def _embed_uncached(y: np.ndarray, sr: int, model: ModelPackage) -> np.ndarray:
    """Write `y` to a temp wav and run the canonical funasr call. No raw audio is kept after
    this call returns -- the temp file is removed in a `finally`, per the no-persistent-raw-audio
    logging rule in SER_Audit_and_Production_Architecture.md section 4."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_path, y, sr)
        out = model.encoder.generate(input=tmp_path, **model.encoder_generate_kwargs)
        return np.asarray(out[0]["feats"]).reshape(-1)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@dataclasses.dataclass
class PredictionResult:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    embedding: np.ndarray  # exposed for explainability's attribution math, not serialized as-is
    model_version: str


def predict(y: np.ndarray, sr: int, model: ModelPackage) -> PredictionResult:
    embedding = embed_raw(y, sr, model)
    scaled = model.scaler.transform(embedding.reshape(1, -1))
    probs = model.clf.predict_proba(scaled)[0]
    pred_idx = int(np.argmax(probs))

    return PredictionResult(
        emotion=model.classes[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities={c: float(p) for c, p in zip(model.classes, probs)},
        embedding=embedding,
        model_version=model.version_tag,
    )
