"""Vercel entrypoint: prepare the serverless environment, then expose the normal SER API app.

Everything model-related is the same ser_api package the local server runs; this file only adapts
to Vercel's sandbox (read-only filesystem apart from /tmp, one vCPU, bundles that drop .pyi files).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# One vCPU: one BLAS/OpenMP thread. (funasr's own thread count comes from encoder.ncpu in config.)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

# Only /tmp is writable on Vercel; point every library cache there.
if os.environ.get("VERCEL"):
    for _k, _v in {"HOME": "/tmp", "MODELSCOPE_CACHE": "/tmp/modelscope", "NUMBA_CACHE_DIR": "/tmp/numba",
                   "HF_HOME": "/tmp/hf", "XDG_CACHE_HOME": "/tmp/cache", "MPLCONFIGDIR": "/tmp/mpl"}.items():
        os.environ[_k] = _v

os.environ.setdefault("SER_API_CONFIG", os.path.join(HERE, "config.yaml"))

# Vercel's bundler drops .pyi files, but librosa reads its __init__.pyi stubs at import time to know
# what to lazy-load. When an original is missing, use the copy shipped in ./stubs.
import lazy_loader  # noqa: E402

_STUBS = os.path.join(HERE, "stubs")
_attach_stub = lazy_loader.attach_stub


def _attach_stub_with_fallback(package_name, filename):
    stub = os.path.splitext(filename)[0] + ".pyi"
    if not os.path.exists(stub):
        shipped = os.path.join(_STUBS, *package_name.split("."), "__init__.pyi")
        if os.path.exists(shipped):
            filename = shipped
    return _attach_stub(package_name, filename)


lazy_loader.attach_stub = _attach_stub_with_fallback

from ser_api.main import app  # noqa: E402,F401
