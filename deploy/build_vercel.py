"""Assemble the Vercel deployment folder.

    ser_api\\.venv\\Scripts\\python.exe deploy\\build_vercel.py

Writes to %LOCALAPPDATA%\\ser-emotion-vercel (outside OneDrive, so ~370 MB of model weights aren't
synced to the cloud), then deploy from there with `vercel deploy --prod`. The .vercel project link
in that folder is preserved across rebuilds, so redeploys update the same project.

Needs the project's venv (it uses torch to write the weights-only encoder checkpoint).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "deploy" / "templates" / "vercel"
OUT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ser-emotion-vercel"
ENCODER_CACHE = Path.home() / ".cache" / "modelscope" / "models" / "iic--emotion2vec_plus_base" / "snapshots" / "master"
VENV_SITE = ROOT / "ser_api" / ".venv" / "Lib" / "site-packages"
PART_BYTES = 95 * 1024 * 1024  # Vercel rejects any single file over 100 MB

ROUTER_FILES = [
    "serving_spec.json", "demo_clips_manifest.json", "notebook28_results.json",
    "mfcc_scaler.joblib", "asca_minmax_scaler.joblib", "light_mfcc_mlp.joblib",
    "router_recalibrated.joblib", "heavy_e2vplus_lr.joblib",
]


def clean_except_link(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    keep = {".vercel", "models"}  # project link + the expensive-to-rebuild weights
    for child in d.iterdir():
        if child.name in keep:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def build_encoder(dst: Path):
    """Weights-only copy of the encoder checkpoint (drops ~700 MB of training optimiser state),
    split into <100 MB parts plus a sha256. Verified bit-identical in output to the original."""
    if (dst / "model.pt.sha256").is_file() and any(dst.glob("model.pt.part*")):
        print("encoder parts already built; reusing")
        return
    import torch

    dst.mkdir(parents=True, exist_ok=True)
    for f in ("config.yaml", "configuration.json", "tokens.txt"):
        shutil.copy2(ENCODER_CACHE / f, dst / f)
    ck = torch.load(ENCODER_CACHE / "model.pt", map_location="cpu", weights_only=False)
    slim = {k: v for k, v in ck.items() if k not in ("last_optimizer_state", "optimizer_history")}
    tmp = dst / "model.pt"
    torch.save(slim, tmp)
    data = tmp.read_bytes()
    tmp.unlink()
    for i in range(0, len(data), PART_BYTES):
        (dst / f"model.pt.part{i // PART_BYTES}").write_bytes(data[i:i + PART_BYTES])
    (dst / "model.pt.sha256").write_text(hashlib.sha256(data).hexdigest())
    print(f"encoder: {len(data) / 1e6:.0f} MB in {len(range(0, len(data), PART_BYTES))} parts")


def main():
    clean_except_link(OUT)
    for name in ("app.py", "pyproject.toml", "vercel.json", "config.yaml"):
        shutil.copy2(TEMPLATES / name, OUT / name)

    shutil.copytree(ROOT / "ser_api" / "ser_api", OUT / "ser_api",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "config.yaml", "artifacts"))
    (OUT / "artifacts").mkdir()
    shutil.copy2(ROOT / "ser_api" / "ser_api" / "artifacts" / "tier1b_best_head.joblib",
                 OUT / "artifacts" / "tier1b_best_head.joblib")

    (OUT / "Router").mkdir()
    for name in ROUTER_FILES:
        shutil.copy2(ROOT / "Router" / name, OUT / "Router" / name)
    for wav in (ROOT / "Router").glob("*.wav"):
        shutil.copy2(wav, OUT / "Router" / wav.name)

    for sub in ("", "core", "feature", "util"):  # librosa's lazy-loading stubs (see app.py)
        d = OUT / "stubs" / "librosa" / sub
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(VENV_SITE / "librosa" / sub / "__init__.pyi", d / "__init__.pyi")

    (OUT / "public").mkdir()
    shutil.copy2(ROOT / "ser_frontend.html", OUT / "public" / "index.html")  # same-origin API

    build_encoder(OUT / "models" / "emotion2vec_plus_base")

    big = [p for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > 100 * 1000 * 1000 and ".vercel" not in p.parts]
    assert not big, f"files over Vercel's 100 MB limit: {big}"
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and ".vercel" not in p.parts)
    print(f"vercel package -> {OUT}  ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
