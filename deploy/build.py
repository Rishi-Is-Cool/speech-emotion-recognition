"""Assemble the two deployable folders from the working project.

    python deploy/build.py --api-url https://OWNER-SPACENAME.hf.space

Produces
    deploy/out/space/  push to a Hugging Face Space (Docker): the API, the models, and the page
    deploy/out/web/    deploy to Vercel (static): the same page, pointed at the Space

It copies an explicit allow-list only. Notebooks, papers, datasets, the local venv and anything
else in the project folder never reach either host.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
OUT = DEPLOY / "out"

ROUTER_FILES = [
    "serving_spec.json", "demo_clips_manifest.json", "notebook28_results.json",
    "mfcc_scaler.joblib", "asca_minmax_scaler.joblib", "light_mfcc_mlp.joblib",
    "router_recalibrated.joblib", "heavy_e2vplus_lr.joblib",
]


def set_api_meta(html: str, url: str) -> str:
    new, n = re.subn(r'<meta name="ser-api" content="[^"]*" />',
                     f'<meta name="ser-api" content="{url}" />', html)
    assert n == 1, "could not find the ser-api meta tag in ser_frontend.html"
    return new


def build_space() -> Path:
    dst = OUT / "space"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in ("Dockerfile", "requirements.txt", "config.yaml", "README.md"):
        shutil.copy2(DEPLOY / "templates" / "space" / name, dst / name)

    shutil.copytree(ROOT / "ser_api" / "ser_api", dst / "ser_api",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "config.yaml", "artifacts"))

    (dst / "artifacts").mkdir()
    shutil.copy2(ROOT / "ser_api" / "ser_api" / "artifacts" / "tier1b_best_head.joblib",
                 dst / "artifacts" / "tier1b_best_head.joblib")

    router = ROOT / "Router"
    (dst / "Router").mkdir()
    for name in ROUTER_FILES:
        shutil.copy2(router / name, dst / "Router" / name)
    for wav in router.glob("*.wav"):                 # the three demo clips
        shutil.copy2(wav, dst / "Router" / wav.name)

    # served by the API itself: empty meta = talk to its own origin
    shutil.copy2(ROOT / "ser_frontend.html", dst / "index.html")
    return dst


def build_web(api_url: str) -> Path:
    dst = OUT / "web"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    html = (ROOT / "ser_frontend.html").read_text(encoding="utf-8")
    (dst / "index.html").write_text(set_api_meta(html, api_url.rstrip("/")), encoding="utf-8")
    shutil.copy2(DEPLOY / "templates" / "web" / "vercel.json", dst / "vercel.json")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", required=True, help="public URL of the Space, e.g. https://you-ser.hf.space")
    args = ap.parse_args()
    space, web = build_space(), build_web(args.api_url)

    def size(p: Path) -> str:
        return f"{sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6:.1f} MB"

    print(f"space -> {space}  ({size(space)})")
    for f in sorted(space.rglob("*")):
        if f.is_file():
            print("   ", f.relative_to(space))
    print(f"web   -> {web}  ({size(web)})  api = {args.api_url}")


if __name__ == "__main__":
    main()
