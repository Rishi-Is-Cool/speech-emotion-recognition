#!/usr/bin/env python3
"""Manual smoke test against a RUNNING server (real model, real audio) -- not part of the pytest
suite, which uses a mocked encoder. Run this once the real artifacts are in place, against a
recording you already know the answer for, before trusting the service.

Usage:
    python scripts/manual_test.py path/to/recording.wav [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print(f"health: {health}")
    if not health.get("model_loaded"):
        print("Model is not loaded yet -- wait for the server to finish starting up.")
        sys.exit(1)

    version = requests.get(f"{args.url}/model/version", timeout=10).json()
    print(f"model version: {version['model_version']}  classes: {version['classes']}\n")

    with open(args.audio_path, "rb") as f:
        resp = requests.post(f"{args.url}/predict", files={"file": f}, timeout=120)

    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.json()}")
        sys.exit(1)

    body = resp.json()
    print(f"emotion:    {body['emotion']}  ({body['confidence'] * 100:.1f}% confidence)")
    print(f"duration:   {body['duration_sec']}s")
    print("probabilities:")
    for cls, p in sorted(body["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:10s} {p * 100:5.1f}%")

    tl = body["timeline"]
    print(f"\ntimeline: {tl['n_windows']} window(s), attribution_exact={tl['attribution_exact']}, "
          f"reconstruction_error={tl['reconstruction_error']:.4g}")
    for w in tl["windows"]:
        marker = "  <-- strongest" if w["window_idx"] == tl["top_window_idx"] else ""
        pitch = f"{w['pitch_hz']:.0f}Hz" if w["pitch_hz"] else "n/a"
        print(f"  {w['start_s']:>5.1f}s-{w['end_s']:>5.1f}s  contribution={w['contribution']:+.3f} "
              f"({w['contribution_pct']:+.1f}%)  pitch={pitch}  energy={w['energy_rms']:.4f}{marker}")

    if body["warnings"]:
        print("\nwarnings:")
        for w in body["warnings"]:
            print(f"  - {w}")

    print(f"\nlatency: {json.dumps(body['latency_ms'])}")


if __name__ == "__main__":
    main()
