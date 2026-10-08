"""Create (or update) the Hugging Face Space and push the deploy package to it.

    python deploy/publish_space.py                     # create/update the Space, push, print its URL
    python deploy/publish_space.py --cors https://x.vercel.app   # also lock the API to that site

Uses the token from `hf auth login` / huggingface_hub.login(); never takes one on the command line.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

SPACE_NAME = "speech-emotion"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cors", default="", help="comma-separated origins allowed to call the API")
    ap.add_argument("--no-push", action="store_true", help="only update settings, don't re-upload")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    host = api.space_info(repo_id).host or f"{user.lower()}-{SPACE_NAME}.hf.space"
    api_url = host if host.startswith("http") else f"https://{host}"
    print(f"space   : https://huggingface.co/spaces/{repo_id}")
    print(f"api url : {api_url}")

    if not args.no_push:
        subprocess.run([sys.executable, str(ROOT / "deploy" / "build.py"), "--api-url", api_url], check=True)
        api.upload_folder(
            folder_path=str(ROOT / "deploy" / "out" / "space"),
            repo_id=repo_id, repo_type="space",
            commit_message="Deploy Speech Emotion Recognition",
        )
        print("pushed; Hugging Face is building the image")

    origins = ",".join(o for o in [args.cors, api_url] if o)
    api.add_space_variable(repo_id, "SER_API_CORS_ORIGINS", origins if args.cors else "*")
    print(f"cors    : {origins if args.cors else '*'}")


if __name__ == "__main__":
    main()
