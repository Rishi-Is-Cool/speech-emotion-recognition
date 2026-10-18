"""Load and validate config.yaml. One place that reads it -- nothing else in this package
touches the file directly, so tests can inject a synthetic config without a real YAML file."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import yaml

def default_config_path() -> Path:
    """Resolved per call, not at import. Capturing SER_API_CONFIG at import time meant the env
    var only took effect if it was set before anything first imported this module -- which is
    invisible when it goes wrong and silently loads the wrong config."""
    return Path(os.environ.get("SER_API_CONFIG") or Path(__file__).parent / "config.yaml")


def _resolve(p, base_dir: Path) -> Path | None:
    if not p:
        return None
    p = Path(p)
    return p if p.is_absolute() else (base_dir / p).resolve()


@dataclasses.dataclass
class Config:
    encoder_model_id: str
    encoder_hub: str
    encoder_generate_kwargs: dict
    encoder_local_dir: Path | None
    encoder_ncpu: int | None
    classifier_artifact_path: Path
    classifier_expected_config: str | None
    sample_rate: int
    min_duration_sec: float
    max_duration_sec: float
    silence_rms_threshold: float
    window_sec: float
    max_upload_mb: int
    log_level: str
    youtube_ffmpeg_location: str | None
    youtube_enabled: bool
    frontend_path: Path | None
    adaptive_artifacts_dir: Path | None
    adaptive_threshold: float | None
    adaptive_spec: str
    raw: dict  # the full parsed YAML, for anything not promoted to a typed field

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path) if path is not None else default_config_path()
        if not path.exists():
            raise FileNotFoundError(
                f"SER API config not found at {path}. Set SER_API_CONFIG to point at your "
                "config.yaml, or copy config.yaml.example to config.yaml and fill in the "
                "classifier artifact path."
            )
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path | None = None) -> "Config":
        base_dir = base_dir or Path(".")
        enc = raw["encoder"]
        clf = raw["classifier"]
        audio = raw["audio"]
        expl = raw["explainability"]
        server = raw.get("server", {})
        youtube = raw.get("youtube", {})
        adaptive = raw.get("adaptive", {})

        adaptive_dir = adaptive.get("artifacts_dir")
        if adaptive_dir:
            adaptive_dir = Path(adaptive_dir)
            if not adaptive_dir.is_absolute():
                adaptive_dir = (base_dir / adaptive_dir).resolve()

        artifact_path = Path(clf["artifact_path"])
        if not artifact_path.is_absolute():
            artifact_path = (base_dir / artifact_path).resolve()

        return cls(
            encoder_model_id=enc["model_id"],
            encoder_hub=enc["hub"],
            encoder_generate_kwargs=dict(enc["generate_kwargs"]),
            encoder_local_dir=_resolve(enc.get("local_dir"), base_dir),
            encoder_ncpu=int(enc["ncpu"]) if enc.get("ncpu") else None,
            classifier_artifact_path=artifact_path,
            classifier_expected_config=clf.get("expected_config"),
            sample_rate=int(audio["sample_rate"]),
            min_duration_sec=float(audio["min_duration_sec"]),
            max_duration_sec=float(audio["max_duration_sec"]),
            silence_rms_threshold=float(audio["silence_rms_threshold"]),
            window_sec=float(expl["window_sec"]),
            max_upload_mb=int(server.get("max_upload_mb", 25)),
            log_level=str(server.get("log_level", "info")),
            youtube_ffmpeg_location=youtube.get("ffmpeg_location") or None,
            youtube_enabled=bool(youtube.get("enabled", True)),
            frontend_path=_resolve(server.get("frontend_path"), base_dir),
            adaptive_artifacts_dir=adaptive_dir,
            adaptive_threshold=(
                float(adaptive["threshold"]) if adaptive.get("threshold") is not None else None
            ),
            adaptive_spec=str(adaptive.get("spec") or "serving_spec.json"),
            raw=raw,
        )
