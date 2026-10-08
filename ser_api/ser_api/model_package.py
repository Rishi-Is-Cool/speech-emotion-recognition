"""ModelPackage: loads the encoder + classifier ONCE at process startup, never per-request
(per SER_Audit_and_Production_Architecture.md section 3/4). Validates that the classifier
bundle's emotion order and config match what config.yaml expects -- a silent mismatch here
would misassign every probability to the wrong label, so this fails loudly at startup instead."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import joblib

from ser_api.config import Config

logger = logging.getLogger("ser_api.model_package")


def _materialize_local_encoder(src: Path) -> Path:
    """A local copy of the encoder weights. Hosts that cap single files at 100 MB (Vercel) get the
    weights as model.pt.part0..N plus a sha256; join them once into a writable temp dir and verify
    the checksum before anything loads them."""
    if (src / "model.pt").is_file():
        return src
    parts = sorted(src.glob("model.pt.part*"), key=lambda p: int(p.name.rsplit("part", 1)[1]))
    if not parts:
        raise FileNotFoundError(f"No encoder weights (model.pt or model.pt.part*) in {src}")
    dst = Path(tempfile.gettempdir()) / f"ser_encoder_{src.name}"
    target = dst / "model.pt"
    if not target.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and not f.name.startswith("model.pt"):
                shutil.copy2(f, dst / f.name)
        h = hashlib.sha256()
        tmp = target.with_suffix(".tmp")
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as fh:
                    while chunk := fh.read(8 << 20):
                        h.update(chunk)
                        out.write(chunk)
        expected = (src / "model.pt.sha256").read_text().strip()
        if h.hexdigest() != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Encoder weights failed their checksum after reassembly -- refusing to load them.")
        os.replace(tmp, target)
        logger.info("Reassembled encoder weights from %d parts into %s", len(parts), dst)
    return dst


@dataclasses.dataclass
class ModelPackage:
    encoder: object  # funasr.AutoModel instance
    encoder_generate_kwargs: dict
    scaler: object
    clf: object
    classes: list[str]
    sample_rate: int
    window_sec: int | float
    version_tag: str
    config: Config

    @classmethod
    def load(cls, config: Config) -> "ModelPackage":
        import funasr

        if not config.classifier_artifact_path.exists():
            raise FileNotFoundError(
                f"Classifier artifact not found at {config.classifier_artifact_path}. Download "
                "tier1b_best_head.joblib from Drive (state/notebook19_tier1b_iemocap/) and place "
                "it there, or point classifier.artifact_path in config.yaml at it."
            )

        bundle = joblib.load(config.classifier_artifact_path)
        scaler, clf = bundle["scaler"], bundle["clf"]
        classes = list(clf.classes_)

        if config.classifier_expected_config and bundle.get("config") != config.classifier_expected_config:
            raise RuntimeError(
                f"Classifier bundle config {bundle.get('config')!r} does not match "
                f"config.yaml's expected {config.classifier_expected_config!r}. Loading the "
                "wrong classifier would silently mislabel every prediction -- refusing to start."
            )

        encoder_id = config.encoder_model_id
        if config.encoder_local_dir:
            encoder_id = str(_materialize_local_encoder(config.encoder_local_dir))
        # ModelScope can fail to probe the network even when its complete snapshot is
        # already cached. Prefer that snapshot in offline/local environments.
        cached_encoder = (
            Path.home()
            / ".cache"
            / "modelscope"
            / "models"
            / encoder_id.replace("/", "--")
            / "snapshots"
            / "master"
        )
        if not config.encoder_local_dir and (cached_encoder / "model.pt").is_file() and (cached_encoder / "configuration.json").is_file():
            encoder_id = str(cached_encoder)

        # funasr resets torch to `ncpu` threads (default 4) at load and on every generate(). On a
        # 1-vCPU host that oversubscribes the core and made a 30 s clip take 87 s instead of 15 s,
        # so a small deployment sets encoder.ncpu to its real core count. Unset = funasr's default.
        extra = {"ncpu": config.encoder_ncpu} if config.encoder_ncpu else {}
        logger.info("Loading encoder %s (hub=%s, %s)...", encoder_id, config.encoder_hub, extra or "default threads")
        encoder = funasr.AutoModel(model=encoder_id, hub=config.encoder_hub, disable_update=True, **extra)

        artifact_hash = hashlib.sha256(config.classifier_artifact_path.read_bytes()).hexdigest()[:12]
        version_tag = f"{config.encoder_model_id.split('/')[-1]}+{artifact_hash}"

        logger.info("ModelPackage ready: version=%s classes=%s", version_tag, classes)

        return cls(
            encoder=encoder,
            encoder_generate_kwargs=config.encoder_generate_kwargs,
            scaler=scaler,
            clf=clf,
            classes=classes,
            sample_rate=config.sample_rate,
            window_sec=config.window_sec,
            version_tag=version_tag,
            config=config,
        )
