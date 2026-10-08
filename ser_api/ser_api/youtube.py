"""fetch_audio(): download a YouTube video's audio track and return it as WAV bytes, using
yt-dlp + a local ffmpeg for the extraction step. Scoped to youtube.com/youtu.be on purpose --
this is a convenience for testing the SER pipeline against found speech, not a general media
downloader. You are responsible for respecting the source video's copyright and YouTube's Terms
of Service; this is meant for personal, local analysis, not redistribution.

Duration is checked from yt-dlp's metadata BEFORE downloading, so an over-limit video is
rejected without spending time/bandwidth pulling the full audio track first. Every failure mode
(bad URL, private/region-locked video, missing ffmpeg, generic download failure) is mapped to a
specific SERInputError -- never a bare yt-dlp exception reaching the caller.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import tempfile

from ser_api.config import Config
from ser_api.errors import SERInputError

logger = logging.getLogger("ser_api.youtube")

_YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)", re.IGNORECASE
)


def _check_url(url: str) -> None:
    if not url or not url.strip():
        raise SERInputError("empty_url", "No YouTube URL was provided.")
    if not _YOUTUBE_URL_RE.match(url.strip()):
        raise SERInputError(
            "invalid_youtube_url",
            "That doesn't look like a youtube.com or youtu.be link. This endpoint only "
            "accepts YouTube URLs.",
        )


def fetch_audio(url: str, config: Config) -> tuple[bytes, dict]:
    """Download `url`'s audio as WAV bytes. Returns (wav_bytes, info) where info has 'title'
    and 'duration_sec' from YouTube's own metadata. Raises SERInputError for anything the
    caller should be told about specifically."""
    _check_url(url)

    import yt_dlp  # imported lazily -- a missing/broken install then only breaks this endpoint

    tmpdir = tempfile.mkdtemp(prefix="ser_yt_")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "logger": logging.getLogger("ser_api.youtube.ytdlp"),
    }
    if config.youtube_ffmpeg_location:
        ydl_opts["ffmpeg_location"] = config.youtube_ffmpeg_location

    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Metadata first, no download -- so an over-limit video is rejected before we
            # spend time/bandwidth pulling the audio track.
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration")
            if duration is not None:
                if duration < config.min_duration_sec:
                    raise SERInputError(
                        "too_short",
                        f"That video is {duration:.0f}s, shorter than the "
                        f"{config.min_duration_sec:.0f}s minimum needed for a reliable "
                        "prediction.",
                    )
                if duration > config.max_duration_sec:
                    raise SERInputError(
                        "too_long",
                        f"That video is {duration:.0f}s, longer than the "
                        f"{config.max_duration_sec:.0f}s maximum this service accepts.",
                    )
            ydl.download([url])
    except SERInputError:
        raise
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if "ffmpeg" in msg or "ffprobe" in msg:
            raise SERInputError(
                "ffmpeg_missing",
                "This server needs ffmpeg installed (and on PATH) to extract audio from "
                "YouTube videos. Install it from https://ffmpeg.org/download.html, or set "
                "youtube.ffmpeg_location in config.yaml, then restart the service.",
            ) from e
        raise SERInputError(
            "youtube_unavailable",
            "Could not download that video -- it may be private, age-restricted, "
            "region-locked, or removed.",
        ) from e
    except Exception as e:
        raise SERInputError(
            "youtube_download_failed",
            f"Something went wrong extracting audio from that link: {e}",
        ) from e

    wav_files = glob.glob(os.path.join(tmpdir, "*.wav"))
    if not wav_files:
        raise SERInputError(
            "youtube_download_failed", "The download finished but no audio file was produced."
        )

    try:
        with open(wav_files[0], "rb") as f:
            wav_bytes = f.read()
    finally:
        for f in glob.glob(os.path.join(tmpdir, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    return wav_bytes, {"title": info.get("title"), "duration_sec": info.get("duration")}
