"""Import audio from a YouTube or Spotify link.

YouTube: downloaded directly via yt-dlp.

Spotify: Spotify's API only serves DRM-protected streams — there is no
legitimate way to download the actual Spotify audio, and this deliberately
does not attempt to. Instead it reads the public Open Graph metadata Spotify
serves for link previews (no login, no API key) to get the track's artist
and title, then finds and downloads the matching audio from YouTube. This is
the same approach tools like spotDL use.
"""

import re
import tempfile
from dataclasses import dataclass

import imageio_ffmpeg
import numpy as np
import requests
import yt_dlp

import audio_io

_OG_TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]*)"')
_OG_DESCRIPTION_RE = re.compile(r'<meta property="og:description" content="([^"]*)"')


@dataclass
class ImportResult:
    audio: np.ndarray
    sample_rate: int
    label: str  # description of what was actually downloaded, for display


def _spotify_search_query(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    response.raise_for_status()
    html = response.text

    title_match = _OG_TITLE_RE.search(html)
    if not title_match:
        raise ValueError("Could not read track info from that Spotify link.")
    title = title_match.group(1)

    artist = ""
    description_match = _OG_DESCRIPTION_RE.search(html)
    if description_match:
        artist = description_match.group(1).split(" · ")[0]

    return f"{artist} {title}".strip()


def import_from_url(url: str, target_sample_rate: int = 44100) -> ImportResult:
    """Download audio from a YouTube link, or a Spotify track link (matched to
    YouTube audio by artist/title). Raises ValueError for unsupported links."""
    url = url.strip()

    if "open.spotify.com" in url:
        download_target = f"ytsearch1:{_spotify_search_query(url)}"
    elif "youtube.com" in url or "youtu.be" in url:
        download_target = url
    else:
        raise ValueError("Only youtube.com, youtu.be, and open.spotify.com links are supported.")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmpdir:
        options = {
            "format": "bestaudio/best",
            "outtmpl": f"{tmpdir}/audio.%(ext)s",
            "ffmpeg_location": ffmpeg_path,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(download_target, download=True)
            if "entries" in info:
                info = info["entries"][0]
            label = info.get("title", "Downloaded audio")

        audio, sample_rate = audio_io.load_audio_file(f"{tmpdir}/audio.wav", target_sample_rate)

    return ImportResult(audio=audio, sample_rate=sample_rate, label=label)
