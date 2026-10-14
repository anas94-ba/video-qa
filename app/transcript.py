"""Get a timestamped transcript: YouTube captions first, Gemini audio transcription as fallback."""
import glob
import json
import logging
import shutil
import subprocess
import tempfile
import time

from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

from . import config
from .errors import AppError
from .gemini import call, client

log = logging.getLogger("askvideo")

# youtube-transcript-api exception class names -> our error (matched by name to survive version changes)
CAPTION_ERRORS = {
    "VideoUnavailable": ("VIDEO_UNAVAILABLE", "This video is unavailable, private or removed.", 404),
    "VideoUnplayable": ("VIDEO_UNAVAILABLE", "This video can't be played (private, removed or restricted).", 404),
    "InvalidVideoId": ("INVALID_URL", "Not a valid YouTube video id.", 400),
    "AgeRestricted": ("AGE_RESTRICTED", "Age-restricted videos can't be read without login.", 422),
    "RequestBlocked": ("YOUTUBE_BLOCKED", "YouTube blocked requests from this network. Try later or another network.", 503),
    "IpBlocked": ("YOUTUBE_BLOCKED", "YouTube blocked requests from this network. Try later or another network.", 503),
}
NO_CAPTIONS = {"TranscriptsDisabled", "NoTranscriptFound"}


def fetch_captions(video_id: str):
    """Return [{start, text}] or None when the video simply has no usable captions."""
    try:
        listing = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = listing.find_transcript(["ar", "en"])
        except Exception:
            transcript = next(iter(listing))
        return [{"start": s.start, "text": s.text} for s in transcript.fetch()]
    except StopIteration:
        return None
    except Exception as e:
        name = type(e).__name__
        if name in NO_CAPTIONS:
            return None
        if name in CAPTION_ERRORS:
            code, msg, status = CAPTION_ERRORS[name]
            raise AppError(code, msg, status) from e
        log.warning("Caption fetch failed (%s: %s); trying audio fallback", name, e)
        return None


def transcribe_with_gemini(video_id: str):
    """Fallback for videos without captions: download audio and let Gemini transcribe it."""
    if not shutil.which("yt-dlp"):
        raise AppError("YTDLP_MISSING", "No captions found and yt-dlp is not installed (pip install yt-dlp).", 422)
    if not shutil.which("ffmpeg"):
        raise AppError("FFMPEG_MISSING", "No captions found and ffmpeg is not installed (needed to extract audio).", 422)

    tmp = tempfile.mkdtemp()
    uploaded = None
    try:
        try:
            subprocess.run(
                ["yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "mp3", "--audio-quality", "9",
                 "-o", f"{tmp}/a.%(ext)s", f"https://www.youtube.com/watch?v={video_id}"],
                check=True, capture_output=True, timeout=900,
            )
        except subprocess.TimeoutExpired as e:
            raise AppError("DOWNLOAD_TIMEOUT", "Downloading the audio took too long.", 504) from e
        except subprocess.CalledProcessError as e:
            log.warning("yt-dlp failed: %s", (e.stderr or b"").decode("utf-8", "ignore")[-300:])
            raise AppError("DOWNLOAD_FAILED", "Could not download the audio of this video.", 502) from e

        files = glob.glob(f"{tmp}/a.*")
        if not files:
            raise AppError("DOWNLOAD_FAILED", "Could not download the audio of this video.", 502)

        uploaded = call(lambda: client().files.upload(file=files[0]))
        deadline = time.time() + 300
        while uploaded.state and uploaded.state.name == "PROCESSING":
            if time.time() > deadline:
                raise AppError("UPLOAD_TIMEOUT", "Gemini took too long to process the audio.", 504)
            time.sleep(2)
            uploaded = call(lambda: client().files.get(name=uploaded.name))
        if uploaded.state and uploaded.state.name == "FAILED":
            raise AppError("UPLOAD_FAILED", "Gemini could not process the audio file.", 502)

        prompt = (
            "Transcribe this audio in its original language. Return a JSON array of objects "
            '{"start": <seconds from the beginning, number>, "text": <string>}, one per 20-30 seconds.'
        )
        r = call(lambda: client().models.generate_content(
            model=config.CHAT_MODEL, contents=[uploaded, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        ))
        try:
            data = json.loads(r.text or "")
        except ValueError as e:
            raise AppError("TRANSCRIPTION_FAILED", "Gemini returned an unreadable transcript.", 502) from e
        if not isinstance(data, list) or not data:
            raise AppError("TRANSCRIPTION_FAILED", "Gemini returned an empty transcript.", 502)
        return data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if uploaded is not None:
            try:
                client().files.delete(name=uploaded.name)
            except Exception:
                pass
