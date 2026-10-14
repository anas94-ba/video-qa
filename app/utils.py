"""Pure helpers (no network, no heavy deps) so they are easy to unit-test."""
import re

from .errors import AppError

_URL_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")
_BARE_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_TS = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def video_id_from(url: str) -> str:
    url = (url or "").strip()
    m = _URL_ID.search(url)
    if m:
        return m.group(1)
    if _BARE_ID.fullmatch(url):
        return url
    raise AppError("INVALID_URL", "Not a valid YouTube link.", 400)


def fmt(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def to_seconds(ts: str) -> int:
    total = 0
    for part in ts.split(":"):
        total = total * 60 + int(part)
    return total


def _start(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    try:
        return float(value)
    except ValueError:
        return float(to_seconds(value))


def make_chunks(segments, chunk_seconds: int = 45) -> list[dict]:
    """Group caption segments into ~chunk_seconds windows, each keeping its start time."""
    clean = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            start = _start(seg.get("start"))
        except (ValueError, TypeError):
            continue
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if text and start >= 0:
            clean.append({"start": start, "text": text})
    clean.sort(key=lambda s: s["start"])

    chunks, cur = [], None
    for s in clean:
        if cur is None:
            cur = dict(s)
        elif s["start"] - cur["start"] >= chunk_seconds:
            chunks.append(cur)
            cur = dict(s)
        else:
            cur["text"] += " " + s["text"]
    if cur:
        chunks.append(cur)
    return chunks


def cited_chunks(answer: str, chunks: list[dict]) -> list[dict]:
    """Chunks whose start time the answer cites as [MM:SS]; unknown timestamps are ignored."""
    by_sec = {int(c["start"]): c for c in chunks}
    out, seen = [], set()
    for ts in _TS.findall(answer or ""):
        sec = to_seconds(ts)
        if sec in by_sec and sec not in seen:
            seen.add(sec)
            out.append(by_sec[sec])
    return sorted(out, key=lambda c: c["start"])
