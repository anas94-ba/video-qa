import pytest

from app.errors import AppError
from app.utils import cited_chunks, fmt, make_chunks, to_seconds, video_id_from


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=10",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "dQw4w9WgXcQ",
])
def test_video_id_variants(url):
    assert video_id_from(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize("bad", ["", "hello", "https://example.com/x", None])
def test_invalid_url_raises(bad):
    with pytest.raises(AppError) as e:
        video_id_from(bad)
    assert e.value.code == "INVALID_URL"


def test_fmt_and_to_seconds():
    assert fmt(205) == "03:25"
    assert fmt(3725) == "1:02:05"
    assert to_seconds("03:25") == 205
    assert to_seconds("1:02:05") == 3725


def test_make_chunks_groups_by_window_and_skips_garbage():
    segs = [{"start": 0, "text": "a"}, {"start": 20, "text": "b"}, {"start": 50, "text": "c"},
            {"start": "01:40", "text": "d"}, {"start": "x", "text": "bad"}, {"start": 5, "text": "  "}, "junk"]
    chunks = make_chunks(segs, 45)
    assert [c["start"] for c in chunks] == [0, 50, 100]
    assert chunks[0]["text"] == "a b"


def test_cited_chunks_matches_only_known_timestamps():
    chunks = [{"start": 205.4, "text": "x"}, {"start": 300, "text": "y"}]
    got = cited_chunks("Said so [3:25] and again [03:25] but not [09:99] or [01:00].", chunks)
    assert [c["text"] for c in got] == ["x"]
