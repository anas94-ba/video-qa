# اسأل فيديو · Ask Video

Ask questions about a YouTube video (or a podcast on YouTube). Every answer cites the exact time, and clicking a time jumps the player there. Runs on the **free Gemini API**. UI in Arabic and English.

اسأل عن فيديو يوتيوب أو بودكاست، وستحصل على إجابة مع الدقيقة والثانية، والضغط على الوقت ينقلك إليه في المشغّل.

![اAsk Video](images\image.png)
## Requirements / المتطلبات
- Python 3.10+  (بايثون 3.10 أو أعلى)
- مفتاح Gemini مجاني

## Run / تشغيل
**Windows (PowerShell):**
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

**Linux / macOS:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 · Tests: `pip install -r requirements-dev.txt && pytest`

## How it works
1. Captions with timestamps via `youtube-transcript-api`. No captions? Audio is downloaded (`yt-dlp` + `ffmpeg`) and transcribed by Gemini, then the uploaded file is deleted.
2. Text is grouped into ~45 s chunks, each keeping its start time.
3. Chunks are embedded with Gemini and cached in `cache/` (numpy cosine search, no vector DB needed).
4. For a question: top 6 chunks → Gemini answers in the question's language and cites `[MM:SS]`.

## API
`POST /api/index` `{"url": "..."}` → `{"video_id", "chunks"}`

`POST /api/ask` `{"video_id", "question", "history": [{"role","text"}]}` →
```json
{
  "answer": "He explains X [03:25] and then Y [07:10].",
  "sources":   [{"start": 205, "label": "03:25", "text": "..."}],
  "retrieved": ["03:25", "05:40", "07:10"]
}
```
`sources` = chunks the answer actually cites (validated against real chunks). `retrieved` = everything the model was shown.

## Errors
Every failure returns `{"error": {"code", "message"}}` with a proper HTTP status, and the UI shows a localized message.

| Code | Meaning |
|---|---|
| `INVALID_URL`, `INVALID_REQUEST` | bad link / empty or too-long question |
| `MISSING_API_KEY`, `INVALID_API_KEY`, `MODEL_NOT_FOUND` | key or model misconfigured |
| `RATE_LIMITED`, `GEMINI_UNAVAILABLE`, `NETWORK` | free-tier limit, busy or offline (retried automatically) |
| `VIDEO_UNAVAILABLE`, `AGE_RESTRICTED`, `YOUTUBE_BLOCKED` | video can't be read |
| `EMPTY_TRANSCRIPT`, `YTDLP_MISSING`, `FFMPEG_MISSING`, `DOWNLOAD_FAILED`, `DOWNLOAD_TIMEOUT`, `TRANSCRIPTION_FAILED` | no captions / fallback problems |
| `EMPTY_RESPONSE`, `EMBEDDING_FAILED`, `NOT_INDEXED`, `INTERNAL` | model or server problems |

## Notes
- Free-tier Gemini data may be used by Google to improve its products; don't send sensitive content.
- YouTube links only for now.
- Corrupt cache files are detected and rebuilt automatically.

## License
MIT
