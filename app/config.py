"""Settings read from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
STATIC_DIR = ROOT / "static"

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "45"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_HISTORY = 4
MAX_QUESTION_CHARS = 1000


def api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()
