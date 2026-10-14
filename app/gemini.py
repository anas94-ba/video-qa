"""Gemini client + translation of SDK/network failures into AppError, with retries."""
import logging
import time

import httpx
from google import genai
from google.genai import errors as genai_errors

from . import config
from .errors import AppError

log = logging.getLogger("askvideo")
_client = None
RETRYABLE = {"RATE_LIMITED", "GEMINI_UNAVAILABLE"}


def client() -> genai.Client:
    global _client
    if _client is None:
        key = config.api_key()
        if not key:
            raise AppError("MISSING_API_KEY", "GEMINI_API_KEY is missing. Add it to your .env file.", 500)
        _client = genai.Client(api_key=key)
    return _client


def from_exception(exc: Exception) -> AppError:
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if code == 429 or "resource_exhausted" in text:
        return AppError("RATE_LIMITED", "Gemini free-tier limit reached. Wait a minute and try again.", 429)
    if code in (401, 403) or "api key" in text or "api_key_invalid" in text:
        return AppError("INVALID_API_KEY", "Gemini rejected the API key. Check GEMINI_API_KEY in .env.", 401)
    if code == 404:
        return AppError("MODEL_NOT_FOUND", "Gemini model not found. Update GEMINI_CHAT_MODEL / GEMINI_EMBED_MODEL in .env.", 502)
    if isinstance(code, int) and code >= 500:
        return AppError("GEMINI_UNAVAILABLE", "Gemini is busy right now. Try again shortly.", 503)
    return AppError("GEMINI_ERROR", "Gemini request failed.", 502)


def call(fn, retries: int = 2):
    """Run fn(); map known failures to AppError and retry transient ones with backoff."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except AppError:
            raise
        except genai_errors.APIError as e:
            err = from_exception(e)
        except (httpx.TransportError, ConnectionError, TimeoutError):
            err = AppError("NETWORK", "Cannot reach Gemini. Check your internet connection.", 503)
        log.warning("Gemini call failed (%s), attempt %d", err.code, attempt + 1)
        if err.code in RETRYABLE and attempt < retries:
            time.sleep(2 * (attempt + 1))
            continue
        raise err
