"""FastAPI app. Every failure is returned as {"error": {"code", "message"}}."""
import logging
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config, rag
from .errors import AppError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("askvideo")
app = FastAPI(title="Ask Video")


def error_response(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(AppError)
async def on_app_error(_: Request, exc: AppError):
    return error_response(exc.code, exc.message, exc.status)


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError):
    return error_response("INVALID_REQUEST", "Invalid input. Check the link and the question length.", 422)


@app.exception_handler(StarletteHTTPException)
async def on_http_error(_: Request, exc: StarletteHTTPException):
    return error_response(f"HTTP_{exc.status_code}", str(exc.detail), exc.status_code)


@app.exception_handler(Exception)
async def on_unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return error_response("INTERNAL", "Unexpected server error. See the server logs.", 500)


class IndexReq(BaseModel):
    url: str = Field(min_length=1, max_length=300)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class AskReq(BaseModel):
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


@app.post("/api/index")
def index_video(req: IndexReq):
    return rag.index_video(req.url)


@app.post("/api/ask")
def ask(req: AskReq):
    question = req.question.strip()
    if not question:
        raise AppError("INVALID_REQUEST", "The question is empty.", 422)
    return rag.answer_question(req.video_id, question, [h.model_dump() for h in req.history])


@app.get("/api/health")
def health():
    return {"ok": True, "gemini_key_configured": bool(config.api_key())}


@app.get("/")
def home():
    return FileResponse(config.STATIC_DIR / "index.html")
