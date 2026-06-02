from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


OLLAMA_URL = "http://localhost:11434"
BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "ollama-chat-updated.html"
CHAT_PROMPT = (
    "You are a patient that has gone to do an interview with a psychologist. "
    "The psychologist will ask you a series of questions and you will answer them in a natural way:\n"
    "###Input:\n"
    "{question}\n\n"
    "###Expected Response:\n"
)
app = FastAPI()

# Allow CORS for all origins (simple and permissive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _close_resources(upstream: httpx.Response, client: httpx.AsyncClient) -> None:
    await upstream.aclose()
    await client.aclose()


async def _forward(request: Request, path: str, body: bytes | None = None) -> Response:
    url = f"{OLLAMA_URL}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
    headers.pop("content-length", None)

    client = httpx.AsyncClient(timeout=None)
    upstream = await client.send(
        client.build_request(
            request.method,
            url,
            content=body if body is not None else await request.body(),
            headers=headers,
        ),
        stream=True,
    )

    async def stream():
        async for chunk in upstream.aiter_raw():
            yield chunk

    return StreamingResponse(
        stream(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
        background=BackgroundTask(_close_resources, upstream, client),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    payload = await request.json()
    messages = payload.get("messages", [])
    question = next(
        (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    payload["messages"] = [{"role": "user", "content": CHAT_PROMPT.format(question=question)}]
    return await _forward(
        request,
        "v1/chat/completions",
        json.dumps(payload).encode("utf-8"),
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HTML_FILE, media_type="text/html")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    return await _forward(request, path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
