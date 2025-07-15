from __future__ import annotations
"""
Qwen Enterprise pattern‑2 gateway — router.py  (patched & re‑indented)
---------------------------------------------------------------------
This version simply restores correct structure so the file runs:

1.  A dedicated /v1/chat/completions endpoint handles the incoming
    OpenAI‑compatible requests (optionally multipart with an image).
2.  `stream_openai()` is again a small helper that *only* streams a
    backend response.
"""

import base64
import json
import re
from typing import Any, Iterable, List

import httpx
from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ────────────────────────── CONFIG ────────────────────────────────────
TEXT_URL   = "http://34.145.74.173:8081/v1/chat/completions"   # Qwen‑3‑32B
VISION_URL = "http://34.30.10.52:8080/v1/chat/completions"   # Qwen‑VL
MODEL_ALIAS = "Neptel-R0VL"

BACKENDS = {
    "text":   "Qwen/Qwen3-32B-AWQ",
    "vision": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}

CAP_PROMPT = (
    "USER: <image>\n"
    "You are a meticulous visual analyst. Produce a concise but "
    "information‑dense **structured description** of the image, following the template "
    "below. Use factual language only—no speculation, no personal identification, "
    "and do **not** answer any user question.\n"
    "\n"
    "Scene: <overall setting, environment, time‑of‑day>\n"
    "Key Objects:\n"
    "  • <object>, salient attributes (colour, material, shape, approx. size/state)\n"
    "  • … (repeat for each significant object)\n"
    "Counts & Metrics: <how many of each key object, notable numbers, distances>\n"
    "Relationships & Actions: <spatial relations, interactions, motion directions>\n"
    "Text in Image: <verbatim OCR of *EVERY* readable word/line/number or “None”>\n"
    "QA Clues: <bullet list of details likely to answer who/what/where/"
    "when/how‑many/how questions—e.g., brand logos, safety hazards, emotions, "
    "tools, dates, gauges>\n"
    "Mood & Style: <lighting, atmosphere, artistic style, camera angle>\n"
    "—\n"
    "ASSISTANT:"
)

DEEP_HEADER = "x-deep-reasoning"
DATA_URL_RX = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", re.I)
TIMEOUT = httpx.Timeout(120)




# ──────────────────────── NON‑STANDARD FIELDS ────────────────────────
# Extra keys that OpenWebUI sometimes adds and that vLLM ≤ 0.8 rejects.
NON_OPENAI_KEYS = {
    "features",          # {"web_search": …}
    "metadata",          # request/response tracking data
    "options",           # misc. UI flags
    "background_tasks",  # async jobs started by WebUI
    # → add any new ones you spot in the logs
}

def purge_extra(d: dict) -> dict:
    """Return a shallow copy without the WebUI‑specific keys."""
    return {k: v for k, v in d.items() if k not in NON_OPENAI_KEYS}

    
# ────────────────────────── APP SETUP ─────────────────────────────────
app = FastAPI(title="Qwen Enterprise Gateway (pattern‑2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Expose exactly one virtual model so OpenWebUI is happy."""
    return {
        "object": "list",
        "data": [{"id": MODEL_ALIAS, "object": "model", "created": 0}],
    }

# ────────────────────────── HELPERS ───────────────────────────────────
def latest_user_has_image(msgs: list | None) -> bool:
    """Returns True iff the **latest** user message contains an image."""
    if not msgs:
        return False
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            return (
                isinstance(c, list)
                and any(
                    isinstance(p, dict) and p.get("type") == "image_url"  # noqa: SIM115
                    for p in c
                )
            )
    return False

def strip_images(msgs: list) -> list:
    """Return *copies* of the messages with any image_url parts removed."""
    cleaned: List[dict[str, Any]] = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            content = [
                p for p in content
                if not (isinstance(p, dict) and p.get("type") == "image_url")
            ]
        cleaned.append({**m, "content": content})
    return cleaned

async def sse_response(text: str) -> StreamingResponse:
    """Send one SSE delta chunk + [DONE] so OpenWebUI can parse it."""
    def gen() -> Iterable[bytes]:
        first = {"choices": [{"index": 0, "delta": {"content": text}}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

async def stream_openai(payload: dict, url: str):
    """Stream any OpenAI‑compatible backend and surface *readable* errors."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as response:
            # Bubble up backend errors with a clearer message
            if response.status_code != 200:
                err_bytes = await response.aread()
                raise HTTPException(
                    response.status_code,
                    err_bytes.decode("utf-8", "replace"),
                )
            # Happy path: yield raw SSE chunks
            async for chunk in response.aiter_raw():
                yield chunk

# ──────────────────────── MAIN ENDPOINT ───────────────────────────────
@app.post("/v1/chat/completions")
async def proxy_completions(
    request: Request,
    file: UploadFile | None = File(None),
    x_deep_reasoning: str | None = Header(None, convert_underscores=False),
):
    """
    OpenAI‑compatible chat/completions proxy that:
    * Captions the newest image (upload or embedded) if present.
    * Optionally returns only the caption (`x-deep-reasoning: false`).
    * Otherwise injects the caption as context and forwards to the text model.
    """
    # 1️⃣  Parse caller body ------------------------------------------------
    if file is None:
        body: dict[str, Any] = await request.json()
    else:
        form = await request.form()
        try:
            body = json.loads(form["payload"])
        except (KeyError, json.JSONDecodeError):
            raise HTTPException(400, "multipart must include JSON field 'payload'")

    if body.get("model") != MODEL_ALIAS:
        raise HTTPException(400, f"unknown model '{body.get('model')}'")

    # 2️⃣  Decide whether we need vision -----------------------------------
    need_image = bool(file) or latest_user_has_image(body.get("messages"))

    # ╭────────────────────────── TEXT‑ONLY PATH ─────────────────────────╮
    if not need_image:
        clean = strip_images(body.get("messages", []))

        body  = purge_extra({**body, "messages": clean,
                             "model": BACKENDS["text"]})
                             
        return StreamingResponse(
            stream_openai(body, TEXT_URL),
            media_type="text/event-stream",
        )
    # ╰────────────────────────────────────────────────────────────────────╯

    # 3️⃣  Load / extract image bytes --------------------------------------
    if file:  # multipart path
        img_bytes = await file.read()
        mime = file.content_type or "image/png"
    else:     # embedded data‑URL or http(s) URL
        img_bytes = mime = None
        for m in reversed(body.get("messages", [])):
            c = m.get("content", [])
            if not isinstance(c, list):
                continue
            for p in reversed(c):
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    m_ = DATA_URL_RX.match(url)
                    if m_:
                        mime = m_.group("mime")
                        img_bytes = base64.b64decode(m_.group("b64"))
                    else:
                        async with httpx.AsyncClient() as cli:
                            resp = await cli.get(url)
                            resp.raise_for_status()
                            img_bytes = resp.content
                            mime = resp.headers.get("content-type", "image/png")
                    break
            if img_bytes:
                break
        if img_bytes is None:
            raise HTTPException(400, "Could not retrieve image bytes")

    data_url = (
        f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"  # noqa: S306
    )

    # 4️⃣  Caption with Qwen‑VL --------------------------------------------
    vl_payload = {
        "model": BACKENDS["vision"],
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": CAP_PROMPT},
                ],
            }
        ],
    }

    caption_parts: list[str] = []
    async for chunk in stream_openai(vl_payload, VISION_URL):
        for line in chunk.decode().splitlines():
            if line.startswith("data: {"):
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                txt = delta.get("content", "")
                if txt.strip():
                    caption_parts.append(txt)
    caption = "".join(caption_parts).strip()

    # 5️⃣  Quick caption return? -------------------------------------------
    deep = (x_deep_reasoning is None) or (x_deep_reasoning.lower() == "true")
    if not deep:
        if body.get("stream", True):
            return await sse_response(caption)
        return {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": caption}}
            ]
        }

    # 6️⃣  Merge caption + prompt, forward to text model -------------------
    if "prompt" in body:  # completions API
        patched = {
            **body,
            "prompt": (
                f"### Image context\n{caption}\n\n"
                f"### User\n{body['prompt']}\n\n### Assistant:"
            ),
            "model": BACKENDS["text"],
        }
    else:  # chat‑completions API
        clean_msgs = strip_images(body.get("messages", []))
        patched = {
            **body,
            "messages": [
                *clean_msgs[:-1],
                {"role": "system", "content": f"Image context:\n{caption}"},
                clean_msgs[-1],
            ],
            "model": BACKENDS["text"],
        }

    # 7️⃣  Forward to text model -------------------------------------------
    patched = purge_extra(patched)          # ← NEW: drop WebUI extras
    return StreamingResponse(
        stream_openai(patched, TEXT_URL),
        media_type="text/event-stream",
    )
