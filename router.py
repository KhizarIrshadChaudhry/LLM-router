from __future__ import annotations

"""
Qwen Enterprise pattern‑2 gateway — router.py  (patched)
-------------------------------------------------------
This version incorporates the “small, surgical patch” to ensure that *every* new
image in an ongoing chat gets captioned.
Key changes (≈ 20 LOC):
1.  **Never mutate the persisted history** – we now create a `clean_msgs` copy
    for the text‑model hop.
2.  **only caption if new image is added** – reverse‑iterate through messages and
    their `content` blocks.
3.  Merge caption and user prompt via a *patched* request object so the original
    `body` stays untouched.
"""

import base64
import json
import re
from typing import Any, Iterable

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ────────────────────────── CONFIG ────────────────────────────────────
TEXT_URL = "http://34.169.255.176:8081/v1/chat/completions"  # Qwen‑3‑32B
VISION_URL = "http://34.61.124.48:8080/v1/chat/completions"  # Qwen‑VL
MODEL_ALIAS = "qwen-enterprise"

BACKENDS = {
    "text": "Qwen/Qwen3-32B-AWQ",
    "vision": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}

CAP_PROMPT = (
    "USER: <image>\n"
    "You are a meticulous visual analyst. Produce a concise (≤ 130 words) but "
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
    "Text in Image: <verbatim OCR of *every* readable word/line or “None”>\n"
    "QA Clues: <bullet list of details likely to answer who/what/where/"
    "when/how‑many/how questions—e.g., brand logos, safety hazards, emotions, "
    "tools, dates, gauges>\n"
    "Mood & Style: <lighting, atmosphere, artistic style, camera angle>\n"
    "—\n"
    "ASSISTANT:"
)


DEEP_HEADER = "x-deep-reasoning"
TIMEOUT = httpx.Timeout(120)

# ────────────────────────── APP SETUP ─────────────────────────────────
app = FastAPI(title="Qwen Enterprise Gateway (pattern‑2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
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
DATA_URL_RX = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", re.I)


def has_image(msgs: list | None) -> bool:
    for m in msgs or []:
        c = m.get("content")
        if isinstance(c, list):
            if any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in c
            ):
                return True
    return False


def strip_images(msgs: list) -> list:
    """Return *copies* of the messages with any image_url parts removed."""

    cleaned: list[dict[str, Any]] = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            content = [
                p
                for p in content
                if not (
                    isinstance(p, dict) and p.get("type") == "image_url"
                )
            ]
        cleaned.append({**m, "content": content})
    return cleaned

def latest_user_has_image(msgs: list | None) -> bool:
    """
    Inspect the most‑recent *user* message only and report whether it carries an
    image_url block.
    """
    if not msgs:
        return False
    # find the newest user authored message
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            return (
                isinstance(c, list)
                and any(isinstance(p, dict) and p.get("type") == "image_url" for p in c)
            )
    return False

async def sse_response(text: str) -> StreamingResponse:
    """Send one SSE delta chunk + [DONE] so OpenWebUI can parse it."""

    def gen() -> Iterable[bytes]:
        first = {"choices": [{"index": 0, "delta": {"content": text}}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def stream_openai(payload: dict, url: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                raise HTTPException(r.status_code, await r.text())
            async for chunk in r.aiter_raw():
                yield chunk


# ────────────────────────── MAIN ROUTE ────────────────────────────────
@app.post("/v1/chat/completions")
async def chat(
    request: Request,
    file: UploadFile | None = None,
    x_deep_reasoning: str | None = Header(None, alias=DEEP_HEADER),
):
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

    # 2️⃣  Decide whether an image exists ----------------------------------
    need_image = bool(file) or latest_user_has_image(body.get("messages"))

    if not need_image:
        # Remove any OLD image blocks so the text model won't choke
        clean = strip_images(body.get("messages", []))
        body = {**body, "messages": clean}
        body["model"] = BACKENDS["text"]
        return StreamingResponse(
            stream_openai(body, TEXT_URL),
            media_type="text/event-stream",
        )

    # 3️⃣  Load / extract image bytes --------------------------------------
    if file:  # multipart path
        img_bytes = await file.read()
        mime = file.content_type or "image/png"
    else:  # embedded image_url path — **reverse‑iterate to get newest**
        img_bytes = mime = None
        for m in reversed(body.get("messages", [])):
            if m.get("role") != "user":
                continue
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
                    else:  # http/https URL
                        async with httpx.AsyncClient() as cli:
                            resp = await cli.get(url)
                            resp.raise_for_status()
                            img_bytes = resp.content
                            mime = resp.headers.get("content-type", "image/png")
                    break  # inner loop
            if img_bytes:
                break  # outer loop
        if img_bytes is None:
            raise HTTPException(400, "Could not retrieve image bytes")

    data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

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
        # Honour caller’s streaming preference
        if body.get("stream", True):
            return await sse_response(caption)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": caption},
                }
            ]
        }

    # 6️⃣  Merge caption + prompt, forward to text model -------------------
    if "prompt" in body:  # completions API
        patched = {
            **body,
            "prompt": (
                f"### Image context\n{caption}\n\n" f"### User\n{body['prompt']}\n\n### Assistant:"
            ),
        }
        patched["model"] = BACKENDS["text"]
    else:  # chat‑completions API
        clean_msgs = strip_images(body.get("messages", []))
        patched = {
            **body,
            "messages": [
                *clean_msgs[:-1],
                {"role": "system", "content": f"Image context:\n{caption}"},
                clean_msgs[-1],
            ],
        }
        patched["model"] = BACKENDS["text"]

    # 7️⃣  Forward to text model ------------------------------------------
    return StreamingResponse(
        stream_openai(patched, TEXT_URL), media_type="text/event-stream"
    )