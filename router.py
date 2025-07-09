"""
router.py  –  FastAPI gateway that exposes **one** visible model ("qwen-enterprise")
and implements pattern‑2 routing:

  • no image  →  Qwen‑3‑32B   (single hop)
  • image + X-Deep-Reasoning:true
                →  Qwen‑VL caption
                →  merged prompt to Qwen‑3‑32B
                →  stream answer back

Run:
    uvicorn router:app --host 0.0.0.0 --port 8000 --workers 2
"""

import base64, json, httpx
from fastapi import FastAPI, UploadFile, Request, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------
# CONFIG – adjust to your vLLM pod addresses
# ------------------------------------------------------------------------
TEXT_URL   = "http://34.169.255.176:8081/v1/chat/completions"   # Qwen‑3‑32B
VISION_URL = "http://34.61.124.48:8080/v1/chat/completions"     # Qwen‑VL
MODEL_ALIAS = "qwen-enterprise"      # the ONLY model users will see
BACKENDS = {                        # internal IDs as known by vLLM
    "text":   "Qwen/Qwen3-32B-AWQ",
    "vision": "Qwen/2.5-VL-7B-Instruct-AWQ",
}
CAP_PROMPT = (
    "USER: <image>\n"
    "Describe the image in ≤100 words. Do NOT answer any question here.\n"
    "ASSISTANT:"
)
DEEP_HEADER = "x-deep-reasoning"
TIMEOUT = httpx.Timeout(120)

# ------------------------------------------------------------------------
# APP  &  CORS  &  /v1/models   (so OpenWebUI probe succeeds)
# ------------------------------------------------------------------------
app = FastAPI(title="Qwen Enterprise Gateway (pattern‑2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models")
def list_models():
    """Just one visible model for the user/UI."""
    return {"object": "list", "data": [
        {"id": MODEL_ALIAS, "object": "model", "created": 0, "owned_by": "gateway"}
    ]}

# ------------------------------------------------------------------------
# Streaming helper  (passes SSE chunks verbatim)
# ------------------------------------------------------------------------
async def openai_stream(payload: dict, url: str):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                detail = await r.aread()          # <-- read the body once
                raise HTTPException(r.status_code,
                                    detail.decode() if isinstance(detail, bytes) else str(detail))
            async for chunk in r.aiter_raw():
                yield chunk

# ------------------------------------------------------------------------
# Main endpoint  /v1/chat/completions
# ------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat(
    request: Request,
    file: UploadFile | None = None,         # will be populated if multipart
    x_deep_reasoning: str | None = Header(None, alias=DEEP_HEADER),
):
    # 0) Parse caller JSON
    if file is None:
        body = await request.json()
    else:
        form = await request.form()
        try:
            body = json.loads(form["payload"])
        except Exception:
            raise HTTPException(400, "multipart must contain JSON part 'payload'")

    if body.get("model") != MODEL_ALIAS:
        raise HTTPException(400, f"unknown model '{body.get('model')}'")

    # Fast path: no image -> single hop to Qwen‑3‑32B
    if file is None:
        body["model"] = BACKENDS["text"]
        return StreamingResponse(openai_stream(body, TEXT_URL),
                                 media_type="text/event-stream")

    # --------------------------------------------------------------------
    # Caption via Qwen‑VL
    # --------------------------------------------------------------------
    img_b64 = base64.b64encode(await file.read()).decode()
    data_url = f"data:{file.content_type or 'image/png'};base64,{img_b64}"

    vl_payload = {
    "model": BACKENDS["vision"],        # <= tell vLLM explicitly
    "stream": True,
    "messages": [
        {"role": "user",
         "content": [
             {"type": "image_url",
              "image_url": {"url": data_url}},
             {"type": "text", "text": CAP_PROMPT},
         ]},
    ],
    }

    caption_parts: list[str] = []
    async for chunk in openai_stream(vl_payload, VISION_URL):
        # chunk lines:  data: {...}\n
        for line in chunk.decode().splitlines():
            if line.startswith("data: {"):
                delta = json.loads(line[6:])["choices"][0]["delta"]
                caption_parts.append(delta.get("content", ""))
    caption = "".join(caption_parts).strip()

    # Quick caption return unless deep reasoning requested
    if not (x_deep_reasoning and x_deep_reasoning.lower() == "true"):
        return {
            "id": "caption-only",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": caption}}
            ],
        }

    # --------------------------------------------------------------------
    # Deep reasoning hop  (caption + original prompt -> text model)
    # --------------------------------------------------------------------
    if "prompt" in body:          # completions style
        merged = (
            f"### Image context\n{caption}\n\n"
            f"### User\n{body['prompt']}\n\n### Assistant:"
        )
        body["prompt"] = merged
    else:                         # chat style (preferred by WebUI)
        body.setdefault("messages", [])
        body["messages"] = [
            *body["messages"][:-1],
            {"role": "system", "content": f"Image context:\n{caption}"},
            body["messages"][-1],
        ]

    body["model"] = BACKENDS["text"]

    return StreamingResponse(openai_stream(body, TEXT_URL),
                             media_type="text/event-stream")
