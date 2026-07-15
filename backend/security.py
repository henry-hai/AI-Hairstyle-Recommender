"""Endpoint-protection concerns for the API, kept separate from the analysis
flow in main.py.

Groups the guardrails that keep the public `/analyze` endpoint safe:
  * CORS locked to the configured origins (cookieless, so no credentials)
  * a small baseline of security response headers
  * per-IP rate limiting, because the endpoint calls a paid LLM
  * upload validation (content type and size) before the vision pipeline runs
"""

import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Upload guardrails: cap the request body and only accept real image types so a
# bad or oversized payload is rejected before it reaches the vision pipeline.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))  # 8 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Rate limit on the analyze endpoint. It calls a paid LLM, so an unbounded
# public endpoint is a standing bill risk; this caps requests per client IP.
ANALYZE_RATE_LIMIT = os.getenv("ANALYZE_RATE_LIMIT", "10/minute")

# Shared limiter instance. main.py applies it to the endpoint with a decorator,
# and configure_security wires it into the app.
limiter = Limiter(key_func=get_remote_address)


def configure_security(app: FastAPI, allowed_origins):
    """Attach rate limiting, CORS, and baseline security headers to the app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Origins are locked to the configured list; the API is cookieless, so
    # credentials stay off and only the methods the frontend uses are allowed.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        return response


async def decode_validated_image(request: Request, file: UploadFile):
    """Validate and decode an uploaded image, or raise an HTTPException.

    Client-side `accept="image/*"` is only UX; the server re-checks the type
    and size here and confirms the bytes actually decode to an image, so a
    wrong, empty, oversized, or corrupt upload fails cleanly instead of
    crashing the vision pipeline downstream.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPG, PNG, or WebP image.",
        )

    # Reject oversized bodies early when the client declares the length.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the image. Upload a valid JPG, PNG, or WebP.",
        )
    return image