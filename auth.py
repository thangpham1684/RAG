"""
API Key Authentication & Rate Limiting for RAG Enterprise Pro.

Usage:
    from auth import verify_api_key, limiter, setup_rate_limiting

    @app.get("/api/v1/protected")
    @limiter.limit("30/minute")
    async def protected_endpoint(request: Request, api_key: str | None = Depends(verify_api_key)):
        ...
"""

import os
import hmac
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS
from slowapi import Limiter as SlowAPILimiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded as SlowAPIRateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse
from logging_config import get_logger

logger = get_logger(__name__)

# ── API Key Configuration (lazy — reads env vars on each call) ────────


# APIKeyHeader scheme — reads header name from env at module load time.
# In practice, the server restarts to pick up env changes anyway.
_HEADER_SCHEME = APIKeyHeader(
    name=os.getenv("API_KEY_HEADER", "X-API-Key"),
    auto_error=False,
)
from key_manager import get_active_keys as _get_dynamic_keys


def _get_api_keys() -> set[str]:
    """Read API keys from environment AND dynamic key store.

    1. Env var API_KEYS — supports comma, semicolon, or space-separated.
    2. Dynamic keys from key_manager.py (JSON file-based, CRUD at runtime).

    Returns empty set if both sources are empty (auth disabled).
    """
    keys: set[str] = set()

    # Env var keys
    keys_str = os.getenv("API_KEYS", "").strip()
    if keys_str:
        for sep in (",", ";", " "):
            if sep in keys_str:
                keys.update(k.strip() for k in keys_str.split(sep) if k.strip())
                break
        else:
            keys.add(keys_str)

    # Dynamic keys from JSON file store
    keys.update(_get_dynamic_keys())

    return keys


def _auth_enabled() -> bool:
    """Check whether API key authentication is active."""
    return len(_get_api_keys()) > 0


def _api_key_header_name() -> str:
    return os.getenv("API_KEY_HEADER", "X-API-Key")


def verify_api_key(
    api_key: str | None = Security(_HEADER_SCHEME),
) -> str | None:
    """Validate API key from request header.

    Reads API_KEYS from environment lazily so tests can override via
    monkeypatch without needing a module reload.

    Returns the API key if valid, None if auth is disabled.
    Raises 401 if auth is enabled and key is missing/invalid.
    """
    valid_keys = _get_api_keys()
    if not valid_keys:
        return None

    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Thiếu API key. Vui lòng gửi header X-API-Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    for valid_key in valid_keys:
        if hmac.compare_digest(api_key, valid_key):
            return api_key

    logger.warning("⚠️ Invalid API key attempt: %s...", api_key[:8])
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="API key không hợp lệ.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


# ── Rate Limiting Configuration ────────────────────────────────────────

def _rate_limit_key_func(request: Request) -> str:
    """Generate a rate limit key based on API key or IP address.

    If API key auth is enabled and a valid key is present, use the key as identity.
    Otherwise, fall back to client IP.
    """
    header_name = _api_key_header_name()
    api_key = request.headers.get(header_name, "")
    if api_key:
        for valid_key in _get_api_keys():
            if hmac.compare_digest(api_key, valid_key):
                return f"apikey:{api_key[:8]}"
    return get_remote_address(request)


def _rate_limit_exceeded_handler(request: Request, exc: SlowAPIRateLimitExceeded):
    """Custom handler for rate limit exceeded errors with Vietnamese message.

    Must return a JSONResponse (not HTTPException) because Starlette's exception
    middleware expects a valid ASGI application, not a raw exception object.
    """
    logger.warning("⏰ Rate limit exceeded for %s", request.client.host if request.client else "unknown")
    return JSONResponse(
        status_code=HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Quá nhiều yêu cầu. Vui lòng đợi và thử lại sau."},
    )


limiter = SlowAPILimiter(key_func=_rate_limit_key_func)


def setup_rate_limiting(app):
    """Attach rate limiter to a FastAPI application.

    Call this during app initialization.
    """
    app.state.limiter = limiter
    app.add_exception_handler(SlowAPIRateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info(
        "🔄 Rate limiting enabled: %s",
        os.getenv("RATE_LIMIT_CHAT", "30/minute"),
    )
