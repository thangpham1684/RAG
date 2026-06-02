"""
Tests for API authentication, rate limiting, and evidence policy enforcement.

Uses monkeypatch for env vars (auth reads env lazily) and dependency_overrides
for clean FastAPI dependency mocking — no fragile importlib.reload() needed.
"""

import os
import pytest
from fastapi.testclient import TestClient
import api
from auth import verify_api_key


# ── Helpers ─────────────────────────────────────────────────────────────

_TEST_API_KEY = "test-key-123"
_TEST_HEADERS = {"X-API-Key": _TEST_API_KEY}


def _make_client(monkeypatch, api_keys: str = ""):
    """Create a TestClient with API_KEYS env var set.

    auth.py reads env lazily, so monkeypatch is sufficient.
    """
    monkeypatch.setenv("API_KEYS", api_keys)
    return TestClient(api.app)


# ── Evidence / Policy Tests ─────────────────────────────────────────────

def test_abstain_returns_safe_message(monkeypatch):
    """When retriever abstains, API returns a safe 'not found' message (200)."""
    client = _make_client(monkeypatch, api_keys="")  # auth off

    class FakeRetriever:
        def retrieve_and_rerank(self, *args, **kwargs):
            class Evidence:
                decision = "abstain"
            return [], [], Evidence()

    api.state.retriever = FakeRetriever()
    response = client.post("/api/v1/chat", json={"query": "x", "selected_files": None})
    assert response.status_code == 200
    assert "Không tìm thấy" in response.text


# ── Authentication Tests ────────────────────────────────────────────────

class TestAuthentication:
    """Tests for API key authentication."""

    def test_valid_key_allows_access(self, monkeypatch):
        """Request with valid API key -> 200."""
        client = _make_client(monkeypatch, api_keys=_TEST_API_KEY)
        response = client.get("/api/v1/files", headers=_TEST_HEADERS)
        # Should succeed with valid key (files list may be empty, that's fine)
        assert response.status_code == 200

    def test_missing_key_returns_401(self, monkeypatch):
        """Request without API key when auth is enabled -> 401."""
        client = _make_client(monkeypatch, api_keys=_TEST_API_KEY)
        response = client.post(
            "/api/v1/chat",
            json={"query": "x", "selected_files": None},
            # No X-API-Key header
        )
        assert response.status_code == 401
        data = response.json()
        assert "API key" in data.get("detail", "")

    def test_invalid_key_returns_401(self, monkeypatch):
        """Request with wrong API key when auth is enabled -> 401."""
        client = _make_client(monkeypatch, api_keys=_TEST_API_KEY)
        response = client.post(
            "/api/v1/chat",
            json={"query": "x", "selected_files": None},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "không hợp lệ" in data.get("detail", "")

    def test_auth_disabled_bypasses_check(self, monkeypatch):
        """When API_KEYS is empty/not set, requests succeed without a key."""
        client = _make_client(monkeypatch, api_keys="")  # auth off
        response = client.get("/api/v1/files")
        assert response.status_code == 200

    def test_health_endpoint_public(self, monkeypatch):
        """Health check endpoint does NOT require auth even when auth is enabled."""
        client = _make_client(monkeypatch, api_keys=_TEST_API_KEY)
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_multiple_api_keys(self, monkeypatch):
        """Multiple comma-separated API keys all work."""
        client = _make_client(monkeypatch, api_keys="key-a,key-b,key-c")
        for key in ("key-a", "key-b", "key-c"):
            resp = client.get("/api/v1/files", headers={"X-API-Key": key})
            assert resp.status_code == 200, f"Key '{key}' should be valid"


# ── Rate Limiting Tests ─────────────────────────────────────────────────

class TestRateLimiting:
    """Basic rate limiting checks."""

    def test_rate_limit_not_triggered_on_light_use(self, monkeypatch):
        """A single request should not hit rate limits."""
        client = _make_client(monkeypatch, api_keys=_TEST_API_KEY)
        for _ in range(3):
            resp = client.get("/api/v1/files", headers=_TEST_HEADERS)
            assert resp.status_code == 200

    def test_rate_limit_returns_429_when_exceeded(self):
        """When rate limit is 1/minute, the second request returns 429."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from auth import limiter, setup_rate_limiting

        # Create a separate test app — fresh middleware, fresh counters
        test_app = FastAPI()
        setup_rate_limiting(test_app)

        @test_app.get("/test-ratelimit")
        @limiter.limit("1/minute")
        async def test_endpoint(request: Request):
            return {"ok": True}

        client = TestClient(test_app)

        # First request: should succeed
        resp1 = client.get("/test-ratelimit")
        assert resp1.status_code == 200
        assert resp1.json() == {"ok": True}

        # Second request: must be rate limited
        resp2 = client.get("/test-ratelimit")
        assert resp2.status_code == 429
        data = resp2.json()
        assert "Quá nhiều" in data.get("detail", "")
