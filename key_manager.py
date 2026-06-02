"""
Dynamic API Key Manager — JSON file-based key store.

Allows runtime creation and revocation of API keys without server restart.
Keys are persisted to a JSON file and merged with env var keys in auth.py.

File format (api_keys.json):
{
    "keys": {
        "<full-key>": {
            "name": "Dev Key",
            "prefix": "rag_abc123",
            "created_at": "2026-05-29T10:00:00",
            "revoked": false
        }
    }
}
"""

import os
import json
import secrets
import threading
from datetime import datetime, timezone
from logging_config import get_logger

logger = get_logger(__name__)

# Default file path — override via env
_DEFAULT_KEYS_FILE = "api_keys.json"
_keys_file: str = os.getenv("API_KEYS_FILE", _DEFAULT_KEYS_FILE)
_lock = threading.Lock()


# ── Internal helpers ───────────────────────────────────────────────────

def _load_keys() -> dict:
    """Load all keys from the JSON file. Returns dict of key -> metadata."""
    if not os.path.exists(_keys_file):
        return {}
    try:
        with open(_keys_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("keys", {})
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load %s: %s", _keys_file, exc)
        return {}


def _save_keys(keys: dict):
    """Persist all keys to the JSON file atomically (write+rename)."""
    tmp = _keys_file + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"keys": keys}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _keys_file)
    except OSError as exc:
        logger.error("Failed to save %s: %s", _keys_file, exc)
        raise


def _generate_key() -> str:
    """Generate a cryptographically secure API key with 'rag_' prefix."""
    return "rag_" + secrets.token_urlsafe(32)


def _prefix(key: str) -> str:
    """Return a short prefix for display (rag_ + first 12 chars of the random part)."""
    return key[:16] + "..."


# ── Public API ─────────────────────────────────────────────────────────

def get_active_keys() -> set[str]:
    """Return all non-revoked dynamic keys."""
    with _lock:
        keys_dict = _load_keys()
        return {k for k, v in keys_dict.items() if not v.get("revoked", False)}


def get_all_keys() -> list[dict]:
    """Return all keys with metadata (for admin listing)."""
    with _lock:
        keys_dict = _load_keys()
        result = []
        for key, meta in keys_dict.items():
            result.append({
                "key_prefix": meta.get("prefix", _prefix(key)),
                "name": meta.get("name", ""),
                "created_at": meta.get("created_at", ""),
                "revoked": meta.get("revoked", False),
                # Never expose full key hash in list
            })
        # Sort: active first (revoked=False), then newest first
        result.sort(key=lambda x: (x["revoked"], x.get("created_at", "") or ""), reverse=False)
        return result


def create_key(name: str = "") -> tuple[str, dict]:
    """Create a new API key with an optional human-readable name.

    Returns:
        Tuple of (full_key, metadata_dict).
        The full key is shown ONCE to the user and then only the prefix is stored.
    """
    raw_key = _generate_key()
    key_prefix = _prefix(raw_key)
    meta = {
        "name": name.strip() or f"Key {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
        "prefix": key_prefix,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revoked": False,
    }

    with _lock:
        keys_dict = _load_keys()
        keys_dict[raw_key] = meta
        _save_keys(keys_dict)

    logger.info("🔑 Created API key: %s", key_prefix)
    return raw_key, meta


def revoke_key(key_prefix: str) -> bool:
    """Revoke a key by its prefix (partial match on prefix field).

    Returns True if a key was revoked, False if not found.
    """
    with _lock:
        keys_dict = _load_keys()
        for key, meta in list(keys_dict.items()):
            if meta.get("prefix", "") == key_prefix and not meta.get("revoked", False):
                meta["revoked"] = True
                meta["revoked_at"] = datetime.now(timezone.utc).isoformat()
                _save_keys(keys_dict)
                logger.info("🔑 Revoked API key: %s", key_prefix)
                return True
        return False


def count_active_keys() -> int:
    """Return the number of active (non-revoked) dynamic keys."""
    with _lock:
        return sum(1 for v in _load_keys().values() if not v.get("revoked", False))
