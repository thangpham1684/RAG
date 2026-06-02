# Junior Readiness Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Dockerized local deployment, minimal observability (request logs + metrics), clearer env configuration, and integration tests so the project is junior-ready.

**Architecture:** Introduce a lightweight env helper (`config.py`) for consistent parsing, a request middleware that adds request IDs, logs, and records metrics into an in-memory store with a `/metrics` endpoint. Skip heavy model/database startup when `APP_ENV=test` to keep integration tests fast. Provide Dockerfile + docker-compose for running the API with Qdrant.

**Tech Stack:** FastAPI, pytest, python-dotenv, uvicorn, Qdrant, Docker

---

## File Structure

- Create: `config.py` — env helper utilities (bool/int/csv parsing).
- Create: `metrics.py` — in-memory metrics store + Prometheus-style text render.
- Create: `tests/test_config.py` — unit tests for env helpers.
- Create: `tests/test_metrics.py` — unit tests for metrics store.
- Create: `tests/test_api_integration.py` — API smoke/integration tests.
- Create: `Dockerfile` — container image for API.
- Create: `.dockerignore` — exclude local artifacts from build context.
- Create: `docker-compose.yml` — API + Qdrant local dev stack.
- Modify: `.env.example` — document APP_ENV + observability flags.
- Modify: `api.py` — use env helpers, add request-id/logging/metrics middleware, add /metrics, skip heavy startup in tests.
- Modify: `auth.py` — use env helpers for API key header and rate-limit logs.

---

### Task 1: Add env helper utilities + tests

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from config import get_env, get_bool, get_int, get_csv


def test_get_bool_parsing(monkeypatch):
    monkeypatch.setenv("BOOL_X", "true")
    assert get_bool("BOOL_X") is True
    monkeypatch.setenv("BOOL_X", "0")
    assert get_bool("BOOL_X") is False
    monkeypatch.delenv("BOOL_X", raising=False)
    assert get_bool("BOOL_X", default=True) is True


def test_get_int_parsing(monkeypatch):
    monkeypatch.setenv("INT_X", "42")
    assert get_int("INT_X", 7) == 42
    monkeypatch.setenv("INT_X", "not-a-number")
    assert get_int("INT_X", 7) == 7


def test_get_csv_parsing(monkeypatch):
    monkeypatch.setenv("CSV_X", "a, b, ,c")
    assert get_csv("CSV_X") == ["a", "b", "c"]
    monkeypatch.delenv("CSV_X", raising=False)
    assert get_csv("CSV_X", "x,y") == ["x", "y"]


def test_get_env_default(monkeypatch):
    monkeypatch.delenv("ENV_X", raising=False)
    assert get_env("ENV_X", "fallback") == "fallback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write minimal implementation**

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_csv(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [value.strip() for value in raw.split(",") if value.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add env helper utilities"
```

---

### Task 2: Wire env helpers into API/auth + update .env.example

**Files:**
- Modify: `api.py`
- Modify: `auth.py`
- Modify: `.env.example`

- [ ] **Step 1: Update `.env.example` with new flags**

```env
# ---------- App ----------
APP_ENV=dev

# ---------- Observability ----------
LOG_REQUESTS=false
REQUEST_ID_HEADER=X-Request-Id
METRICS_ENABLED=false
```

- [ ] **Step 2: Update `api.py` to use env helpers**

```python
# api.py (imports)
from config import get_env, get_int, get_bool, get_csv

# api.py (env access)
_INGESTION_MAX_RETRIES = get_int("INGEST_MAX_RETRIES", 2)

cors_origins = get_csv("CORS_ORIGINS")
if not cors_origins:
    cors_origins = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
    ]

# rate limiting
rate_limit_chat = get_env("RATE_LIMIT_CHAT", "20/minute")
rate_limit_upload = get_env("RATE_LIMIT_UPLOAD", "10/minute")
rate_limit_heavy = get_env("RATE_LIMIT_HEAVY", "10/minute")
rate_limit_medium = get_env("RATE_LIMIT_MEDIUM", "20/minute")
rate_limit_light = get_env("RATE_LIMIT_LIGHT", "60/minute")

# upload size + data dir
max_size_bytes = get_int("MAX_UPLOAD_SIZE_MB", 100) * 1024 * 1024
data_dir = get_env("DATA_DIR", "data")
```

- [ ] **Step 3: Update `auth.py` to use env helpers**

```python
# auth.py (imports)
from config import get_env

# auth.py (API key header)
_HEADER_SCHEME = APIKeyHeader(
    name=get_env("API_KEY_HEADER", "X-API-Key"),
    auto_error=False,
)


def _api_key_header_name() -> str:
    return get_env("API_KEY_HEADER", "X-API-Key")


def log_rate_limit_config():
    logger.info("🔄 Rate limiting enabled: %s", get_env("RATE_LIMIT_CHAT", "30/minute"))
```

- [ ] **Step 4: Run a targeted auth test**

Run: `pytest tests/test_api_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api.py auth.py .env.example
git commit -m "chore: centralize env access for api and auth"
```

---

### Task 3: Add metrics store + tests

**Files:**
- Create: `metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
from metrics import MetricsStore


def test_metrics_render_contains_counters():
    store = MetricsStore()
    store.inc_inflight()
    store.record_request("GET", "/health", 200, 12.5)
    store.dec_inflight()
    text = store.render_prometheus()
    assert 'requests_total{method="GET",path="/health",status="200"} 1' in text
    assert "requests_inflight" in text
    assert "request_duration_ms_sum" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# metrics.py
from collections import defaultdict
import threading


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight = 0
        self._requests_total = defaultdict(int)
        self._duration_sum = defaultdict(float)
        self._duration_count = defaultdict(int)

    def inc_inflight(self) -> None:
        with self._lock:
            self._inflight += 1

    def dec_inflight(self) -> None:
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1

    def record_request(self, method: str, path: str, status: int, duration_ms: float) -> None:
        key = (method, path, str(status))
        path_key = (method, path)
        with self._lock:
            self._requests_total[key] += 1
            self._duration_sum[path_key] += duration_ms
            self._duration_count[path_key] += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP requests_total Total HTTP requests.",
            "# TYPE requests_total counter",
        ]
        for (method, path, status), value in sorted(self._requests_total.items()):
            lines.append(
                f'requests_total{{method="{method}",path="{path}",status="{status}"}} {value}'
            )

        lines += [
            "# HELP requests_inflight Inflight HTTP requests.",
            "# TYPE requests_inflight gauge",
            f"requests_inflight {self._inflight}",
            "# HELP request_duration_ms_sum Sum of request durations in ms.",
            "# TYPE request_duration_ms_sum counter",
        ]
        for (method, path), value in sorted(self._duration_sum.items()):
            lines.append(
                f'request_duration_ms_sum{{method="{method}",path="{path}"}} {value:.2f}'
            )

        lines += [
            "# HELP request_duration_ms_count Count of request durations.",
            "# TYPE request_duration_ms_count counter",
        ]
        for (method, path), value in sorted(self._duration_count.items()):
            lines.append(
                f'request_duration_ms_count{{method="{method}",path="{path}"}} {value}'
            )

        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_metrics.py
git commit -m "feat: add in-memory metrics store"
```

---

### Task 4: Add request-id/logging/metrics middleware + integration tests

**Files:**
- Modify: `api.py`
- Create: `tests/test_api_integration.py`

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/test_api_integration.py
import os

os.environ["APP_ENV"] = "test"
os.environ["METRICS_ENABLED"] = "true"
os.environ["LOG_REQUESTS"] = "false"
os.environ["REQUEST_ID_HEADER"] = "X-Request-Id"

from fastapi.testclient import TestClient
import api


def test_health_includes_request_id_header():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-Id" in response.headers


def test_metrics_endpoint_exposes_counters():
    client = TestClient(api.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "requests_total" in response.text


def test_files_and_upload_work_in_test_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "example.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    async def fake_start_ingestion_job(data_dir: str) -> str:
        return "job-test-123"

    monkeypatch.setattr(api, "_start_ingestion_job", fake_start_ingestion_job)

    client = TestClient(api.app)
    files_response = client.get("/api/v1/files")
    assert files_response.status_code == 200
    assert any(item["name"] == "example.csv" for item in files_response.json()["files"])

    upload_response = client.post(
        "/api/v1/upload",
        files={"file": ("new.csv", b"a,b\n3,4\n", "text/csv")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["ingestion_job_id"] == "job-test-123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api_integration.py -v`  
Expected: FAIL with `404` for `/metrics` and missing `X-Request-Id` header

- [ ] **Step 3: Implement middleware, /metrics, and test-mode startup skip**

```python
# api.py (imports)
import time
import uuid
from fastapi.responses import PlainTextResponse
from config import get_env, get_int, get_bool, get_csv
from metrics import MetricsStore

metrics_store = MetricsStore()
REQUEST_ID_HEADER = get_env("REQUEST_ID_HEADER", "X-Request-Id")
LOG_REQUESTS = get_bool("LOG_REQUESTS", False)
METRICS_ENABLED = get_bool("METRICS_ENABLED", False)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    start = time.perf_counter()
    if METRICS_ENABLED:
        metrics_store.inc_inflight()

    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code if response else 500
        if METRICS_ENABLED:
            route = request.scope.get("route")
            path = route.path if route else request.url.path
            metrics_store.record_request(request.method, path, status_code, duration_ms)
            metrics_store.dec_inflight()
        if LOG_REQUESTS:
            logger.info(
                "➡️ %s %s %s %.2fms rid=%s",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                request_id,
            )
        if response:
            response.headers[REQUEST_ID_HEADER] = request_id


@app.get("/metrics")
async def metrics_endpoint(api_key: str | None = Depends(verify_api_key)):
    if not METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return PlainTextResponse(metrics_store.render_prometheus())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_env = get_env("APP_ENV", "dev").lower()
    if app_env == "test":
        logger.info("🧪 Test mode: skip heavy startup.")
        yield
        return
    # existing startup logic continues here...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api_integration.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api_integration.py
git commit -m "feat: add request logging, metrics, and integration tests"
```

---

### Task 5: Add Dockerization for local dev

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `.dockerignore`**

```dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
venv/
.git/
.idea/
.vscode/
qdrant_data/
docstore/
data/
graph_data/
dist/
build/
*.log
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:v1.11.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  api:
    build: .
    env_file:
      - .env
    environment:
      QDRANT_URL: http://qdrant:6333
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./docstore:/app/docstore
    depends_on:
      - qdrant

volumes:
  qdrant_data:
```

- [ ] **Step 4: Build the image**

Run: `docker build -t rag-api .`  
Expected: `Successfully tagged rag-api:latest`

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml
git commit -m "chore: add dockerized local dev stack"
```
