from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import shutil
import uuid
import uvicorn
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from embeddings.vector_db import QdrantDBManager
from generators.llm_generator import ResponseGenerator
from retrievers.hybrid_search import AdvancedHybridRetriever
from ingest_worker import run_ingestion_job, SUPPORTED_EXTS
from dotenv import load_dotenv
from logging_config import get_logger

from auth import verify_api_key, setup_rate_limiting, limiter

load_dotenv()

# Ensure stdout/stderr use UTF-8 on Windows to avoid UnicodeEncodeError for emoji
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def _format_evidence_status(ev):
    """Map evidence.decision values to safe, human-readable Vietnamese labels.
    Handles missing or None evidence gracefully.
    """
    dec = (getattr(ev, "decision", "") or "").lower()
    mapping = {
        "ok": "Đủ bằng chứng",
        "conflict": "Có mâu thuẫn",
        "abstain": "Thiếu bằng chứng",
    }
    return mapping.get(dec, "Không rõ")

# --- STATE QUẢN LÝ TÀI NGUYÊN (Singleton) ---
class AppState:
    db_manager = None
    generator = None
    retriever = None

state = AppState()
state_lock = asyncio.Lock()
logger = get_logger(__name__)

# --- BACKGROUND TASK INFRASTRUCTURE ---
_ingestion_executor = ThreadPoolExecutor(max_workers=1)
_ingestion_jobs: dict[str, dict] = {}
_ingestion_jobs_lock = asyncio.Lock()
_INGESTION_MAX_RETRIES = int(os.getenv("INGEST_MAX_RETRIES", "2"))
_main_event_loop: asyncio.AbstractEventLoop | None = None


def _build_retriever_sync():
    """Build a new retriever from current index (called in thread pool)."""
    existing_index = state.db_manager.get_existing_index()
    if existing_index:
        nodes = list(existing_index.docstore.docs.values())
        return AdvancedHybridRetriever(existing_index, nodes, state.generator.llm, state.db_manager)
    return None


async def _swap_retriever(new_retriever):
    """Atomically swap retriever under the state lock."""
    async with state_lock:
        state.retriever = new_retriever


def _run_ingestion_in_thread(job_id: str, data_dir: str, max_retries: int, main_loop: asyncio.AbstractEventLoop):
    """Run ingestion in thread pool, then schedule retriever swap on main loop.

    All _ingestion_jobs mutations are done via run_coroutine_threadsafe
    to stay thread-safe with the async lock.
    """
    def _sched_update(updates: dict):
        """Schedule a job status update on the main event loop."""
        coro = _update_job_status(job_id, updates)
        fut = asyncio.run_coroutine_threadsafe(coro, main_loop)
        fut.result(timeout=10)

    try:
        _sched_update({"status": "running"})

        result = run_ingestion_job(data_dir=data_dir, max_retries=max_retries)

        # Build new retriever (sync, no lock needed — models are read-only)
        new_retriever = _build_retriever_sync()

        # Schedule retriever swap on the main event loop
        if new_retriever:
            future = asyncio.run_coroutine_threadsafe(_swap_retriever(new_retriever), main_loop)
            future.result(timeout=30)  # Wait for swap to complete

        _sched_update({
            "status": "success",
            "result": result,
            "total_nodes": result.get("total_nodes", 0),
            "processed_files": result.get("processed_files", 0),
        })
        logger.info(f"✅ Background ingestion {job_id} completed.")
    except Exception as exc:
        logger.error(f"❌ Background ingestion {job_id} failed: {exc}")
        _sched_update({
            "status": "error",
            "error": str(exc),
        })


async def _update_job_status(job_id: str, updates: dict):
    """Thread-safe job status update (called via run_coroutine_threadsafe)."""
    async with _ingestion_jobs_lock:
        if job_id in _ingestion_jobs:
            _ingestion_jobs[job_id].update(updates)


async def _start_ingestion_job(data_dir: str) -> str:
    """Start ingestion in background and return job_id.

    Raises HTTPException 409 if a job is already running.
    """
    async with _ingestion_jobs_lock:
        for jid, jinfo in list(_ingestion_jobs.items()):
            if jinfo.get("status") in ("running", "queued"):
                if jinfo.get("data_dir") == data_dir:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Đã có một job ingestion đang chạy (job_id={jid}). Vui lòng đợi hoặc kiểm tra status.",
                    )
                # Clean up stale jobs from other data dirs
                if jinfo.get("status") == "queued":
                    _ingestion_jobs.pop(jid, None)

        job_id = str(uuid.uuid4())
        _ingestion_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_dir": data_dir,
        }

        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            _ingestion_executor,
            _run_ingestion_in_thread,
            job_id,
            data_dir,
            _INGESTION_MAX_RETRIES,
            loop,
        )

        return job_id


# --- LIFESPAN CHUẨN CỦA FASTAPI ĐỂ LOAD MODEL LÚC KHỞI ĐỘNG ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_event_loop
    _main_event_loop = asyncio.get_event_loop()

    logger.info("🚀 Khởi động Backend API... Đang tải Model và Database.")
    state.db_manager = QdrantDBManager()
    state.generator = ResponseGenerator()
    
    # Load dữ liệu docstore local (nếu có sẵn)
    existing_index = state.db_manager.get_existing_index()
    if existing_index:
        logger.info("📦 Đã load thành công Index từ docstore.")
        # Load các node (docs) từ docstore để nạp cho thuật toán BM25 
        nodes = list(existing_index.docstore.docs.values())
        state.retriever = AdvancedHybridRetriever(existing_index, nodes, state.generator.llm, state.db_manager)
    else:
        logger.warning("⚠️ Chưa có dữ liệu docstore. Hệ thống sẽ ở trạng thái chờ.")
    
    yield # API bắt đầu chạy sau Yield
    
    logger.info("🛑 Đang tắt Backend API. Đóng kết nối DB...")
    _ingestion_executor.shutdown(wait=False)

# Khởi tạo App FastAPI
app = FastAPI(
    title="RAG Enterprise Core API", 
    version="1.0.0", 
    description="Backend xử lý truy vấn văn bản tốc độ cao",
    lifespan=lifespan
)

# Cấu hình CORS — chỉ cho phép các origin được khai báo trong biến môi trường CORS_ORIGINS
# Mặc định hỗ trợ các port dev phổ biến (Streamlit 8501, Live Server 5500, dev server 3000, Swagger UI)
_DEFAULT_ORIGINS = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)
cors_origins_str = os.getenv("CORS_ORIGINS", "")
cors_origins = [
    o.strip()
    for o in cors_origins_str.split(",")
    if o.strip()
] if cors_origins_str else list(_DEFAULT_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kích hoạt rate limiting
setup_rate_limiting(app)

# --- MODELS BẮT REQUEST TỪ CLIENT ---
class ChatRequest(BaseModel):
    query: str
    selected_files: Optional[List[str]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

# --- CÁC ENDPOINTS (APIs) ---

@app.get("/health")
async def health_check():
    async with state_lock:
        db_loaded = state.retriever is not None
    return {
        "status": "Khoẻ mạnh", 
        "db_loaded": db_loaded
    }

@app.get("/api/v1/ingestion/status")
@limiter.limit(os.getenv("RATE_LIMIT_LIGHT", "60/minute"))
async def ingestion_status(request: Request, api_key: str | None = Depends(verify_api_key)):
    """Return the latest background ingestion job status.

    This endpoint provides a quick overview of the most recent job.
    For a specific job, use GET /api/v1/ingestion/status/{job_id}.
    """
    async with _ingestion_jobs_lock:
        if not _ingestion_jobs:
            return {
                "status": "unknown",
                "message": "Chưa có job ingestion nào được chạy.",
            }
        # Return the most recent job (last inserted)
        latest_job_id = max(_ingestion_jobs.keys(), key=lambda jid: _ingestion_jobs[jid].get("created_at", ""))
        job = _ingestion_jobs[latest_job_id]
        return {
            "status": job["status"],
            "job_id": job["job_id"],
            "created_at": job.get("created_at"),
            "total_nodes": job.get("total_nodes"),
            "processed_files": job.get("processed_files"),
            "error": job.get("error"),
        }

@app.post("/api/v1/chat")
@limiter.limit(os.getenv("RATE_LIMIT_CHAT", "20/minute"))
async def chat_endpoint(request: Request, req: ChatRequest, api_key: str | None = Depends(verify_api_key)):
    """
    API nhận câu hỏi và stream lại câu trả lời theo thời gian thực (Server-Sent Events)
    """
    # Snapshot retriever/generator under lock để tránh TOCTOU race với build_index_endpoint
    async with state_lock:
        retriever = state.retriever
        generator = state.generator

    if not retriever:
        raise HTTPException(status_code=400, detail="Hệ thống chưa có dữ liệu. Hãy nạp tài liệu và call /api/v1/index trước.")
        
    try:
        # 1. Truy xuất & Tái xếp hạng (Retrieval & Rerank)
        #    Truyền conversation_history để contextualize follow-up queries trước khi search
        conversation_history = req.conversation_history or []
        best_nodes, raw_nodes, evidence = retriever.retrieve_and_rerank(
            req.query, req.selected_files, conversation_history
        )

        # Safely derive decision
        decision = (getattr(evidence, "decision", "") or "").lower()

        # If retriever explicitly abstains, return an immediate not-found message
        if decision == "abstain":
            def empty_response():
                yield "Không tìm thấy thông tin phù hợp trong tài liệu hiện có."
            return StreamingResponse(empty_response(), media_type="text/plain")

        if not best_nodes:
            # Nếu không tìm thấy, trả về ngay lập tức một chuỗi văn bản cứng
            def empty_response():
                yield "Không tìm thấy thông tin phù hợp trong tài liệu của hệ thống."
            return StreamingResponse(empty_response(), media_type="text/plain")

        # 2. Sinh câu trả lời stream qua Generator; pass mapped evidence status + conversation history
        mapped_status = _format_evidence_status(evidence)
        def response_generator():
            for chunk in generator.generate_answer_stream(req.query, best_nodes, mapped_status, conversation_history):
                yield chunk

        # Cho phép Client nhận luồng streaming Text
        return StreamingResponse(response_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"❌ API Chat lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/files")
@limiter.limit(os.getenv("RATE_LIMIT_MEDIUM", "60/minute"))
async def list_files(request: Request, api_key: str | None = Depends(verify_api_key)):
    """Return list of available documents in the data directory."""
    data_dir = os.getenv("DATA_DIR", "data")
    if not os.path.exists(data_dir):
        return {"files": []}
    try:
        files = []
        for f in os.listdir(data_dir):
            fpath = os.path.join(data_dir, f)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                })
        files.sort(key=lambda x: x["name"].lower())
        return {"files": files}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/upload")
@limiter.limit(os.getenv("RATE_LIMIT_UPLOAD", "10/minute"))
async def upload_file(request: Request, file: UploadFile = File(...), api_key: str | None = Depends(verify_api_key)):
    """
    Upload a document file to the data directory and trigger incremental ingestion.
    Supports: PDF, DOC, DOCX, XLSX, CSV, PPTX.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {ext}. Các định dạng hỗ trợ: {', '.join(sorted(SUPPORTED_EXTS))}",
        )

    # File size limit (default 100MB)
    max_size_bytes = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")) * 1024 * 1024
    file_size = file.size
    if file_size is None:
        # FastAPI không populate file.size cho tất cả clients, cần đọc thủ công
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
    if file_size > max_size_bytes:
        max_mb = max_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File quá lớn (tối đa {max_mb}MB). File này: {file_size // (1024 * 1024)}MB",
        )

    data_dir = os.getenv("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)

    dest_path = os.path.join(data_dir, file.filename)
    if os.path.exists(dest_path):
        raise HTTPException(
            status_code=409,
            detail=f"Tệp '{file.filename}' đã tồn tại. Vui lòng xoá file cũ trước khi upload lại.",
        )

    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file: {exc}")
    finally:
        file.file.close()

    logger.info(f"📤 File uploaded: {file.filename} ({os.path.getsize(dest_path)} bytes)")

    # Trigger ingestion as background task — return immediately with job_id
    try:
        job_id = await _start_ingestion_job(data_dir)
        return {
            "status": "uploaded",
            "file_name": file.filename,
            "size": os.path.getsize(dest_path),
            "ingestion_job_id": job_id,
            "message": "File đã được lưu. Ingestion đang chạy nền.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Failed to start background ingestion after upload: {exc}")
        return {
            "status": "uploaded",
            "file_name": file.filename,
            "size": os.path.getsize(dest_path),
            "warning": f"File đã được lưu nhưng không thể khởi động ingestion nền: {exc}",
        }


@app.delete("/api/v1/files/{file_name}")
@limiter.limit(os.getenv("RATE_LIMIT_MEDIUM", "30/minute"))
async def delete_file(request: Request, file_name: str, api_key: str | None = Depends(verify_api_key)):
    """
    Delete a document from the data directory and remove its nodes from the index.
    """
    data_dir = os.getenv("DATA_DIR", "data")
    file_path = os.path.join(data_dir, file_name)

    # Security: prevent path traversal
    abs_data = os.path.abspath(data_dir)
    abs_file = os.path.abspath(file_path)
    if not abs_file.startswith(abs_data + os.sep) and abs_file != abs_data:
        raise HTTPException(status_code=400, detail="Tên tệp không hợp lệ.")

    if not os.path.exists(abs_file):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy tệp '{file_name}'.")

    if not os.path.isfile(abs_file):
        raise HTTPException(status_code=400, detail=f"'{file_name}' không phải là tệp.")

    # 1. Remove from vector store & docstore
    try:
        deleted_count = state.db_manager.delete_file_nodes(file_name)
        logger.info(f"🗑️ Deleted {deleted_count} nodes for file: {file_name}")
    except Exception as exc:
        logger.error(f"❌ Failed to delete nodes for {file_name}: {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xoá dữ liệu index: {exc}")

    # 2. Remove physical file
    try:
        os.remove(abs_file)
        logger.info(f"🗑️ Deleted file: {file_name}")
    except Exception as exc:
        logger.error(f"❌ Failed to delete physical file {file_name}: {exc}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xoá tệp: {exc}")

    # 3. Rebuild retriever
    try:
        existing_index = state.db_manager.get_existing_index()
        if existing_index:
            nodes = list(existing_index.docstore.docs.values())
            async with state_lock:
                state.retriever = AdvancedHybridRetriever(
                    existing_index, nodes, state.generator.llm, state.db_manager
                )
    except Exception as exc:
        logger.warning(f"⚠️ Retriever rebuild after delete failed: {exc}")

    return {
        "status": "success",
        "file_name": file_name,
        "nodes_deleted": deleted_count,
    }


@app.post("/api/v1/index")
@limiter.limit(os.getenv("RATE_LIMIT_HEAVY", "5/minute"))
async def build_index_endpoint(request: Request, api_key: str | None = Depends(verify_api_key)):
    """
    Trigger incremental ingestion pipeline in background and return job_id immediately.
    """
    try:
        data_dir = os.getenv("DATA_DIR", "data")
        job_id = await _start_ingestion_job(data_dir)
        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Ingestion đang chạy nền. Poll /api/v1/ingestion/status/{job_id} để theo dõi.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/ingestion/status/{job_id}")
@limiter.limit(os.getenv("RATE_LIMIT_LIGHT", "60/minute"))
async def ingestion_job_status(request: Request, job_id: str, api_key: str | None = Depends(verify_api_key)):
    """
    Check the status of a background ingestion job.
    Returns the current job state (queued/running/success/error).
    """
    async with _ingestion_jobs_lock:
        job = _ingestion_jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy job ingestion '{job_id}'.",
            )
        # Return a clean copy
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "created_at": job.get("created_at"),
            "total_nodes": job.get("total_nodes"),
            "processed_files": job.get("processed_files"),
            "error": job.get("error"),
            "result": job.get("result"),
        }


@app.delete("/api/v1/ingestion/jobs/{job_id}")
@limiter.limit(os.getenv("RATE_LIMIT_MEDIUM", "30/minute"))
async def clear_ingestion_job(request: Request, job_id: str, api_key: str | None = Depends(verify_api_key)):
    """Remove a finished ingestion job from tracking (cleanup)."""
    async with _ingestion_jobs_lock:
        job = _ingestion_jobs.pop(job_id, None)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy job '{job_id}'.")
    return {"status": "cleared", "job_id": job_id}


# ── Admin: API Key Management ────────────────────────────────────────────

from key_manager import create_key, get_all_keys, revoke_key, count_active_keys
from pydantic import BaseModel as PydanticBaseModel


class CreateKeyRequest(PydanticBaseModel):
    name: str = ""


@app.get("/api/v1/admin/keys")
@limiter.limit(os.getenv("RATE_LIMIT_MEDIUM", "30/minute"))
async def admin_list_keys(request: Request, api_key: str | None = Depends(verify_api_key)):
    """List all API keys with metadata (prefixes only, not full keys)."""
    keys = get_all_keys()
    active = count_active_keys()
    return {"keys": keys, "total": len(keys), "active": active}


@app.post("/api/v1/admin/keys")
@limiter.limit(os.getenv("RATE_LIMIT_HEAVY", "10/minute"))
async def admin_create_key(request: Request, body: CreateKeyRequest, api_key: str | None = Depends(verify_api_key)):
    """Create a new API key. Returns the full key ONCE."""
    full_key, meta = create_key(name=body.name)
    return {
        "key": full_key,
        "key_prefix": meta["prefix"],
        "name": meta["name"],
        "created_at": meta["created_at"],
        "message": "API key created. Lưu key này — nó sẽ không được hiển thị lại.",
    }


@app.delete("/api/v1/admin/keys/{key_prefix}")
@limiter.limit(os.getenv("RATE_LIMIT_HEAVY", "10/minute"))
async def admin_revoke_key(request: Request, key_prefix: str, api_key: str | None = Depends(verify_api_key)):
    """Revoke an API key by its prefix."""
    if revoke_key(key_prefix):
        return {"status": "revoked", "key_prefix": key_prefix}
    raise HTTPException(status_code=404, detail=f"Không tìm thấy key: {key_prefix}")


# Mount UI static files — serve frontend cùng origin với API
# Phải mount ở cuối để API routes (/health, /api/v1/*) chiếm ưu tiên trước
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)