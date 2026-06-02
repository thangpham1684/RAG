import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

from dotenv import load_dotenv
from logging_config import get_logger

logger = get_logger(__name__)

from embeddings.chunker import AdvancedChunker
from embeddings.vector_db import QdrantDBManager
from parsers.router import DocumentRouter

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".xlsx", ".csv", ".pptx"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _write_status(path: str, status: str, message: str | None = None, total_nodes: int | None = None, extra=None):
    payload = {
        "status": status,
        "message": message,
        "total_nodes": total_nodes,
        "timestamp": _now(),
    }
    if extra:
        payload.update(extra)
    _atomic_write_json(path, payload)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_files(data_dir: str):
    files = {}
    for root, _, names in os.walk(data_dir):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, data_dir).replace("\\", "/")
            stat = os.stat(abs_path)
            files[rel_path] = {
                "rel_path": rel_path,
                "abs_path": abs_path,
                "file_name": name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": _sha256_file(abs_path),
            }
    return files


def _dataset_key(data_dir: str) -> str:
    return os.path.normcase(os.path.abspath(data_dir))


def _compute_delta(current_files, manifest_files):
    changed = []
    for rel_path, meta in current_files.items():
        prev = manifest_files.get(rel_path)
        if not prev:
            changed.append(meta)
            continue
        if prev.get("sha256") != meta["sha256"] or prev.get("size") != meta["size"]:
            changed.append(meta)

    deleted = [rel_path for rel_path in manifest_files.keys() if rel_path not in current_files]
    changed.sort(key=lambda x: x["rel_path"])
    deleted.sort()
    return changed, deleted


def _new_job(data_dir: str, changed_files, deleted_files, max_retries: int, full_rebuild: bool = False):
    deleted_file_names = sorted({f["file_name"] for f in deleted_files})
    return {
        "job_id": f"ingest-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "data_dir": data_dir,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "max_retries": max_retries,
        "full_rebuild": full_rebuild,
        "deleted_files": [f["rel_path"] for f in deleted_files],
        "deleted_file_names": deleted_file_names,
        "files": [
            {
                "rel_path": f["rel_path"],
                "file_name": f["file_name"],
                "abs_path": f["abs_path"],
                "sha256": f["sha256"],
                "attempts": 0,
                "status": "pending",
                "error": None,
                "nodes_indexed": 0,
            }
            for f in changed_files
        ],
        "processed_files": 0,
        "total_nodes": 0,
    }


def _load_queue(queue_path: str):
    return _safe_load_json(queue_path, {"jobs": []})


def _save_queue(queue_path: str, queue_payload):
    _atomic_write_json(queue_path, queue_payload)


def _pick_resumable_job(queue_payload, data_dir: str):
    for job in queue_payload["jobs"]:
        if job.get("data_dir") != data_dir:
            continue
        if job.get("status") in {"queued", "running", "error"}:
            return job
    return None


def _stage_single_file(source_abs_path: str):
    tmp_dir = tempfile.mkdtemp(prefix="ingest_stage_")
    target = os.path.join(tmp_dir, os.path.basename(source_abs_path))
    shutil.copy2(source_abs_path, target)
    return tmp_dir


import concurrent.futures


def _ingest_file(file_item, db_manager: QdrantDBManager, router: DocumentRouter, chunker: AdvancedChunker, timeout_seconds: int = 300):
    """Xử lý một file với cơ chế timeout (cross-platform, dùng ThreadPoolExecutor)."""
    def _process():
        stage_dir = _stage_single_file(file_item["abs_path"])
        try:
            docs = router.process_directory(stage_dir)
            docs = [d for d in docs if str(d.metadata.get("file_name", "")) == file_item["file_name"]]
            if not docs:
                raise RuntimeError(f"No parsed content for file: {file_item['file_name']}")

            nodes = chunker.split_into_chunks(docs)
            db_manager.delete_file_nodes(file_item["file_name"])
            db_manager.save_and_index(nodes)
            return len(nodes)
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_process)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        logger.error(f"⏰ Timeout khi xử lý file: {file_item['file_name']} (>{timeout_seconds}s)")
        raise RuntimeError(f"File processing timed out after {timeout_seconds}s: {file_item['file_name']}")
    finally:
        executor.shutdown(wait=False)  # Không chờ thread đang chạy, để GC dọn sau


def run_ingestion_job(
    data_dir: str = "data",
    max_retries: int = 2,
    resume: bool = True,
    enqueue_only: bool = False,
    full_rebuild: bool = False,
):
    load_dotenv()
    os.makedirs(data_dir, exist_ok=True)

    db_manager = QdrantDBManager()
    status_path = os.path.join(db_manager.persist_dir, "ingestion_status.json")
    manifest_path = os.path.join(db_manager.persist_dir, "ingestion_manifest.json")
    queue_path = os.path.join(db_manager.persist_dir, "ingestion_queue.json")

    manifest = _safe_load_json(manifest_path, {"datasets": {}, "updated_at": None})
    data_key = _dataset_key(data_dir)
    if "datasets" in manifest:
        dataset_manifest = manifest["datasets"].get(data_key, {"files": {}, "updated_at": None})
    else:
        # Backward compatibility with old single-dataset schema.
        dataset_manifest = {
            "files": manifest.get("files", {}),
            "updated_at": manifest.get("updated_at"),
        }
    current_files = _scan_files(data_dir)
    changed_files, deleted_rel_paths = _compute_delta(current_files, dataset_manifest.get("files", {}))
    deleted_files = []
    for rel_path in deleted_rel_paths:
        prev = dataset_manifest.get("files", {}).get(rel_path, {})
        deleted_files.append(
            {
                "rel_path": rel_path,
                "file_name": prev.get("file_name", os.path.basename(rel_path)),
            }
        )

    queue_payload = _load_queue(queue_path)
    job = _pick_resumable_job(queue_payload, data_dir) if resume else None
    if job is None:
        if not changed_files and not deleted_files and not full_rebuild:
            _write_status(
                status_path,
                "success",
                message="No file changes detected. Incremental ingestion skipped.",
                total_nodes=0,
                extra={"processed_files": 0, "queued_files": 0},
            )
            return {"status": "success", "message": "No changes", "total_nodes": 0, "processed_files": 0}
        job = _new_job(
            data_dir,
            changed_files,
            deleted_files,
            max_retries=max_retries,
            full_rebuild=full_rebuild,
        )
        queue_payload["jobs"].append(job)
        _save_queue(queue_path, queue_payload)

    if enqueue_only:
        _write_status(
            status_path,
            "queued",
            message=f"Queued ingestion job {job['job_id']}",
            total_nodes=job.get("total_nodes", 0),
            extra={"job_id": job["job_id"], "queued_files": len(job["files"])},
        )
        return {"status": "queued", "job_id": job["job_id"], "queued_files": len(job["files"])}

    job["status"] = "running"
    job["updated_at"] = _now()
    _save_queue(queue_path, queue_payload)
    _write_status(
        status_path,
        "running",
        message=f"Ingestion job {job['job_id']} running",
        total_nodes=job.get("total_nodes", 0),
        extra={"job_id": job["job_id"], "queued_files": len(job["files"])},
    )

    router = DocumentRouter()
    chunker = AdvancedChunker()

    try:
        if job.get("full_rebuild"):
            db_manager.reset_index_storage()
            job["files"] = [
                {
                    "rel_path": f["rel_path"],
                    "file_name": f["file_name"],
                    "abs_path": f["abs_path"],
                    "sha256": f["sha256"],
                    "attempts": 0,
                    "status": "pending",
                    "error": None,
                    "nodes_indexed": 0,
                }
                for f in sorted(current_files.values(), key=lambda x: x["rel_path"])
            ]
        else:
            for deleted_file_name in job.get("deleted_file_names", []):
                db_manager.delete_file_nodes(deleted_file_name)

        for file_item in job["files"]:
            if file_item["status"] == "done":
                continue

            success = False
            while file_item["attempts"] <= job["max_retries"] and not success:
                try:
                    file_item["attempts"] += 1
                    nodes_count = _ingest_file(file_item, db_manager, router, chunker)
                    file_item["nodes_indexed"] = nodes_count
                    file_item["status"] = "done"
                    file_item["error"] = None
                    job["processed_files"] += 1
                    job["total_nodes"] += nodes_count
                    success = True
                except Exception as exc:
                    file_item["status"] = "retry"
                    file_item["error"] = str(exc)
                    if file_item["attempts"] > job["max_retries"]:
                        file_item["status"] = "failed"
                finally:
                    job["updated_at"] = _now()
                    _save_queue(queue_path, queue_payload)
                    _write_status(
                        status_path,
                        "running",
                        message=f"Processing {file_item['file_name']}",
                        total_nodes=job["total_nodes"],
                        extra={
                            "job_id": job["job_id"],
                            "processed_files": job["processed_files"],
                            "queued_files": len(job["files"]),
                            "current_file": file_item["file_name"],
                            "current_file_status": file_item["status"],
                        },
                    )

            if file_item["status"] != "done":
                raise RuntimeError(f"File failed after retries: {file_item['file_name']} - {file_item['error']}")

        dataset_manifest["files"] = {
            rel_path: {
                "sha256": meta["sha256"],
                "size": meta["size"],
                "mtime": meta["mtime"],
                "file_name": meta["file_name"],
            }
            for rel_path, meta in current_files.items()
        }
        dataset_manifest["updated_at"] = _now()
        manifest.setdefault("datasets", {})
        manifest["datasets"][data_key] = dataset_manifest
        manifest["updated_at"] = _now()
        _atomic_write_json(manifest_path, manifest)

        job["status"] = "success"
        job["updated_at"] = _now()
        _save_queue(queue_path, queue_payload)
        _write_status(
            status_path,
            "success",
            message=f"Ingestion job {job['job_id']} completed",
            total_nodes=job["total_nodes"],
            extra={
                "job_id": job["job_id"],
                "processed_files": job["processed_files"],
                "queued_files": len(job["files"]),
                "full_rebuild": job.get("full_rebuild", False),
            },
        )
        return {
            "status": "success",
            "job_id": job["job_id"],
            "total_nodes": job["total_nodes"],
            "processed_files": job["processed_files"],
            "queued_files": len(job["files"]),
            "full_rebuild": job.get("full_rebuild", False),
        }
    except Exception as exc:
        job["status"] = "error"
        job["updated_at"] = _now()
        _save_queue(queue_path, queue_payload)
        _write_status(
            status_path,
            "error",
            message=str(exc),
            total_nodes=job.get("total_nodes", 0),
            extra={"job_id": job["job_id"], "processed_files": job.get("processed_files", 0)},
        )
        raise


def run_queue_once(data_dir: str = "data", max_retries: int = 2):
    return run_ingestion_job(data_dir=data_dir, max_retries=max_retries, resume=True, enqueue_only=False)


def main():
    parser = argparse.ArgumentParser(description="Incremental ingestion worker with queue/retry/resume.")
    parser.add_argument("--data-dir", default="data", help="Directory containing input documents.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count per file.")
    parser.add_argument("--enqueue-only", action="store_true", help="Only enqueue ingestion job, do not process now.")
    parser.add_argument("--run-queue", action="store_true", help="Process one queued/resumable ingestion job.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore resumable jobs and create a new job.")
    parser.add_argument("--full-rebuild", action="store_true", help="Force full rebuild for this data-dir.")
    args = parser.parse_args()

    try:
        if args.run_queue:
            result = run_queue_once(data_dir=args.data_dir, max_retries=args.max_retries)
        else:
            result = run_ingestion_job(
                data_dir=args.data_dir,
                max_retries=args.max_retries,
                resume=not args.no_resume,
                enqueue_only=args.enqueue_only,
                full_rebuild=args.full_rebuild,
            )
        print(json.dumps(result, ensure_ascii=False))  # JSON output for CLI piping
        return 0
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
