"""Unit tests for ingest_worker.py — pure helper functions tested with temp files."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

# Functions under test
from ingest_worker import (
    _safe_load_json,
    _atomic_write_json,
    _sha256_file,
    _scan_files,
    _dataset_key,
    _compute_delta,
    _new_job,
    _load_queue,
    _save_queue,
    _pick_resumable_job,
    _stage_single_file,
    SUPPORTED_EXTS,
)


class TestSafeLoadJson(unittest.TestCase):
    """_safe_load_json: load JSON with fallback."""

    def test_load_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
            path = f.name
        try:
            result = _safe_load_json(path, {"default": True})
            self.assertEqual(result, {"key": "value"})
        finally:
            os.remove(path)

    def test_load_missing(self):
        result = _safe_load_json("/nonexistent/path.json", {"default": True})
        self.assertEqual(result, {"default": True})

    def test_load_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(json.JSONDecodeError):
                _safe_load_json(path, {"fallback": True})
        finally:
            os.remove(path)

    def test_load_corrupted_json_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{invalid json}")
            path = f.name
        try:
            with self.assertRaises(json.JSONDecodeError):
                _safe_load_json(path, {"fallback": True})
        finally:
            os.remove(path)


class TestAtomicWriteJson(unittest.TestCase):
    """_atomic_write_json: atomic JSON write via tmp + replace."""

    def test_write_and_read(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "test.json")
            payload = {"a": 1, "b": [2, 3]}
            _atomic_write_json(path, payload)

            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_unicode_content(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "unicode.json")
            payload = {"text": "Tiếng Việt có dấu: ắ, ễ, ơ"}
            _atomic_write_json(path, payload)

            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_overwrite_existing(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "overwrite.json")
            _atomic_write_json(path, {"v1": True})
            _atomic_write_json(path, {"v2": True})

            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"v2": True})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_no_tmp_left_behind(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "clean.json")
            _atomic_write_json(path, {"clean": True})

            tmp_files = [f for f in os.listdir(tmp_dir) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_creates_missing_directory(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            nested = os.path.join(tmp_dir, "sub", "nested", "test.json")
            _atomic_write_json(nested, {"nested": True})
            self.assertTrue(os.path.exists(nested))

            with open(nested, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"nested": True})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestSha256File(unittest.TestCase):
    """_sha256_file: compute SHA-256 hash of a file."""

    def test_known_hash(self):
        content = b"Hello, World!"
        expected = hashlib.sha256(content).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = _sha256_file(path)
            self.assertEqual(result, expected)
        finally:
            os.remove(path)

    def test_empty_file(self):
        expected = hashlib.sha256(b"").hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            result = _sha256_file(path)
            self.assertEqual(result, expected)
        finally:
            os.remove(path)

    def test_large_file(self):
        """Test with content larger than the 1MB read chunks."""
        content = b"A" * (2 * 1024 * 1024 + 13)  # ~2MB + 13 bytes
        expected = hashlib.sha256(content).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = _sha256_file(path)
            self.assertEqual(result, expected)
        finally:
            os.remove(path)


class TestScanFiles(unittest.TestCase):
    """_scan_files: walk data dir and collect supported files with metadata."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _touch(self, rel_path: str, content: bytes = b"dummy"):
        full = os.path.join(self.tmp_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(content)
        return full

    def test_finds_supported_extensions(self):
        self._touch("doc1.pdf")
        self._touch("doc2.docx")
        self._touch("data.csv")
        files = _scan_files(self.tmp_dir)
        self.assertEqual(len(files), 3)
        self.assertIn("doc1.pdf", files)
        self.assertIn("doc2.docx", files)
        self.assertIn("data.csv", files)

    def test_skips_unsupported_extensions(self):
        self._touch("notes.txt")
        self._touch("image.png")
        self._touch("script.py")
        files = _scan_files(self.tmp_dir)
        self.assertEqual(len(files), 0)

    def test_skips_directories(self):
        os.makedirs(os.path.join(self.tmp_dir, "subdir"), exist_ok=True)
        files = _scan_files(self.tmp_dir)
        self.assertEqual(len(files), 0)

    def test_nested_directories(self):
        self._touch("sub/a.pdf")
        self._touch("sub/deep/b.docx")
        files = _scan_files(self.tmp_dir)
        self.assertIn("sub/a.pdf", files)
        self.assertIn("sub/deep/b.docx", files)

    def test_mixed_supported_and_unsupported(self):
        self._touch("keep.pdf")
        self._touch("skip.txt")
        self._touch("also_keep.xlsx")
        files = _scan_files(self.tmp_dir)
        self.assertIn("keep.pdf", files)
        self.assertIn("also_keep.xlsx", files)
        self.assertNotIn("skip.txt", files)
        self.assertEqual(len(files), 2)

    def test_metadata_fields(self):
        content = b"test content"
        self._touch("report.pdf", content)
        files = _scan_files(self.tmp_dir)
        meta = files["report.pdf"]
        self.assertEqual(meta["file_name"], "report.pdf")
        self.assertEqual(meta["rel_path"], "report.pdf")
        self.assertIn("abs_path", meta)
        self.assertIn("size", meta)
        self.assertIn("mtime", meta)
        self.assertIn("sha256", meta)
        self.assertEqual(meta["size"], len(content))
        self.assertEqual(meta["sha256"], hashlib.sha256(content).hexdigest())

    def test_empty_directory(self):
        files = _scan_files(self.tmp_dir)
        self.assertEqual(files, {})


class TestDatasetKey(unittest.TestCase):
    """_dataset_key: normalize path for dict key."""

    def test_absolute_path_normalized(self):
        key = _dataset_key("/some/path")
        expected = os.path.normcase(os.path.abspath("/some/path"))
        self.assertEqual(key, expected)

    def test_relative_path_resolved(self):
        key = _dataset_key("data")
        expected = os.path.normcase(os.path.abspath("data"))
        self.assertEqual(key, expected)

    def test_trailing_separator(self):
        key1 = _dataset_key("/path/to/dir")
        key2 = _dataset_key("/path/to/dir/")
        self.assertEqual(key1, key2)


class TestComputeDelta(unittest.TestCase):
    """_compute_delta: detect changed, new, and deleted files."""

    @staticmethod
    def _entry(rel_path, sha256, size):
        """Build a file entry matching _scan_files output (must include rel_path)."""
        return {"rel_path": rel_path, "sha256": sha256, "size": size}

    def test_no_changes(self):
        current = {
            "a.pdf": self._entry("a.pdf", "abc", 100),
            "b.pdf": self._entry("b.pdf", "def", 200),
        }
        manifest = {
            "a.pdf": {"sha256": "abc", "size": 100},
            "b.pdf": {"sha256": "def", "size": 200},
        }
        changed, deleted = _compute_delta(current, manifest)
        self.assertEqual(changed, [])
        self.assertEqual(deleted, [])

    def test_new_file(self):
        current = {
            "existing.pdf": self._entry("existing.pdf", "abc", 100),
            "new.pdf": self._entry("new.pdf", "xyz", 50),
        }
        manifest = {
            "existing.pdf": {"sha256": "abc", "size": 100},
        }
        changed, deleted = _compute_delta(current, manifest)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["rel_path"], "new.pdf")
        self.assertEqual(deleted, [])

    def test_modified_sha256(self):
        current = {
            "a.pdf": self._entry("a.pdf", "newhash", 100),
        }
        manifest = {
            "a.pdf": {"sha256": "oldhash", "size": 100},
        }
        changed, deleted = _compute_delta(current, manifest)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["rel_path"], "a.pdf")

    def test_modified_size(self):
        current = {
            "a.pdf": self._entry("a.pdf", "samehash", 999),
        }
        manifest = {
            "a.pdf": {"sha256": "samehash", "size": 100},
        }
        changed, deleted = _compute_delta(current, manifest)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["rel_path"], "a.pdf")

    def test_deleted_file(self):
        current = {
            "a.pdf": self._entry("a.pdf", "abc", 100),
        }
        manifest = {
            "a.pdf": {"sha256": "abc", "size": 100},
            "deleted.pdf": {"sha256": "xyz", "size": 200},
        }
        changed, deleted = _compute_delta(current, manifest)
        self.assertEqual(changed, [])
        self.assertEqual(deleted, ["deleted.pdf"])

    def test_mixed_changes(self):
        current = {
            "unchanged.pdf": self._entry("unchanged.pdf", "abc", 100),
            "new.pdf": self._entry("new.pdf", "xyz", 50),
            "modified.pdf": self._entry("modified.pdf", "newhash", 300),
        }
        manifest = {
            "unchanged.pdf": {"sha256": "abc", "size": 100},
            "modified.pdf": {"sha256": "oldhash", "size": 300},
            "deleted.pdf": {"sha256": "gone", "size": 500},
        }
        changed, deleted = _compute_delta(current, manifest)
        self.assertEqual(len(changed), 2)  # new.pdf + modified.pdf
        self.assertEqual(deleted, ["deleted.pdf"])
        # Result is sorted
        self.assertEqual(changed[0]["rel_path"], "modified.pdf")
        self.assertEqual(changed[1]["rel_path"], "new.pdf")

    def test_empty_current(self):
        manifest = {"a.pdf": {"sha256": "abc", "size": 100}}
        changed, deleted = _compute_delta({}, manifest)
        self.assertEqual(changed, [])
        self.assertEqual(deleted, ["a.pdf"])

    def test_empty_manifest(self):
        current = {"a.pdf": self._entry("a.pdf", "abc", 100)}
        changed, deleted = _compute_delta(current, {})
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["rel_path"], "a.pdf")
        self.assertEqual(deleted, [])


class TestNewJob(unittest.TestCase):
    """_new_job: create job dict with expected structure."""

    def test_structure(self):
        changed_files = [
            {"rel_path": "a.pdf", "file_name": "a.pdf", "abs_path": "/x/a.pdf", "sha256": "abc", "size": 100},
            {"rel_path": "b.pdf", "file_name": "b.pdf", "abs_path": "/x/b.pdf", "sha256": "def", "size": 200},
        ]
        deleted_files = [
            {"rel_path": "old.pdf", "file_name": "old.pdf"},
        ]
        job = _new_job("/data", changed_files, deleted_files, max_retries=2)

        self.assertIn("job_id", job)
        self.assertEqual(job["data_dir"], "/data")
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["max_retries"], 2)
        self.assertFalse(job["full_rebuild"])
        self.assertEqual(job["processed_files"], 0)
        self.assertEqual(job["total_nodes"], 0)

        # Deleted files
        self.assertEqual(job["deleted_files"], ["old.pdf"])
        self.assertEqual(job["deleted_file_names"], ["old.pdf"])

        # Files list
        self.assertEqual(len(job["files"]), 2)
        self.assertEqual(job["files"][0]["rel_path"], "a.pdf")
        self.assertEqual(job["files"][0]["attempts"], 0)
        self.assertEqual(job["files"][0]["status"], "pending")
        self.assertIsNone(job["files"][0]["error"])
        self.assertEqual(job["files"][0]["nodes_indexed"], 0)

    def test_files_preserves_order(self):
        """_new_job preserves the order of changed_files (sorted upstream by _compute_delta)."""
        changed_files = [
            {"rel_path": "a.pdf", "file_name": "a.pdf", "abs_path": "/a.pdf", "sha256": "a", "size": 1},
            {"rel_path": "z.pdf", "file_name": "z.pdf", "abs_path": "/z.pdf", "sha256": "z", "size": 1},
        ]
        job = _new_job("/data", changed_files, [], max_retries=1)
        self.assertEqual(job["files"][0]["rel_path"], "a.pdf")
        self.assertEqual(job["files"][1]["rel_path"], "z.pdf")
        self.assertEqual(len(job["files"]), 2)

    def test_full_rebuild_flag(self):
        job = _new_job("/data", [], [], max_retries=1, full_rebuild=True)
        self.assertTrue(job["full_rebuild"])

    def test_deleted_file_names_deduplicated(self):
        deleted_files = [
            {"rel_path": "dup.pdf", "file_name": "dup.pdf"},
            {"rel_path": "dup.pdf", "file_name": "dup.pdf"},
        ]
        # Since we use a set, duplicates are deduplicated
        job = _new_job("/data", [], deleted_files, max_retries=1)
        self.assertEqual(len(job["deleted_file_names"]), 1)
        self.assertEqual(job["deleted_file_names"], ["dup.pdf"])

    def test_no_changes(self):
        job = _new_job("/data", [], [], max_retries=1)
        self.assertEqual(job["files"], [])
        self.assertEqual(job["deleted_files"], [])


class TestLoadSaveQueue(unittest.TestCase):
    """_load_queue / _save_queue: roundtrip and edge cases."""

    def test_save_then_load(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "queue.json")
            payload = {"jobs": [{"id": 1}, {"id": 2}]}
            _save_queue(path, payload)

            loaded = _load_queue(path)
            self.assertEqual(loaded, payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_load_missing_queue(self):
        result = _load_queue("/nonexistent/queue.json")
        self.assertEqual(result, {"jobs": []})

    def test_load_empty_queue_file_raises(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "empty_queue.json")
            # Create empty file
            with open(path, "w") as f:
                pass
            # Empty file causes JSONDecodeError (propagated from _safe_load_json)
            with self.assertRaises(json.JSONDecodeError):
                _load_queue(path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_save_unicode(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "unicode_queue.json")
            payload = {"jobs": [{"name": "Tiếng Việt"}]}
            _save_queue(path, payload)

            loaded = _load_queue(path)
            self.assertEqual(loaded, payload)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestPickResumableJob(unittest.TestCase):
    """_pick_resumable_job: find resumable job by data_dir and status."""

    def test_picks_queued_job(self):
        payload = {
            "jobs": [
                {"data_dir": "/other", "status": "queued"},
                {"data_dir": "/data", "status": "queued"},
            ]
        }
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "queued")

    def test_picks_running_job(self):
        payload = {
            "jobs": [
                {"data_dir": "/other", "status": "queued"},
                {"data_dir": "/data", "status": "running"},
            ]
        }
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "running")

    def test_picks_error_job(self):
        payload = {
            "jobs": [
                {"data_dir": "/data", "status": "error"},
            ]
        }
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "error")

    def test_skips_success_job(self):
        payload = {
            "jobs": [
                {"data_dir": "/data", "status": "success"},
            ]
        }
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNone(job)

    def test_wrong_data_dir(self):
        payload = {
            "jobs": [
                {"data_dir": "/other", "status": "queued"},
            ]
        }
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNone(job)

    def test_prefers_first_matching(self):
        payload = {
            "jobs": [
                {"data_dir": "/data", "status": "queued"},
                {"data_dir": "/data", "status": "running"},
            ]
        }
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "queued")

    def test_empty_jobs(self):
        payload = {"jobs": []}
        job = _pick_resumable_job(payload, "/data")
        self.assertIsNone(job)

    def test_no_jobs_key(self):
        payload = {}
        # _pick_resumable_job accesses payload["jobs"] directly
        with self.assertRaises(KeyError):
            _pick_resumable_job(payload, "/data")


class TestStageSingleFile(unittest.TestCase):
    """_stage_single_file: copy file to temp directory."""

    def test_file_copied_correctly(self):
        tmp_src = tempfile.mkdtemp()
        try:
            src_path = os.path.join(tmp_src, "source.pdf")
            content = b"PDF content \x89\x50\x4e\x47"
            with open(src_path, "wb") as f:
                f.write(content)

            stage_dir = _stage_single_file(src_path)
            try:
                staged = os.path.join(stage_dir, "source.pdf")
                self.assertTrue(os.path.exists(staged))
                with open(staged, "rb") as f:
                    self.assertEqual(f.read(), content)
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)
        finally:
            shutil.rmtree(tmp_src, ignore_errors=True)

    def test_metadata_preserved(self):
        tmp_src = tempfile.mkdtemp()
        try:
            src_path = os.path.join(tmp_src, "doc.docx")
            with open(src_path, "wb") as f:
                f.write(b"word doc")

            src_stat = os.stat(src_path)
            stage_dir = _stage_single_file(src_path)
            try:
                staged = os.path.join(stage_dir, "doc.docx")
                staged_stat = os.stat(staged)
                # copy2 preserves mtime
                self.assertEqual(src_stat.st_mtime, staged_stat.st_mtime)
                self.assertEqual(src_stat.st_size, staged_stat.st_size)
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)
        finally:
            shutil.rmtree(tmp_src, ignore_errors=True)

    def test_stage_dir_is_different(self):
        tmp_src = tempfile.mkdtemp()
        try:
            src_path = os.path.join(tmp_src, "file.pdf")
            with open(src_path, "wb") as f:
                f.write(b"data")

            stage_dir = _stage_single_file(src_path)
            try:
                self.assertNotEqual(os.path.dirname(src_path), stage_dir)
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)
        finally:
            shutil.rmtree(tmp_src, ignore_errors=True)


class TestWriteStatus(unittest.TestCase):
    """_write_status (integration via _atomic_write_json)."""

    def test_write_status_structure(self):
        from ingest_worker import _write_status, _now

        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "status.json")
            _write_status(path, "running", message="Processing", total_nodes=42)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["status"], "running")
            self.assertEqual(data["message"], "Processing")
            self.assertEqual(data["total_nodes"], 42)
            self.assertIn("timestamp", data)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_write_status_with_extra(self):
        from ingest_worker import _write_status

        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "status.json")
            _write_status(path, "success", total_nodes=10, extra={"job_id": "abc", "processed_files": 5})

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["status"], "success")
            self.assertEqual(data["total_nodes"], 10)
            self.assertEqual(data["job_id"], "abc")
            self.assertEqual(data["processed_files"], 5)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
