"""Tests for sorter.py"""
import json

import pytest

from sorter import Sorter, resolve_destination
from pathlib import Path


# ---------------------------------------------------------------------------
# resolve_destination
# ---------------------------------------------------------------------------


class TestResolveDestination:
    def test_no_conflict(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        src = tmp_path / "file.txt"

        dest = resolve_destination(src, output, "Finance", "file.txt")
        assert dest == output / "Finance" / "file.txt"

    def test_single_conflict(self, tmp_path):
        output = tmp_path / "output"
        (output / "Finance").mkdir(parents=True)
        (output / "Finance" / "file.txt").write_text("existing")

        src = tmp_path / "file.txt"
        dest = resolve_destination(src, output, "Finance", "file.txt")
        assert dest == output / "Finance" / "file_1.txt"

    def test_multiple_conflicts(self, tmp_path):
        output = tmp_path / "output"
        (output / "Finance").mkdir(parents=True)
        (output / "Finance" / "file.txt").write_text("v1")
        (output / "Finance" / "file_1.txt").write_text("v2")

        src = tmp_path / "file.txt"
        dest = resolve_destination(src, output, "Finance", "file.txt")
        assert dest == output / "Finance" / "file_2.txt"

    def test_nested_folder(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        src = tmp_path / "report.pdf"

        dest = resolve_destination(src, output, "Work/Reports", "report.pdf")
        assert dest == output / "Work" / "Reports" / "report.pdf"


# ---------------------------------------------------------------------------
# Sorter – dry-run behaviour
# ---------------------------------------------------------------------------


class TestSorterDryRun:
    def _config(self, tmp_path):
        return {"sort_log": str(tmp_path / "sort_log.jsonl")}

    def _meta(self, doc: Path, category="Finance/Invoices", confidence=0.9, destination="Finance/Invoices"):
        return {
            "path": str(doc),
            "filename": doc.name,
            "classification": category,
            "confidence": confidence,
            "destination": destination,
        }

    def test_dry_run_does_not_move_file(self, tmp_path):
        doc = tmp_path / "invoice.txt"
        doc.write_text("Invoice content")
        out = tmp_path / "output"
        out.mkdir()

        sorter = Sorter(self._config(tmp_path), out, dry_run=True)
        result = sorter.sort_document(self._meta(doc))

        assert result["action"] == "would_move"
        assert doc.exists()  # untouched

    def test_execute_moves_file(self, tmp_path):
        doc = tmp_path / "invoice.txt"
        doc.write_text("Invoice content")
        out = tmp_path / "output"
        out.mkdir()

        sorter = Sorter(self._config(tmp_path), out, dry_run=False)
        result = sorter.sort_document(self._meta(doc))

        assert result["action"] == "moved"
        assert not doc.exists()
        assert (out / "Finance" / "Invoices" / "invoice.txt").exists()

    def test_never_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "output"
        (out / "Finance" / "Invoices").mkdir(parents=True)
        existing = out / "Finance" / "Invoices" / "invoice.txt"
        existing.write_text("original")

        doc = tmp_path / "invoice.txt"
        doc.write_text("new content")

        sorter = Sorter(self._config(tmp_path), out, dry_run=False)
        result = sorter.sort_document(self._meta(doc))

        assert result["action"] == "moved"
        assert existing.read_text() == "original"  # original unchanged
        assert (out / "Finance" / "Invoices" / "invoice_1.txt").exists()

    def test_skips_needs_review(self, tmp_path):
        doc = tmp_path / "mystery.txt"
        doc.write_text("content")
        out = tmp_path / "output"
        out.mkdir()

        sorter = Sorter(self._config(tmp_path), out, dry_run=False)
        result = sorter.sort_document(self._meta(doc, category="needs_review", confidence=0.3))

        assert result["action"] == "skipped"
        assert doc.exists()

    def test_skips_empty_classification(self, tmp_path):
        doc = tmp_path / "unknown.txt"
        doc.write_text("content")
        out = tmp_path / "output"
        out.mkdir()

        sorter = Sorter(self._config(tmp_path), out, dry_run=False)
        result = sorter.sort_document(self._meta(doc, category="", confidence=0.0))

        assert result["action"] == "skipped"
        assert doc.exists()

    def test_missing_source_file(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        sorter = Sorter(self._config(tmp_path), out, dry_run=False)
        meta = {
            "path": str(tmp_path / "ghost.txt"),
            "filename": "ghost.txt",
            "classification": "Work",
            "confidence": 0.9,
            "destination": "Work",
        }
        result = sorter.sort_document(meta)
        assert result["action"] == "missing"

    def test_logs_action_to_jsonl(self, tmp_path):
        doc = tmp_path / "report.txt"
        doc.write_text("Report content")
        out = tmp_path / "output"
        out.mkdir()

        log_path = tmp_path / "sort_log.jsonl"
        sorter = Sorter({"sort_log": str(log_path)}, out, dry_run=True)
        sorter.sort_document(self._meta(doc, category="Work/Reports", destination="Work/Reports"))

        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["action"] == "would_move"
        assert entry["dry_run"] is True
        assert entry["category"] == "Work/Reports"

    def test_creates_destination_directory(self, tmp_path):
        doc = tmp_path / "paper.txt"
        doc.write_text("Research paper")
        out = tmp_path / "output"
        out.mkdir()

        sorter = Sorter(self._config(tmp_path), out, dry_run=False)
        sorter.sort_document(self._meta(doc, category="Research/Papers", destination="Research/Papers"))

        assert (out / "Research" / "Papers").is_dir()
