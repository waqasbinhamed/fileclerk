"""Tests for extractor.py"""
from extractor import MAX_CHARS, _clean_text, extract_text


class TestPlainText:
    def test_txt_extraction(self, sample_txt):
        result = extract_text(sample_txt)
        assert result is not None
        assert result["filetype"] == ".txt"
        assert result["extraction_method"] == "plain_text"
        assert len(result["text_snippet"]) > 0

    def test_md_extraction(self, sample_md):
        result = extract_text(sample_md)
        assert result is not None
        assert result["filetype"] == ".md"
        assert "Research" in result["text_snippet"]

    def test_csv_extraction(self, sample_csv):
        result = extract_text(sample_csv)
        assert result is not None
        assert result["filetype"] == ".csv"
        assert "Alice" in result["text_snippet"]

    def test_py_extraction(self, sample_py):
        result = extract_text(sample_py)
        assert result is not None
        assert result["filetype"] == ".py"
        assert "hello" in result["text_snippet"]

    def test_snippet_capped_at_max_chars(self, sample_txt):
        result = extract_text(sample_txt)
        assert result["char_count"] <= MAX_CHARS

    def test_result_has_required_keys(self, sample_txt):
        result = extract_text(sample_txt)
        for key in ("path", "filename", "filetype", "text_snippet", "char_count", "extraction_method"):
            assert key in result


class TestNotebook:
    def test_ipynb_extraction(self, sample_ipynb):
        result = extract_text(sample_ipynb)
        assert result is not None
        assert result["filetype"] == ".ipynb"
        assert result["extraction_method"] == "notebook"
        # Should contain code or markdown cell source
        assert "numpy" in result["text_snippet"] or "Analysis" in result["text_snippet"]


class TestDocx:
    def test_docx_extraction(self, sample_docx):
        result = extract_text(sample_docx)
        assert result is not None
        assert result["filetype"] == ".docx"
        assert result["extraction_method"] == "python-docx"
        assert "Invoice" in result["text_snippet"] or "Services" in result["text_snippet"]


class TestXlsx:
    def test_xlsx_extraction(self, sample_xlsx):
        result = extract_text(sample_xlsx)
        assert result is not None
        assert result["filetype"] == ".xlsx"
        assert result["extraction_method"] == "openpyxl"
        assert "Item" in result["text_snippet"] or "Service" in result["text_snippet"]


class TestUnsupportedType:
    def test_unsupported_returns_none(self, tmp_path):
        f = tmp_path / "photo.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = extract_text(f)
        assert result is None

    def test_unsupported_writes_skipped_log(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04")
        extract_text(f)
        assert (tmp_path / "skipped.log").exists()


class TestCleanText:
    def test_removes_blank_lines(self):
        text = "Line 1\n\n\nLine 2\n\nLine 3"
        result = _clean_text(text)
        assert "\n\n" not in result

    def test_strips_whitespace_from_lines(self):
        text = "  Line 1  \n  Line 2  \n"
        result = _clean_text(text)
        for line in result.splitlines():
            assert line == line.strip()

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_all_blank_lines(self):
        assert _clean_text("\n\n\n") == ""
