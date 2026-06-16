"""Tests for classifier.py"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from classifier import _call_ollama, _parse_json_response, classify_document

TAXONOMY = [
    "Finance",
    "Finance/Invoices",
    "Finance/Tax",
    "Work",
    "Work/Reports",
    "Research",
    "Personal",
    "Unsorted",
]

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "ollama_model": "phi3:mini",
    "confidence_threshold": 0.6,
}

INVOICE_EXTRACTION = {
    "filename": "invoice_jan_2024.pdf",
    "text_snippet": "Invoice #1234\nDate: January 15, 2024\nAmount due: $500.00\nFor consulting services",
}


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_valid_json(self):
        text = '{"category": "Finance", "confidence": 0.9, "reason": "test"}'
        result = _parse_json_response(text)
        assert result["category"] == "Finance"
        assert result["confidence"] == 0.9

    def test_json_embedded_in_prose(self):
        text = 'Sure, here is the answer:\n{"category": "Finance/Invoices", "confidence": 0.85, "reason": "invoice"}\nThat is all.'
        result = _parse_json_response(text)
        assert result is not None
        assert result["category"] == "Finance/Invoices"

    def test_json_with_whitespace(self):
        text = '  \n  {"category": "Work", "confidence": 0.7, "reason": "code"}  \n  '
        result = _parse_json_response(text)
        assert result["category"] == "Work"

    def test_invalid_json_returns_none(self):
        assert _parse_json_response("not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_json_response("") is None


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------


class TestCallOllama:
    def _mock_response(self, response_text: str):
        r = MagicMock()
        r.json.return_value = {"response": response_text}
        return r

    def test_successful_call_returns_parsed_dict(self):
        payload = '{"category": "Finance/Invoices", "subcategory": "", "suggested_folder": "Finance/Invoices/2024", "confidence": 0.9, "reason": "invoice"}'
        with patch("classifier.requests.post", return_value=self._mock_response(payload)):
            result = _call_ollama("http://localhost:11434", "phi3:mini", "test prompt")

        assert result is not None
        assert result["category"] == "Finance/Invoices"
        assert result["confidence"] == 0.9

    def test_json_parse_failure_retries_once(self):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._mock_response("not valid json at all")
            return self._mock_response('{"category": "Finance", "confidence": 0.8, "reason": "retry ok"}')

        with patch("classifier.requests.post", side_effect=side_effect):
            with patch("classifier.time.sleep"):
                result = _call_ollama("http://localhost:11434", "phi3:mini", "test prompt")

        assert result is not None
        assert call_count == 2  # called twice — one retry

    def test_both_attempts_fail_returns_none(self):
        with patch("classifier.requests.post", return_value=self._mock_response("not json")):
            with patch("classifier.time.sleep"):
                result = _call_ollama("http://localhost:11434", "phi3:mini", "test prompt")

        assert result is None

    def test_http_error_returns_none_immediately(self):
        with patch("classifier.requests.post", side_effect=requests.RequestException("Connection refused")):
            result = _call_ollama("http://localhost:11434", "phi3:mini", "test prompt")

        assert result is None

    def test_timeout_returns_none(self):
        with patch("classifier.requests.post", side_effect=requests.Timeout("timed out")):
            result = _call_ollama("http://localhost:11434", "phi3:mini", "test prompt")

        assert result is None


# ---------------------------------------------------------------------------
# classify_document
# ---------------------------------------------------------------------------


class TestClassifyDocument:
    def test_high_confidence_classified(self):
        mock_result = {
            "category": "Finance/Invoices",
            "subcategory": "consulting",
            "suggested_folder": "Finance/Invoices/2024",
            "confidence": 0.92,
            "reason": "Document contains invoice data with amount and date.",
        }
        with patch("classifier._call_ollama", return_value=mock_result):
            result = classify_document(INVOICE_EXTRACTION, TAXONOMY, CONFIG)

        assert result["status"] == "classified"
        assert result["category"] == "Finance/Invoices"
        assert result["confidence"] == 0.92

    def test_low_confidence_marked_needs_review(self):
        mock_result = {
            "category": "Finance",
            "subcategory": "",
            "suggested_folder": "Finance",
            "confidence": 0.45,  # below 0.6 threshold
            "reason": "Possibly financial but uncertain.",
        }
        with patch("classifier._call_ollama", return_value=mock_result):
            result = classify_document(INVOICE_EXTRACTION, TAXONOMY, CONFIG)

        assert result["status"] == "needs_review"
        assert result["category"] == "needs_review"

    def test_exactly_at_threshold_classified(self):
        mock_result = {
            "category": "Work",
            "subcategory": "",
            "suggested_folder": "Work",
            "confidence": 0.6,  # exactly at threshold — should pass
            "reason": "Work-related document.",
        }
        with patch("classifier._call_ollama", return_value=mock_result):
            result = classify_document(INVOICE_EXTRACTION, TAXONOMY, CONFIG)

        # 0.6 >= 0.6 → classified
        assert result["status"] == "classified"

    def test_ollama_failure_returns_failed_status(self):
        with patch("classifier._call_ollama", return_value=None):
            result = classify_document(INVOICE_EXTRACTION, TAXONOMY, CONFIG)

        assert result["status"] == "failed"
        assert result["category"] == "needs_review"
        assert result["confidence"] == 0.0

    def test_custom_confidence_threshold(self):
        config_strict = {**CONFIG, "confidence_threshold": 0.9}
        mock_result = {
            "category": "Finance/Invoices",
            "subcategory": "",
            "suggested_folder": "Finance/Invoices",
            "confidence": 0.8,  # passes 0.6 but fails 0.9
            "reason": "Invoice.",
        }
        with patch("classifier._call_ollama", return_value=mock_result):
            result = classify_document(INVOICE_EXTRACTION, TAXONOMY, config_strict)

        assert result["status"] == "needs_review"

    def test_prompt_includes_taxonomy(self):
        captured = {}

        def capture_call(host, model, prompt):
            captured["prompt"] = prompt
            return {
                "category": "Finance/Invoices",
                "subcategory": "",
                "suggested_folder": "Finance/Invoices",
                "confidence": 0.9,
                "reason": "Invoice.",
            }

        with patch("classifier._call_ollama", side_effect=capture_call):
            classify_document(INVOICE_EXTRACTION, TAXONOMY, CONFIG)

        for category in TAXONOMY:
            assert category in captured["prompt"]
