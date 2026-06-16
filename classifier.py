"""Document classification using Ollama (local LLM)."""
import json
import logging
import re
import time
from typing import Optional

import requests
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def classify_document(extraction: dict, taxonomy: list, config: dict) -> dict:
    """Classify a single document. Returns classification dict with a 'status' field."""
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    model = config.get("ollama_model", "phi3:mini")
    threshold = float(config.get("confidence_threshold", 0.6))

    taxonomy_str = "\n".join(f"  - {cat}" for cat in taxonomy)
    prompt = (
        "Given this document snippet, classify it.\n\n"
        f"Available categories:\n{taxonomy_str}\n\n"
        f"Filename: {extraction['filename']}\n"
        "Content preview:\n"
        f"{extraction['text_snippet']}\n\n"
        "Respond with JSON only, no explanation:\n"
        "{\n"
        '  "category": "<one of the categories listed above>",\n'
        '  "subcategory": "<optional, freeform>",\n'
        '  "suggested_folder": "<relative path like Finance/Invoices/2024>",\n'
        '  "confidence": <float 0.0 to 1.0>,\n'
        '  "reason": "<one sentence>"\n'
        "}"
    )

    result = _call_ollama(ollama_host, model, prompt)

    if result is None:
        return {
            "category": "needs_review",
            "subcategory": "",
            "suggested_folder": "Unsorted",
            "confidence": 0.0,
            "reason": "LLM call failed after retry",
            "status": "failed",
        }

    if float(result.get("confidence", 0.0)) < threshold:
        result["status"] = "needs_review"
        result["category"] = "needs_review"
    else:
        result["status"] = "classified"

    return result


def _call_ollama(host: str, model: str, prompt: str) -> Optional[dict]:
    """Call Ollama API; retry once on JSON parse failure."""
    url = f"{host.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}

    for attempt in range(2):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            text = response.json().get("response", "")
            parsed = _parse_json_response(text)
            if parsed is not None:
                return parsed
            if attempt == 0:
                logger.warning("JSON parse failed on attempt 1, retrying...")
                time.sleep(1)
            else:
                logger.error("JSON parse failed after retry — marking document as failed")
                return None
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            return None  # don't retry on HTTP errors

    return None


def _parse_json_response(text: str) -> Optional[dict]:
    # Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Extract JSON object from surrounding text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.debug("Could not parse JSON from response: %.200s", text)
    return None
