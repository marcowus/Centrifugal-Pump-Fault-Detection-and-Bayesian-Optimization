"""Client abstraction for calling LLM services with retries and structured parsing."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional

import requests

logger = logging.getLogger(__name__)

try:  # The OpenAI SDK is used for SiliconFlow compatibility.
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for SiliconFlow usage
    OpenAI = None  # type: ignore[assignment]


class FaultLabel(str, Enum):
    NORMAL = "normal"
    BEARING_DAMAGE = "bearing_damage"
    MISALIGNMENT = "misalignment"
    CAVITATION = "cavitation"
    IMPELLER_DAMAGE = "impeller_damage"
    LOOSENESS = "looseness"
    UNKNOWN = "unknown"

    @classmethod
    def from_text(cls, text: str) -> "FaultLabel":
        normalized = (text or "").strip().lower()
        aliases = {
            "ok": cls.NORMAL,
            "healthy": cls.NORMAL,
            "bearing": cls.BEARING_DAMAGE,
            "bearing fault": cls.BEARING_DAMAGE,
            "bearing failure": cls.BEARING_DAMAGE,
            "misaligned": cls.MISALIGNMENT,
            "shaft misalignment": cls.MISALIGNMENT,
            "cavitation": cls.CAVITATION,
            "impeller": cls.IMPELLER_DAMAGE,
            "impeller fault": cls.IMPELLER_DAMAGE,
            "loose": cls.LOOSENESS,
        }
        return aliases.get(normalized, cls._value2member_map_.get(normalized, cls.UNKNOWN))


@dataclass
class DiagnosisResult:
    diagnosis: FaultLabel
    confidence: float
    evidence: str
    inspection_recommendations: str
    maintenance_plan: str
    arbitration_note: Optional[str] = None

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis.value,
            "confidence": float(self.confidence),
            "evidence": self.evidence,
            "inspection_recommendations": self.inspection_recommendations,
            "maintenance_plan": self.maintenance_plan,
            "arbitration_note": self.arbitration_note,
        }


class RateLimiter:
    """Simple thread-safe rate limiter."""

    def __init__(self, max_calls_per_minute: int) -> None:
        self._lock = threading.Lock()
        self._interval = 60.0 / max(1, max_calls_per_minute)
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_call = time.monotonic()


class LLMClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        max_calls_per_minute: int = 20,
    ) -> None:
        provider = (os.getenv("LLM_PROVIDER") or "siliconflow").lower()
        self.provider = provider

        if provider == "siliconflow":
            default_base = "https://api.siliconflow.cn/v1"
            default_model = "deepseek-ai/DeepSeek-V3"
            env_api_key = os.getenv("SILICON_FLOW_API_KEY")
        else:
            default_base = "https://api.openai.com/v1"
            default_model = "gpt-3.5-turbo"
            env_api_key = None

        resolved_base = base_url or os.getenv("LLM_BASE_URL") or default_base
        self.base_url = resolved_base.rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", default_model)
        self.api_key = api_key or os.getenv("LLM_API_KEY") or env_api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(max_calls_per_minute)
        offline_flag = os.getenv("LLM_OFFLINE_MODE", "auto").lower()
        if offline_flag in {"1", "true", "yes"}:
            self.offline_mode = True
        elif offline_flag in {"0", "false", "no"}:
            self.offline_mode = False
        else:
            # Auto-enable offline mode when no API key is configured.
            self.offline_mode = self.api_key is None

        self._openai_client = None
        if self.provider == "siliconflow" and not self.offline_mode:
            if OpenAI is None:
                raise ImportError(
                    "SiliconFlow provider requires the openai package. "
                    "Install it or enable offline mode."
                )
            self._openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _headers(self) -> Dict[str, str]:
        if not self.api_key and not self.offline_mode:
            raise RuntimeError("LLM API key is required unless offline mode is enabled.")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_payload(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an industrial fault diagnosis assistant. Always respond in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        return payload

    def _call_remote(self, prompt: str) -> str:
        payload = self._post_payload(prompt)
        if self.provider == "siliconflow" and self._openai_client is not None:
            return self._call_siliconflow(prompt)
        for attempt in range(1, self.max_retries + 1):
            try:
                self.rate_limiter.wait()
                response = requests.post(
                    self._chat_endpoint(),
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(f"Server error {response.status_code}: {response.text}")
                response.raise_for_status()
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not message:
                    raise ValueError("Empty response from LLM service")
                return message
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call failed on attempt %s/%s: %s", attempt, self.max_retries, exc)
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("LLM call failed after retries")

    def _chat_endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _call_siliconflow(self, prompt: str) -> str:
        assert self._openai_client is not None  # For type checkers
        for attempt in range(1, self.max_retries + 1):
            try:
                self.rate_limiter.wait()
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an industrial fault diagnosis assistant. Always respond in JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                message = response.choices[0].message.content if response.choices else ""
                if not message:
                    raise ValueError("Empty response from SiliconFlow service")
                return message
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SiliconFlow call failed on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError("SiliconFlow call failed after retries")

    def _offline_stub(self, prompt: str) -> str:
        # Simple heuristic stub: return a deterministic JSON object.
        logger.info("Using offline mode for LLM response.")
        fallback = {
            "diagnosis": "bearing_damage" if "bearing" in prompt.lower() else "misalignment",
            "confidence": 0.55,
            "evidence": "Heuristic inference based on anomaly magnitude.",
            "inspection_recommendations": "Inspect bearings, alignment, and lubrication.",
            "maintenance_plan": "Schedule targeted maintenance within 48 hours.",
        }
        return json.dumps(fallback, ensure_ascii=False)

    def request_secondary_diagnosis(
        self,
        prompt: str,
    ) -> DiagnosisResult:
        if self.offline_mode:
            raw_response = self._offline_stub(prompt)
        else:
            try:
                raw_response = self._call_remote(prompt)
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM request failed after retries: %s", exc)
                raise

        return self._parse_response(raw_response)

    def _parse_response(self, content: str) -> DiagnosisResult:
        structured: Dict[str, Any]
        try:
            structured = json.loads(content)
        except json.JSONDecodeError:
            logger.debug("Attempting to parse Markdown structured response.")
            structured = self._parse_markdown(content)

        diagnosis_label = FaultLabel.from_text(str(structured.get("diagnosis", "")))
        confidence = float(structured.get("confidence", 0.0))
        evidence = str(structured.get("evidence", ""))
        inspection = str(structured.get("inspection_recommendations", structured.get("inspection", "")))
        maintenance = str(structured.get("maintenance_plan", structured.get("maintenance", "")))

        return DiagnosisResult(
            diagnosis=diagnosis_label,
            confidence=confidence,
            evidence=evidence,
            inspection_recommendations=inspection,
            maintenance_plan=maintenance,
        )

    @staticmethod
    def _parse_markdown(content: str) -> Dict[str, Any]:
        structured: Dict[str, Any] = {}
        current_key: Optional[str] = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer, current_key
            if current_key is not None:
                structured[current_key] = "\n".join(buffer).strip()
            buffer = []
            current_key = None

        for line in content.splitlines():
            header = line.strip().lower()
            if header.startswith("## "):
                flush()
                title = header[3:].strip().replace(" ", "_")
                current_key = title
            else:
                buffer.append(line)
        flush()
        return structured


__all__ = [
    "LLMClient",
    "FaultLabel",
    "DiagnosisResult",
]
