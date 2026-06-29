"""
Core PHI redaction engine.

PHIRedactor is the main entry point for phi-guard. It wraps a backend
(RegexBackend by default) and provides:

  - redact(text)             — redact a plain string
  - redact_dict(data)        — recursively redact all string values in a dict
  - redact_messages(msgs)    — redact an OpenAI-format messages list

Usage:
    from phi_guard import PHIRedactor

    redactor = PHIRedactor()
    result = redactor.redact("Patient DOB is 03/15/1982, SSN 123-45-6789")
    print(result.redacted)
    # → "Patient DOB is [REDACTED:DATE], SSN [REDACTED:SSN]"
    print(result.match_summary)
    # → {'date': 1, 'ssn': 1}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from phi_guard.backends.base import Backend, RedactionResult
from phi_guard.backends.regex_backend import RegexBackend
from phi_guard.config import RedactionConfig


class PHIRedactor:
    """
    PHI-aware redaction engine.

    Args:
        config: RedactionConfig. Defaults to standard configuration covering
                structured HIPAA Safe Harbor identifiers.
        backend: Backend instance. Defaults to RegexBackend(config).
                 Pass a custom backend to use LLM-based detection or
                 your own implementation.

    Examples:
        # Default configuration
        redactor = PHIRedactor()

        # Custom configuration — add ZIP codes, remove dates
        from phi_guard import RedactionConfig
        config = RedactionConfig().with_pattern("zip_code").without_pattern("date")
        redactor = PHIRedactor(config=config)

        # Strict mode — all patterns enabled
        redactor = PHIRedactor(config=RedactionConfig.strict())

        # LLM backend (read the BAA warning in llm_backend.py first)
        from phi_guard.backends.llm_backend import LLMBackend
        backend = LLMBackend(config=RedactionConfig())
        redactor = PHIRedactor(backend=backend)
    """

    def __init__(
        self,
        config: Optional[RedactionConfig] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        self.config = config or RedactionConfig()
        self.backend = backend or RegexBackend(self.config)

    # ── Public API ────────────────────────────────────────────────────────────

    def redact(self, text: str) -> RedactionResult:
        """
        Redact PHI from a plain string.

        Returns a RedactionResult. Use `.redacted` for the safe string.
        Use `.match_summary` for audit counts (safe to log).
        Never log `.matches` — it contains the original matched text.
        """
        if not isinstance(text, str):
            raise TypeError(f"redact() expects str, got {type(text).__name__}")
        return self.backend.redact_text(text)

    def redact_dict(
        self,
        data: Dict[str, Any],
        *,
        depth: int = 0,
        max_depth: int = 10,
    ) -> Dict[str, Any]:
        """
        Recursively redact PHI from all string values in a dict.

        Skips keys listed in config.skip_fields at any depth.
        Handles nested dicts, lists, and scalar values.
        Non-string scalars (int, float, bool, None) are passed through unchanged.

        Args:
            data: Input dict (typically an LLM API request/response payload).
            depth: Current recursion depth (internal use).
            max_depth: Maximum recursion depth to prevent stack overflow on
                       pathological inputs. Default: 10.

        Returns:
            New dict with PHI redacted. Original dict is not modified.
        """
        if depth > max_depth:
            return data  # bail out silently on deeply nested structures

        result = {}
        for key, value in data.items():
            if key in self.config.skip_fields:
                result[key] = value
                continue

            redacted_key = key
            if self.config.redact_keys and isinstance(key, str):
                redacted_key = self.backend.redact_text(key).redacted

            result[redacted_key] = self._redact_value(value, depth=depth, max_depth=max_depth)

        return result

    def redact_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Redact PHI from an OpenAI/Anthropic-format messages list.

        Handles the content field as either a string or a list of content blocks.
        Role and other metadata fields are not modified.

        Args:
            messages: List of message dicts, e.g.:
                [{"role": "user", "content": "My MRN is 12345678"}]

        Returns:
            New list of message dicts with PHI redacted in content.
        """
        redacted = []
        for message in messages:
            if not isinstance(message, dict):
                redacted.append(message)
                continue

            msg_copy = dict(message)
            content = msg_copy.get("content")

            if isinstance(content, str):
                msg_copy["content"] = self.backend.redact_text(content).redacted
            elif isinstance(content, list):
                # Content blocks: [{"type": "text", "text": "..."}, ...]
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block = dict(block)
                        block["text"] = self.backend.redact_text(
                            block.get("text", "")
                        ).redacted
                    new_blocks.append(block)
                msg_copy["content"] = new_blocks

            redacted.append(msg_copy)

        return redacted

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _redact_value(
        self,
        value: Any,
        *,
        depth: int,
        max_depth: int,
    ) -> Any:
        """Dispatch redaction based on value type."""
        if isinstance(value, str):
            return self.backend.redact_text(value).redacted
        elif isinstance(value, dict):
            return self.redact_dict(value, depth=depth + 1, max_depth=max_depth)
        elif isinstance(value, list):
            return [
                self._redact_value(item, depth=depth + 1, max_depth=max_depth)
                for item in value
            ]
        else:
            # int, float, bool, None — pass through unchanged
            return value
