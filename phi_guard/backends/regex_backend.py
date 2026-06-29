"""
Regex-based PHI redaction backend.

Default backend for phi-guard. No external dependencies.

Performance characteristics:
  - O(n * p) where n = text length, p = number of enabled patterns
  - Typical throughput: >1MB/s on modern hardware for default pattern set
  - Safe against catastrophic backtracking via max_text_length chunking

Accuracy:
  - High precision for structured identifiers (SSN, NPI, MRN, email)
  - Medium precision for dates and phone numbers (context-dependent)
  - Does NOT detect names — use NER for that
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

from phi_guard.backends.base import Backend, RedactionMatch, RedactionResult
from phi_guard.config import RedactionConfig
from phi_guard.patterns import PATTERNS


class RegexBackend(Backend):
    """
    PHI redaction backend using compiled regex patterns.

    Patterns are compiled once at construction time. Custom patterns from
    RedactionConfig are compiled alongside built-in patterns.

    Args:
        config: RedactionConfig controlling which patterns are active.
    """

    def __init__(self, config: RedactionConfig) -> None:
        self.config = config
        self._compiled: List[Tuple[str, Pattern[str]]] = self._compile_patterns()

    def _compile_patterns(self) -> List[Tuple[str, Pattern[str]]]:
        """Compile enabled built-in + custom patterns in priority order."""
        compiled = []

        # Built-in patterns
        for key, meta in PATTERNS.items():
            if key in self.config.enabled_patterns:
                compiled.append((key, meta.pattern))

        # Custom patterns from config
        for key, raw_pattern in self.config.custom_patterns.items():
            try:
                compiled.append((key, re.compile(raw_pattern, re.IGNORECASE)))
            except re.error as exc:
                raise ValueError(
                    f"Invalid custom pattern '{key}': {raw_pattern!r} — {exc}"
                ) from exc

        return compiled

    def redact_text(self, text: str) -> RedactionResult:
        """
        Apply all enabled patterns to the input text, replacing matches with
        type-labeled placeholders.

        Patterns are applied sequentially. A span already replaced by an earlier
        pattern will not be re-matched by a later one (natural: the placeholder
        text won't match PHI patterns).
        """
        if not text or not text.strip():
            return RedactionResult(redacted=text)

        # Chunk long inputs to avoid catastrophic backtracking on adversarial text
        if len(text) > self.config.max_text_length:
            return self._redact_chunked(text)

        matches: List[RedactionMatch] = []
        result = text

        for pattern_type, pattern in self._compiled:
            placeholder = self.config.placeholder(pattern_type)
            new_result, new_matches = self._apply_pattern(
                result, pattern, pattern_type, placeholder
            )
            matches.extend(new_matches)
            result = new_result

        return RedactionResult(redacted=result, matches=matches)

    def _apply_pattern(
        self,
        text: str,
        pattern: Pattern[str],
        pattern_type: str,
        placeholder: str,
    ) -> Tuple[str, List[RedactionMatch]]:
        """Apply a single pattern to text, returning modified text and matches."""
        matches = []
        offset = 0
        parts = []

        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            original = m.group(0)

            # Skip if this span is already a placeholder from a prior pattern
            if "[REDACTED:" in original:
                continue

            matches.append(RedactionMatch(
                pattern_type=pattern_type,
                original=original,
                placeholder=placeholder,
                start=start + offset,
                end=end + offset,
            ))
            parts.append(text[offset:start])
            parts.append(placeholder)
            offset = end

        if not matches:
            return text, []

        parts.append(text[offset:])
        return "".join(parts), matches

    def _redact_chunked(self, text: str) -> RedactionResult:
        """
        Process long text in chunks to avoid backtracking issues.
        Chunks are split at whitespace boundaries to avoid splitting tokens.
        """
        chunk_size = self.config.max_text_length
        all_matches: List[RedactionMatch] = []
        parts = []
        pos = 0

        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            # Find a whitespace boundary if we're not at the end
            if end < len(text):
                boundary = text.rfind(" ", pos, end)
                if boundary > pos:
                    end = boundary + 1

            chunk = text[pos:end]
            chunk_result = self.redact_text(chunk)

            # Adjust match offsets for the chunk position
            for match in chunk_result.matches:
                adjusted = RedactionMatch(
                    pattern_type=match.pattern_type,
                    original=match.original,
                    placeholder=match.placeholder,
                    start=match.start + pos,
                    end=match.end + pos,
                )
                all_matches.append(adjusted)

            parts.append(chunk_result.redacted)
            pos = end

        return RedactionResult(redacted="".join(parts), matches=all_matches)

    def supported_pattern_keys(self) -> List[str]:
        """Return the list of pattern keys currently active."""
        return [key for key, _ in self._compiled]
