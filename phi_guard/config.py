from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set

_DEFAULT_ENABLED: Set[str] = {
    "ssn", "npi", "mrn", "insurance_id", "account_number",
    "phone", "fax", "email", "date", "age_over_89", "ip_address",
}

@dataclass
class RedactionConfig:
    enabled_patterns: Set[str] = field(default_factory=lambda: set(_DEFAULT_ENABLED))
    custom_patterns: Dict[str, str] = field(default_factory=dict)
    placeholder_template: str = "[REDACTED:{type}]"
    skip_fields: Set[str] = field(default_factory=lambda: {
        "model", "temperature", "max_tokens", "top_p", "stream",
        "id", "created", "object", "usage", "finish_reason", "index",
    })
    redact_keys: bool = False
    max_text_length: int = 100_000

    def placeholder(self, pattern_type: str) -> str:
        return self.placeholder_template.format(type=pattern_type.upper())

    def with_pattern(self, key: str) -> "RedactionConfig":
        return RedactionConfig(
            enabled_patterns=self.enabled_patterns | {key},
            custom_patterns=self.custom_patterns,
            placeholder_template=self.placeholder_template,
            skip_fields=self.skip_fields,
            redact_keys=self.redact_keys,
            max_text_length=self.max_text_length,
        )

    def without_pattern(self, key: str) -> "RedactionConfig":
        return RedactionConfig(
            enabled_patterns=self.enabled_patterns - {key},
            custom_patterns=self.custom_patterns,
            placeholder_template=self.placeholder_template,
            skip_fields=self.skip_fields,
            redact_keys=self.redact_keys,
            max_text_length=self.max_text_length,
        )

    @classmethod
    def strict(cls) -> "RedactionConfig":
        from phi_guard.patterns import get_all_patterns
        return cls(enabled_patterns=set(get_all_patterns().keys()))

    @classmethod
    def minimal(cls) -> "RedactionConfig":
        return cls(enabled_patterns={"ssn", "npi", "email", "mrn"})