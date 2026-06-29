from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class RedactionMatch:
    pattern_type: str
    original: str
    placeholder: str
    start: int
    end: int

@dataclass
class RedactionResult:
    redacted: str
    matches: List[RedactionMatch] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return len(self.matches) > 0

    @property
    def match_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for m in self.matches:
            summary[m.pattern_type] = summary.get(m.pattern_type, 0) + 1
        return summary

    def __repr__(self) -> str:
        if not self.was_modified:
            return "RedactionResult(no PHI detected)"
        counts = ", ".join(f"{k}={v}" for k, v in self.match_summary.items())
        return f"RedactionResult(redacted: {counts})"

class Backend(ABC):
    @abstractmethod
    def redact_text(self, text: str) -> RedactionResult: ...

    def supports_streaming(self) -> bool:
        return False