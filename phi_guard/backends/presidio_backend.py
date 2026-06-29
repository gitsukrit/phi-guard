"""
Presidio-based PHI detection backend.

Microsoft Presidio provides NER (Named Entity Recognition) that catches
what regex cannot — primarily PERSON names, but also contextual LOCATION,
DATE, and other entities that appear without keyword anchors.

Requires: pip install phi-guard[presidio]
Then download the spaCy model:
    python -m spacy download en_core_web_lg

Why this solves the names problem:
  The regex baseline catches structured identifiers (SSN, MRN, NPI, email, phone).
  It cannot catch "Janet Morrison" because names have no fixed lexical pattern.
  Presidio uses a transformer-based NER model that understands context:
  "Dr. Patel ordered..." → "Patel" is a PERSON even without surrounding markers.

Trade-offs vs RegexBackend:
  + Catches person names (HIPAA Safe Harbor #1 — the biggest gap in the regex backend)
  + Catches contextual addresses, locations, and dates without keyword anchors
  + Lower false negative rate on clinical narrative text
  - Requires ~500MB spaCy model download on first use
  - 10-50x slower than RegexBackend (depends on model; en_core_web_sm is faster)
  - Adds spaCy and Presidio to your dependency tree
  - Still misses adversarially obfuscated PHI

Recommended usage pattern:
  Use PresidioBackend for high-risk clinical narrative text where names matter.
  Use RegexBackend for high-throughput structured logs where speed matters.
  Compose them: RegexBackend first, PresidioBackend for the residual.

Composing both backends:
    from phi_guard import PHIRedactor, RedactionConfig
    from phi_guard.backends.presidio_backend import PresidioBackend, CompositeBackend
    from phi_guard.backends.regex_backend import RegexBackend

    config = RedactionConfig()
    composite = CompositeBackend([
        RegexBackend(config),       # fast, structured PHI
        PresidioBackend(config),    # NER, catches names
    ])
    redactor = PHIRedactor(backend=composite)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Set

from phi_guard.backends.base import Backend, RedactionMatch, RedactionResult
from phi_guard.config import RedactionConfig


# Mapping from Presidio entity types to phi-guard placeholder types.
# Keys must match what Presidio's AnalyzerEngine returns.
PRESIDIO_TO_PHI_TYPE: Dict[str, str] = {
    "PERSON": "person_name",
    "DATE_TIME": "date",
    "PHONE_NUMBER": "phone",
    "EMAIL_ADDRESS": "email",
    "US_SSN": "ssn",
    "LOCATION": "location",
    "IP_ADDRESS": "ip_address",
    "URL": "url",
    "US_DRIVER_LICENSE": "driver_license",
    "MEDICAL_LICENSE": "medical_license",
    "IBAN_CODE": "account_number",
    "CREDIT_CARD": "credit_card",
    "NRP": "demographic",           # Nationality, religion, political group
    "IN_PAN": "tax_id",
    "IN_AADHAAR": "national_id",
}

# Default set of Presidio entities to analyze for HIPAA coverage.
# Omit CREDIT_CARD and NRP unless your use case requires them.
DEFAULT_PRESIDIO_ENTITIES: Set[str] = {
    "PERSON",
    "DATE_TIME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "LOCATION",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
}


class PresidioBackend(Backend):
    """
    PHI detection backend using Microsoft Presidio + spaCy NER.

    This backend is best used for clinical narrative text where person names
    and contextual identifiers are the primary PHI risk. For high-throughput
    structured logs, prefer RegexBackend or CompositeBackend.

    Args:
        config: RedactionConfig. Note: enabled_patterns is ignored by this backend.
                Presidio decides what to detect based on `entities`.
        entities: Set of Presidio entity type strings to analyze.
                  Defaults to DEFAULT_PRESIDIO_ENTITIES.
        model: spaCy model name. "en_core_web_lg" gives the best accuracy.
               "en_core_web_sm" is faster but less accurate. "en_core_web_trf"
               (transformer-based) gives the highest accuracy at the highest cost.
        score_threshold: Minimum confidence score for a Presidio result to be
                         used. Range 0.0–1.0. Default 0.5. Lower = more recall,
                         more false positives. Raise to 0.7–0.8 for lower FP rate.
        language: Language code for Presidio analysis. Default "en".

    Setup:
        pip install presidio-analyzer presidio-anonymizer spacy
        python -m spacy download en_core_web_lg
    """

    def __init__(
        self,
        config: RedactionConfig,
        entities: Optional[Set[str]] = None,
        model: str = "en_core_web_lg",
        score_threshold: float = 0.5,
        language: str = "en",
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PresidioBackend requires the Presidio packages and a spaCy model.\n"
                "Install with:\n"
                "  pip install presidio-analyzer presidio-anonymizer spacy\n"
                "  python -m spacy download en_core_web_lg\n"
                "Or: pip install 'phi-guard[presidio]'"
            ) from exc

        self.config = config
        self.entities = list(entities or DEFAULT_PRESIDIO_ENTITIES)
        self.score_threshold = score_threshold
        self.language = language

        # Build the NLP engine with the specified spaCy model
        try:
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": model}],
            })
            nlp_engine = provider.create_engine()
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' not found. Download it with:\n"
                f"  python -m spacy download {model}"
            ) from exc

    def redact_text(self, text: str) -> RedactionResult:
        """
        Analyze text with Presidio NER and redact all detected PHI entities.

        Results are sorted by position (right to left) so that replacements
        don't shift offsets of earlier matches.
        """
        if not text or not text.strip():
            return RedactionResult(redacted=text)

        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
        )

        if not results:
            return RedactionResult(redacted=text)

        # Sort right-to-left to preserve offsets during replacement
        results_sorted = sorted(results, key=lambda r: r.start, reverse=True)

        redacted_text = text
        matches: List[RedactionMatch] = []

        for result in results_sorted:
            entity_type = result.entity_type
            phi_type = PRESIDIO_TO_PHI_TYPE.get(entity_type, entity_type.lower())
            placeholder = self.config.placeholder(phi_type)
            original = redacted_text[result.start:result.end]

            matches.append(RedactionMatch(
                pattern_type=phi_type,
                original=original,
                placeholder=placeholder,
                start=result.start,
                end=result.end,
            ))

            redacted_text = (
                redacted_text[:result.start]
                + placeholder
                + redacted_text[result.end:]
            )

        # matches were collected right-to-left; reverse for logical order
        matches.reverse()
        return RedactionResult(redacted=redacted_text, matches=matches)

    def supported_entities(self) -> List[str]:
        """Return the list of Presidio entity types currently active."""
        return list(self.entities)

    def supports_streaming(self) -> bool:
        # Presidio requires the full sentence for NER context.
        # Per-chunk streaming degrades accuracy significantly.
        return False


class CompositeBackend(Backend):
    """
    Run multiple backends in sequence, accumulating all matches.

    The output of each backend is fed into the next. This ensures that
    patterns caught by RegexBackend are not re-processed by PresidioBackend
    (the [REDACTED:SSN] placeholder won't trigger the NER model as a name).

    Typical composition:
        composite = CompositeBackend([
            RegexBackend(config),    # fast structured PHI first
            PresidioBackend(config), # NER for names and contextual PHI
        ])

    Args:
        backends: List of Backend instances, applied in order.
    """

    def __init__(self, backends: List[Backend]) -> None:
        if not backends:
            raise ValueError("CompositeBackend requires at least one backend.")
        self.backends = backends

    def redact_text(self, text: str) -> RedactionResult:
        """Apply each backend in sequence to the text."""
        current_text = text
        all_matches: List[RedactionMatch] = []

        for backend in self.backends:
            result = backend.redact_text(current_text)
            all_matches.extend(result.matches)
            current_text = result.redacted

        return RedactionResult(redacted=current_text, matches=all_matches)

    def supports_streaming(self) -> bool:
        return all(b.supports_streaming() for b in self.backends)
