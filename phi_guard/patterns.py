"""
PHI pattern definitions for HIPAA Safe Harbor de-identification.

Reference: 45 CFR §164.514(b)(2) — 18 Safe Harbor identifiers.

IMPORTANT LIMITATIONS:
  1. Names (identifier #1) cannot be reliably detected with regex.
     They require Named Entity Recognition (NER). See README.
  2. Geographic data smaller than state (identifier #3) requires
     contextual parsing beyond what regex provides.
  3. All patterns here produce some false positives. Risk level is
     documented per pattern. Tune disabled_by_default accordingly.
  4. This is a developer tool for catching careless mistakes early.
     It is not a HIPAA compliance product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Pattern


@dataclass(frozen=True)
class PHIPatternMeta:
    """Metadata for a PHI pattern."""
    hipaa_identifier: str          # Which Safe Harbor identifier this covers
    false_positive_risk: str       # "low" | "medium" | "high"
    enabled_by_default: bool
    notes: str
    pattern: Pattern[str]


# -----------------------------------------------------------------------------
# Pattern definitions
# Each key becomes the identifier used in [REDACTED:<key>] placeholders
# and in RedactionResult.matches.
# -----------------------------------------------------------------------------

PATTERNS: Dict[str, PHIPatternMeta] = {

    # ── Structured identifiers (low false-positive risk) ─────────────────────

    "ssn": PHIPatternMeta(
        hipaa_identifier="Social Security Number",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="Matches 9-digit SSNs with optional dashes/spaces. Excludes invalid prefixes.",
        pattern=re.compile(
            r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"
        ),
    ),

    "npi": PHIPatternMeta(
        hipaa_identifier="License/Certificate Number",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="National Provider Identifier — 10 digits prefixed by NPI keyword.",
        pattern=re.compile(
            r"\bNPI[:\s#]*(\d{10})\b",
            re.IGNORECASE,
        ),
    ),

    "mrn": PHIPatternMeta(
        hipaa_identifier="Medical Record Number",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="Keyword-anchored. Catches 'MRN: 1234567', 'Medical Record #ABC123'. "
              "Does not catch bare numeric IDs — that would catch everything.",
        pattern=re.compile(
            r"\b(?:MRN|Medical\s+Record(?:\s+(?:No|Number|#))?|Patient\s+(?:ID|No|Number))"
            r"[:\s#]*([A-Z0-9]{4,14})\b",
            re.IGNORECASE,
        ),
    ),

    "insurance_id": PHIPatternMeta(
        hipaa_identifier="Health Plan Beneficiary Number",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="Keyword-anchored: Member ID, Policy No, Beneficiary ID. "
              "Allows hyphens in ID values (e.g., POL-789012, BCBS-091827364).",
        pattern=re.compile(
            r"\b(?:Member\s+(?:ID|No|Number)|Policy\s+(?:No|Number)|"
            r"Insurance\s+(?:ID|No)|Beneficiary\s+(?:ID|No))"
            r"[:\s#]*([A-Z0-9][A-Z0-9\-]{4,15})\b",
            re.IGNORECASE,
        ),
    ),

    "account_number": PHIPatternMeta(
        hipaa_identifier="Account Number",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="Keyword-anchored: Account No, Acct #.",
        pattern=re.compile(
            r"\b(?:Account|Acct)[\s.]?(?:No|Number|#)[:\s]*([A-Z0-9]{4,16})\b",
            re.IGNORECASE,
        ),
    ),

    # ── Contact identifiers (medium false-positive risk) ─────────────────────

    "phone": PHIPatternMeta(
        hipaa_identifier="Telephone Number",
        false_positive_risk="medium",
        enabled_by_default=True,
        notes="Covers US formats with/without country code. "
              "May match version numbers like 1.800.123.4567 in some contexts.",
        pattern=re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
    ),

    "fax": PHIPatternMeta(
        hipaa_identifier="Fax Number",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="Same format as phone but keyword-anchored to 'Fax' or 'Facsimile'.",
        pattern=re.compile(
            r"\b(?:Fax|Facsimile)[:\s]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            re.IGNORECASE,
        ),
    ),

    "email": PHIPatternMeta(
        hipaa_identifier="Email Address",
        false_positive_risk="low",
        enabled_by_default=True,
        notes="Standard email format. Occasional false positive on malformed model outputs.",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
    ),

    # ── Dates (medium false-positive risk) ────────────────────────────────────

    "date": PHIPatternMeta(
        hipaa_identifier="Dates (except year)",
        false_positive_risk="medium",
        enabled_by_default=True,
        notes="Catches MM/DD/YYYY and Month DD, YYYY formats. Year-only is NOT caught "
              "(HIPAA Safe Harbor permits retaining year for non-centenarians). "
              "May produce false positives on non-date numeric sequences.",
        pattern=re.compile(
            r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b"
            r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
            r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
            re.IGNORECASE,
        ),
    ),

    "age_over_89": PHIPatternMeta(
        hipaa_identifier="Ages over 89",
        false_positive_risk="medium",
        enabled_by_default=True,
        notes="HIPAA requires aggregation for ages >89. Catches 'aged 95', '102 years old', "
              "'age: 91'. Handles both keyword-before-number and number-before-keyword forms.",
        pattern=re.compile(
            # keyword before number: "aged 95", "age: 91"
            r"\b(?:age[d]?)[:\s]+(?:9[0-9]|[1-9]\d{2,})\b"
            # number before keyword: "102 years old", "90-year-old"
            r"|\b(?:9[0-9]|[1-9]\d{2,})[\s\-]?(?:years?\s+old|yo)\b",
            re.IGNORECASE,
        ),
    ),

    # ── Network identifiers (medium false-positive risk) ─────────────────────

    "ip_address": PHIPatternMeta(
        hipaa_identifier="IP Address",
        false_positive_risk="medium",
        enabled_by_default=True,
        notes="IPv4 only. May match version strings (4.1.2.3) in some contexts.",
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),

    # ── High false-positive risk (disabled by default) ────────────────────────

    "zip_code": PHIPatternMeta(
        hipaa_identifier="Geographic — ZIP Code",
        false_positive_risk="high",
        enabled_by_default=False,
        notes="DISABLED BY DEFAULT. Matches any 5-digit number. Will catch port numbers, "
              "IDs, timestamps, and legitimate non-PHI numerics. Enable only if your logs "
              "are unlikely to contain other 5-digit numbers, or if you accept the noise.",
        pattern=re.compile(
            r"\b\d{5}(?:-\d{4})?\b"
        ),
    ),

    "url": PHIPatternMeta(
        hipaa_identifier="URLs / Web addresses",
        false_positive_risk="high",
        enabled_by_default=False,
        notes="DISABLED BY DEFAULT. URLs are often legitimate in healthcare AI contexts "
              "(FHIR endpoints, reference links). Enable if your LLM outputs should never "
              "contain URLs.",
        pattern=re.compile(
            r"https?://[^\s<>\"{}|\\^`\[\]]+",
            re.IGNORECASE,
        ),
    ),
}


def get_default_patterns() -> Dict[str, PHIPatternMeta]:
    """Return only patterns enabled by default."""
    return {k: v for k, v in PATTERNS.items() if v.enabled_by_default}


def get_all_patterns() -> Dict[str, PHIPatternMeta]:
    """Return all patterns including high-risk disabled ones."""
    return dict(PATTERNS)
