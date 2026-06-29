"""
Tests for phi_guard.patterns.

Tests are organized by pattern type. Each test has:
  - should_match: strings that MUST produce a match
  - should_not_match: strings that MUST NOT produce a match (false positives)

False positive avoidance is as important as true positive detection.
"""

from __future__ import annotations

import re
import pytest
from phi_guard.patterns import PATTERNS


def matches(pattern_key: str, text: str) -> bool:
    """Return True if the pattern finds at least one match in text."""
    return bool(PATTERNS[pattern_key].pattern.search(text))


# ── SSN ──────────────────────────────────────────────────────────────────────

class TestSSN:
    def test_standard_dashes(self):
        assert matches("ssn", "SSN: 123-45-6789")

    def test_no_separator(self):
        assert matches("ssn", "SSN 123456789")

    def test_spaces_separator(self):
        assert matches("ssn", "social security 123 45 6789")

    def test_in_sentence(self):
        # 987-xx-xxxx starts with 9xx (ITIN range — excluded by pattern).
        # Use a valid SSN prefix (not 000, 666, or 9xx).
        assert matches("ssn", "Her SSN is 234-56-7890 per the form.")

    def test_invalid_prefix_000(self):
        assert not matches("ssn", "000-45-6789")

    def test_invalid_prefix_666(self):
        assert not matches("ssn", "666-45-6789")

    def test_invalid_prefix_900(self):
        assert not matches("ssn", "900-45-6789")

    def test_all_zeros_group2(self):
        assert not matches("ssn", "123-00-6789")

    def test_all_zeros_group3(self):
        assert not matches("ssn", "123-45-0000")


# ── MRN ──────────────────────────────────────────────────────────────────────

class TestMRN:
    def test_mrn_colon(self):
        assert matches("mrn", "MRN: 00123456")

    def test_mrn_hash(self):
        assert matches("mrn", "MRN #AB1234")

    def test_medical_record_number(self):
        assert matches("mrn", "Medical Record Number: 78901234")

    def test_patient_id(self):
        assert matches("mrn", "Patient ID: 456789")

    def test_bare_number_no_keyword(self):
        # Bare numbers without keyword context should NOT match
        assert not matches("mrn", "The value is 12345678")

    def test_model_name_not_mrn(self):
        assert not matches("mrn", "Using model gpt-4-turbo")


# ── Dates ─────────────────────────────────────────────────────────────────────

class TestDate:
    def test_slash_format(self):
        assert matches("date", "DOB: 03/15/1982")

    def test_dash_format(self):
        assert matches("date", "admitted 12-01-2020")

    def test_month_name_full(self):
        assert matches("date", "born January 5, 1955")

    def test_month_name_abbreviated(self):
        assert matches("date", "discharge date: Mar 22, 2023")

    def test_year_only_not_matched(self):
        # HIPAA Safe Harbor permits retaining year alone
        assert not matches("date", "admitted in 2021")

    def test_future_century_not_matched(self):
        # Pattern anchors to 19xx and 20xx
        assert not matches("date", "year 2100")


# ── Phone ─────────────────────────────────────────────────────────────────────

class TestPhone:
    def test_dashes(self):
        assert matches("phone", "call 415-555-1234")

    def test_parens(self):
        assert matches("phone", "(415) 555-1234")

    def test_dots(self):
        assert matches("phone", "415.555.1234")

    def test_country_code(self):
        assert matches("phone", "+1 415 555 1234")

    def test_toll_free(self):
        assert matches("phone", "1-800-555-5555")

    def test_short_number_not_matched(self):
        assert not matches("phone", "call ext 4321")


# ── Email ─────────────────────────────────────────────────────────────────────

class TestEmail:
    def test_standard(self):
        assert matches("email", "contact j.smith@hospital.org")

    def test_plus_addressing(self):
        assert matches("email", "notify patient+alerts@clinic.health")

    def test_subdomains(self):
        assert matches("email", "reach us at support@my.health.system.com")

    def test_no_at_sign(self):
        assert not matches("email", "username without at sign")


# ── NPI ──────────────────────────────────────────────────────────────────────

class TestNPI:
    def test_standard(self):
        assert matches("npi", "NPI: 1234567890")

    def test_with_hash(self):
        assert matches("npi", "NPI#1234567890")

    def test_nine_digits_not_matched(self):
        # NPI must be exactly 10 digits
        assert not matches("npi", "NPI: 123456789")

    def test_eleven_digits_not_matched(self):
        assert not matches("npi", "NPI: 12345678901")


# ── Insurance ID ─────────────────────────────────────────────────────────────

class TestInsuranceID:
    def test_member_id(self):
        assert matches("insurance_id", "Member ID: XYZ123456")

    def test_policy_number(self):
        assert matches("insurance_id", "Policy Number: POL-789012")

    def test_beneficiary_id(self):
        assert matches("insurance_id", "Beneficiary ID ABC78901")

    def test_bare_id_not_matched(self):
        assert not matches("insurance_id", "your ID is 123456")


# ── IP Address ────────────────────────────────────────────────────────────────

class TestIPAddress:
    def test_standard_ipv4(self):
        assert matches("ip_address", "source IP: 192.168.1.100")

    def test_in_url_context(self):
        assert matches("ip_address", "server at 10.0.0.1:8080")

    def test_invalid_octet(self):
        # 256 is not a valid octet
        assert not matches("ip_address", "256.0.0.1")


# ── Fax ──────────────────────────────────────────────────────────────────────

class TestFax:
    def test_fax_keyword(self):
        assert matches("fax", "Fax: 415-555-9876")

    def test_facsimile_keyword(self):
        assert matches("fax", "Facsimile: (800) 555-1234")

    def test_phone_without_keyword_not_matched(self):
        # Fax pattern requires keyword — plain phone numbers are caught by "phone"
        # This test verifies the fax pattern doesn't double-catch plain phones
        # (they're caught by the phone pattern instead)
        result = PATTERNS["fax"].pattern.search("415-555-9876")
        assert result is None


# ── Age Over 89 ───────────────────────────────────────────────────────────────

class TestAgeOver89:
    def test_age_90(self):
        assert matches("age_over_89", "patient aged 90")

    def test_age_102(self):
        # "102 years old" — number before keyword form
        assert matches("age_over_89", "102 years old")

    def test_age_102_keyword_first(self):
        assert matches("age_over_89", "aged 102")

    def test_age_89_not_matched(self):
        # HIPAA only requires aggregation for >89
        assert not matches("age_over_89", "patient aged 89")

    def test_age_45_not_matched(self):
        assert not matches("age_over_89", "45 years old")


# ── ZIP Code (disabled by default, high FP risk) ─────────────────────────────

class TestZipCode:
    def test_standard_zip(self):
        assert matches("zip_code", "ZIP: 94107")

    def test_zip_plus_four(self):
        assert matches("zip_code", "94107-1234")

    def test_five_digit_number_also_matches(self):
        # This is why ZIP is disabled by default — high FP rate.
        # Any 5-digit number matches, including port 65535.
        assert matches("zip_code", "port 65535 is open")

    def test_is_disabled_by_default(self):
        from phi_guard.config import RedactionConfig
        config = RedactionConfig()
        assert "zip_code" not in config.enabled_patterns
