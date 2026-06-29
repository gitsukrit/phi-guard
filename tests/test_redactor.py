"""
Tests for PHIRedactor, RedactionConfig, and RedactionResult.
"""

from __future__ import annotations

import pytest
from phi_guard import PHIRedactor, RedactionConfig, RedactionResult


# ── Basic redaction ───────────────────────────────────────────────────────────

class TestPHIRedactorBasic:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def test_ssn_in_text(self):
        result = self.redactor.redact("SSN: 123-45-6789")
        assert "[REDACTED:SSN]" in result.redacted
        assert "123-45-6789" not in result.redacted

    def test_email_in_text(self):
        result = self.redactor.redact("email me at jane.doe@hospital.org")
        assert "[REDACTED:EMAIL]" in result.redacted
        assert "jane.doe@hospital.org" not in result.redacted

    def test_phone_in_text(self):
        result = self.redactor.redact("call (415) 555-1234")
        assert "[REDACTED:PHONE]" in result.redacted

    def test_date_in_text(self):
        result = self.redactor.redact("DOB: 03/15/1982")
        assert "[REDACTED:DATE]" in result.redacted

    def test_multiple_phi_types(self):
        text = "Patient MRN: 00123456, DOB: 07/04/1955, email: p@clinic.org"
        result = self.redactor.redact(text)
        assert "00123456" not in result.redacted
        assert "07/04/1955" not in result.redacted
        assert "p@clinic.org" not in result.redacted
        assert result.was_modified

    def test_clean_text_not_modified(self):
        text = "The patient had a follow-up appointment scheduled."
        result = self.redactor.redact(text)
        assert result.redacted == text
        assert not result.was_modified

    def test_empty_string(self):
        result = self.redactor.redact("")
        assert result.redacted == ""
        assert not result.was_modified

    def test_whitespace_only(self):
        result = self.redactor.redact("   ")
        assert not result.was_modified

    def test_type_error_on_non_string(self):
        with pytest.raises(TypeError):
            self.redactor.redact(12345)


# ── RedactionResult ───────────────────────────────────────────────────────────

class TestRedactionResult:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def test_match_summary_safe_to_log(self):
        # Use valid SSN prefixes (not 9xx range which is excluded by pattern)
        result = self.redactor.redact("SSN: 123-45-6789, SSN: 234-56-7890")
        summary = result.match_summary
        assert "ssn" in summary
        assert summary["ssn"] == 2

    def test_matches_contain_original(self):
        result = self.redactor.redact("SSN: 123-45-6789")
        assert len(result.matches) >= 1
        assert any(m.original == "123-45-6789" for m in result.matches)

    def test_repr_no_phi(self):
        result = self.redactor.redact("no PHI here")
        assert "no PHI" in repr(result)

    def test_repr_with_phi(self):
        result = self.redactor.redact("SSN: 123-45-6789")
        assert "ssn=1" in repr(result)


# ── Dict redaction ────────────────────────────────────────────────────────────

class TestRedactDict:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def test_flat_dict(self):
        data = {"message": "Patient SSN: 123-45-6789", "level": "info"}
        redacted = self.redactor.redact_dict(data)
        assert "123-45-6789" not in redacted["message"]
        assert redacted["level"] == "info"

    def test_nested_dict(self):
        data = {
            "response": {
                "content": "Your DOB is 03/15/1982",
                "model": "claude-sonnet-4-6",
            }
        }
        redacted = self.redactor.redact_dict(data)
        assert "03/15/1982" not in redacted["response"]["content"]
        assert redacted["response"]["model"] == "claude-sonnet-4-6"

    def test_list_values_redacted(self):
        data = {"items": ["SSN: 123-45-6789", "no PHI here"]}
        redacted = self.redactor.redact_dict(data)
        assert "123-45-6789" not in redacted["items"][0]
        assert redacted["items"][1] == "no PHI here"

    def test_skip_fields_not_redacted(self):
        # "model" is in the default skip_fields set
        data = {"model": "claude-sonnet-4-6", "content": "SSN 123-45-6789"}
        redacted = self.redactor.redact_dict(data)
        assert redacted["model"] == "claude-sonnet-4-6"

    def test_integer_values_pass_through(self):
        data = {"tokens": 1234, "message": "hello"}
        redacted = self.redactor.redact_dict(data)
        assert redacted["tokens"] == 1234

    def test_none_values_pass_through(self):
        data = {"optional": None, "message": "hello"}
        redacted = self.redactor.redact_dict(data)
        assert redacted["optional"] is None

    def test_original_not_mutated(self):
        data = {"content": "SSN: 123-45-6789"}
        _ = self.redactor.redact_dict(data)
        assert data["content"] == "SSN: 123-45-6789"  # original unchanged


# ── Messages redaction ────────────────────────────────────────────────────────

class TestRedactMessages:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def test_user_message_redacted(self):
        messages = [{"role": "user", "content": "My SSN is 123-45-6789"}]
        redacted = self.redactor.redact_messages(messages)
        assert "123-45-6789" not in redacted[0]["content"]

    def test_role_not_changed(self):
        messages = [{"role": "assistant", "content": "DOB: 03/15/1982"}]
        redacted = self.redactor.redact_messages(messages)
        assert redacted[0]["role"] == "assistant"

    def test_content_block_format(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "SSN: 123-45-6789"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]
        }]
        redacted = self.redactor.redact_messages(messages)
        assert "123-45-6789" not in redacted[0]["content"][0]["text"]
        # Image blocks should pass through
        assert redacted[0]["content"][1]["type"] == "image_url"

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": "My DOB is 01/01/1960"},
            {"role": "assistant", "content": "Thank you for sharing that."},
            {"role": "user", "content": "My email is p@clinic.org"},
        ]
        redacted = self.redactor.redact_messages(messages)
        assert "01/01/1960" not in redacted[0]["content"]
        assert redacted[1]["content"] == "Thank you for sharing that."
        assert "p@clinic.org" not in redacted[2]["content"]


# ── Configuration ─────────────────────────────────────────────────────────────

class TestRedactionConfig:
    def test_default_does_not_include_zip(self):
        config = RedactionConfig()
        assert "zip_code" not in config.enabled_patterns

    def test_with_pattern_adds_zip(self):
        config = RedactionConfig().with_pattern("zip_code")
        assert "zip_code" in config.enabled_patterns

    def test_without_pattern_removes_ssn(self):
        config = RedactionConfig().without_pattern("ssn")
        assert "ssn" not in config.enabled_patterns

    def test_strict_includes_zip_and_url(self):
        config = RedactionConfig.strict()
        assert "zip_code" in config.enabled_patterns
        assert "url" in config.enabled_patterns

    def test_minimal_config(self):
        config = RedactionConfig.minimal()
        redactor = PHIRedactor(config=config)
        # SSN should still be caught
        result = redactor.redact("SSN: 123-45-6789")
        assert result.was_modified

    def test_custom_pattern(self):
        config = RedactionConfig(
            custom_patterns={"patient_portal_id": r"PP-\d{8}"}
        )
        redactor = PHIRedactor(config=config)
        result = redactor.redact("Portal login: PP-12345678")
        assert "PP-12345678" not in result.redacted
        assert "[REDACTED:PATIENT_PORTAL_ID]" in result.redacted

    def test_custom_placeholder_template(self):
        config = RedactionConfig(placeholder_template="***{type}***")
        redactor = PHIRedactor(config=config)
        result = redactor.redact("SSN: 123-45-6789")
        assert "***SSN***" in result.redacted

    def test_invalid_custom_pattern_raises(self):
        from phi_guard.backends.regex_backend import RegexBackend
        config = RedactionConfig(custom_patterns={"bad": r"[unclosed"})
        with pytest.raises(ValueError, match="Invalid custom pattern"):
            RegexBackend(config)
