"""
Tests for phi_guard.adapters.json_logger.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest
from phi_guard import PHIRedactor
from phi_guard.adapters.json_logger import (
    PHIRedactionFilter,
    PHIAwareJSONFormatter,
    build_phi_safe_logger,
)


def make_handler_with_filter(redactor: PHIRedactor, stream: StringIO) -> logging.StreamHandler:
    handler = logging.StreamHandler(stream)
    handler.addFilter(PHIRedactionFilter(redactor))
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


class TestPHIRedactionFilter:
    def setup_method(self):
        self.redactor = PHIRedactor()
        self.stream = StringIO()
        self.logger = logging.getLogger("test_filter")
        self.logger.handlers = []
        self.logger.addHandler(make_handler_with_filter(self.redactor, self.stream))
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def test_ssn_redacted_from_message(self):
        self.logger.info("Patient SSN: 123-45-6789")
        output = self.stream.getvalue()
        assert "123-45-6789" not in output
        assert "[REDACTED:SSN]" in output

    def test_clean_message_passes_through(self):
        self.logger.info("Appointment scheduled for next Tuesday")
        output = self.stream.getvalue()
        assert "Appointment scheduled" in output

    def test_phi_in_args_redacted(self):
        self.logger.info("SSN is %s", "123-45-6789")
        output = self.stream.getvalue()
        assert "123-45-6789" not in output

    def test_format_args_dict_redacted(self):
        self.logger.info("Contact: %(email)s", {"email": "p@clinic.org"})
        output = self.stream.getvalue()
        assert "p@clinic.org" not in output


class TestPHIAwareJSONFormatter:
    def setup_method(self):
        self.redactor = PHIRedactor()
        self.stream = StringIO()
        self.logger = logging.getLogger("test_json")
        self.logger.handlers = []
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(PHIAwareJSONFormatter(self.redactor))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def _parse_output(self) -> dict:
        line = self.stream.getvalue().strip()
        return json.loads(line)

    def test_output_is_valid_json(self):
        self.logger.info("Test message")
        parsed = self._parse_output()
        assert isinstance(parsed, dict)

    def test_json_contains_standard_fields(self):
        self.logger.info("Test message")
        parsed = self._parse_output()
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "message" in parsed
        assert "logger" in parsed

    def test_phi_redacted_in_json_message(self):
        self.logger.info("SSN: 123-45-6789")
        parsed = self._parse_output()
        assert "123-45-6789" not in parsed["message"]
        assert "[REDACTED:SSN]" in parsed["message"]

    def test_extra_fields_included_and_redacted(self):
        self.logger.info("note", extra={"patient_info": "DOB: 03/15/1982"})
        parsed = self._parse_output()
        assert "patient_info" in parsed
        assert "03/15/1982" not in parsed["patient_info"]

    def test_extra_static_fields(self):
        formatter = PHIAwareJSONFormatter(
            self.redactor,
            extra_fields={"service": "triage-api", "env": "staging"},
        )
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(formatter)
        logger = logging.getLogger("test_static")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.info("hello")
        line = self.stream.getvalue().strip().split("\n")[-1]
        parsed = json.loads(line)
        assert parsed["service"] == "triage-api"
        assert parsed["env"] == "staging"


class TestBuildPHISafeLogger:
    def test_returns_logger(self):
        logger = build_phi_safe_logger("test_factory_logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        logger = build_phi_safe_logger("test_factory_logger_2")
        assert len(logger.handlers) > 0
