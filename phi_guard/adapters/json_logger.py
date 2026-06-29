"""
Standard Python logging adapter for phi-guard.

Provides a logging.Filter subclass that intercepts log records and redacts
PHI from any string fields before they are emitted to handlers. Works with
any handler: StreamHandler, FileHandler, RotatingFileHandler, etc.

Also provides a structured JSON formatter that emits log records as JSON
objects (compatible with Datadog, Cloud Logging, Splunk, etc.) with PHI
already redacted.

Usage — basic filter:
    import logging
    from phi_guard import PHIRedactor
    from phi_guard.adapters.json_logger import PHIRedactionFilter

    redactor = PHIRedactor()
    filter_ = PHIRedactionFilter(redactor)

    handler = logging.StreamHandler()
    handler.addFilter(filter_)

    logger = logging.getLogger("my_app")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Patient DOB: 03/15/1982, MRN: 00123456")
    # → "Patient DOB: [REDACTED:DATE], MRN: [REDACTED:MRN]"

Usage — structured JSON with redaction:
    from phi_guard.adapters.json_logger import PHIAwareJSONFormatter

    handler = logging.StreamHandler()
    handler.setFormatter(PHIAwareJSONFormatter(redactor))
    handler.addFilter(PHIRedactionFilter(redactor))
    logger.addHandler(handler)
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from phi_guard.redactor import PHIRedactor


class PHIRedactionFilter(logging.Filter):
    """
    A logging.Filter that redacts PHI from log record messages and extra fields.

    Attaches to a logging handler (not a logger directly) to ensure redaction
    happens just before emission, regardless of which logger originated the record.

    Fields redacted:
      - msg (the log message string)
      - args (string arguments interpolated into msg)
      - exc_text (formatted exception text — stack traces sometimes contain PHI)
      - Any extra fields added via logging.info("...", extra={"patient": "..."})

    Fields NOT redacted:
      - name, levelname, levelno, pathname, filename, module, funcName, lineno
        (these are structural metadata, not user content)

    Args:
        redactor: A configured PHIRedactor instance.
        redact_exc_info: If True (default), also redact formatted exception text.
                         Disable if stack traces never contain PHI and you need
                         the full traceback for debugging.
        log_redaction_summary: If True, append a redaction summary to records
                               where PHI was found. Format: "(phi-guard: ssn=1)"
    """

    def __init__(
        self,
        redactor: PHIRedactor,
        redact_exc_info: bool = True,
        log_redaction_summary: bool = False,
    ) -> None:
        super().__init__()
        self._redactor = redactor
        self._redact_exc = redact_exc_info
        self._log_summary = log_redaction_summary

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Mutate the log record in place to redact PHI.

        Always returns True (the record is always emitted, just sanitized).
        """
        # Redact the message string
        if isinstance(record.msg, str):
            result = self._redactor.redact(record.msg)
            record.msg = result.redacted
            if self._log_summary and result.was_modified:
                summary = ", ".join(f"{k}={v}" for k, v in result.match_summary.items())
                record.msg += f"  (phi-guard redacted: {summary})"

        # Redact string args (these get interpolated into msg by the formatter)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    self._redactor.redact(a).redacted if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: self._redactor.redact(v).redacted if isinstance(v, str) else v
                    for k, v in record.args.items()
                }

        # Redact formatted exception text
        if self._redact_exc and record.exc_text:
            exc_result = self._redactor.redact(record.exc_text)
            record.exc_text = exc_result.redacted

        # Redact extra fields (anything beyond the standard LogRecord attributes)
        standard_attrs = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in standard_attrs and isinstance(value, str):
                setattr(record, key, self._redactor.redact(value).redacted)

        return True


class PHIAwareJSONFormatter(logging.Formatter):
    """
    A logging formatter that emits each record as a JSON object with PHI redacted.

    Compatible with structured logging platforms: Datadog, GCP Cloud Logging,
    Splunk, OpenSearch, Elastic/ECS.

    The JSON output includes:
        timestamp  — ISO 8601 UTC
        level      — log level name
        logger     — logger name
        message    — redacted message
        module     — source module
        function   — source function name
        line       — source line number
        <extra>    — any extra fields passed via the `extra` kwarg, also redacted

    Args:
        redactor: A configured PHIRedactor instance.
        include_source: If True (default), include module/function/line fields.
        extra_fields: Additional static fields to include in every record
                      (e.g., {"service": "triage-api", "env": "prod"}).
    """

    def __init__(
        self,
        redactor: PHIRedactor,
        include_source: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._redactor = redactor
        self._include_source = include_source
        self._extra_fields = extra_fields or {}
        # Apply the filter on the formatter level as a fallback
        self._filter = PHIRedactionFilter(redactor)

    def format(self, record: logging.LogRecord) -> str:
        # Run the filter to ensure redaction even if the handler didn't have it
        self._filter.filter(record)

        message = record.getMessage()

        output: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        if self._include_source:
            output.update({
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            })

        # Exception info
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            output["exception"] = self._redactor.redact(exc_text).redacted

        # Extra fields from the logging call
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "exc_info", "exc_text",
            "message",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                if isinstance(value, str):
                    output[key] = self._redactor.redact(value).redacted
                elif isinstance(value, (int, float, bool)) or value is None:
                    output[key] = value

        # Static extra fields
        output.update(self._extra_fields)

        return json.dumps(output, default=str)


def build_phi_safe_logger(
    name: str,
    redactor: Optional[PHIRedactor] = None,
    level: int = logging.INFO,
    structured: bool = True,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """
    Convenience factory: create a Python logger with PHI redaction pre-configured.

    Args:
        name: Logger name (e.g., "phi_guard.examples.leaky_app")
        redactor: PHIRedactor instance. Defaults to PHIRedactor() with standard config.
        level: Log level. Default: INFO.
        structured: If True, use PHIAwareJSONFormatter. If False, use plain text
                    with PHIRedactionFilter.
        extra_fields: Static fields to include in every structured log record.

    Returns:
        A configured logger that redacts PHI before emission.
    """
    redactor = redactor or PHIRedactor()
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if structured:
        handler.setFormatter(
            PHIAwareJSONFormatter(redactor, extra_fields=extra_fields)
        )
    else:
        handler.addFilter(PHIRedactionFilter(redactor))
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )

    logger.addHandler(handler)
    return logger
