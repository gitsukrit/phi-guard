"""
OpenTelemetry adapter for phi-guard.

Provides a SpanProcessor that redacts PHI from span attributes before
they are exported to any OTel backend (Jaeger, Zipkin, Datadog APM,
Honeycomb, OTLP, etc.).

Requires: pip install phi-guard[otel]

Usage:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from phi_guard import PHIRedactor
    from phi_guard.adapters.otel import PHIRedactionSpanProcessor

    redactor = PHIRedactor()
    provider = TracerProvider()
    provider.add_span_processor(
        PHIRedactionSpanProcessor(redactor)
    )
    provider.add_span_processor(
        BatchSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("patient-query") as span:
        span.set_attribute("llm.prompt", "Patient SSN: 123-45-6789")
        span.set_attribute("llm.completion", "I see your SSN ends in 6789...")
    # ↑ Both attributes are redacted before export.

IMPORTANT: Add PHIRedactionSpanProcessor BEFORE your export processor.
The order matters — processors run in the order they are added.
"""

from __future__ import annotations

from typing import Optional, Sequence, Set

from phi_guard.redactor import PHIRedactor

# Span attributes that commonly contain LLM input/output content
_DEFAULT_REDACT_ATTRIBUTES: Set[str] = {
    # OpenTelemetry LLM semantic conventions (draft)
    "llm.prompt",
    "llm.completion",
    "llm.input.messages",
    "llm.output.messages",
    "gen_ai.prompt",
    "gen_ai.completion",
    # LangChain / LangSmith OTel attributes
    "langchain.input",
    "langchain.output",
    # Common custom attributes
    "user.query",
    "ai.input",
    "ai.output",
    "http.request.body",
    "http.response.body",
    "db.statement",
}


class PHIRedactionSpanProcessor:
    """
    OpenTelemetry SpanProcessor that redacts PHI from span attributes.

    Runs on span end (not start) to capture output attributes set during
    the span's lifetime. String attributes matching redact_attributes are
    redacted in place before the span is forwarded to the next processor.

    Args:
        redactor: A configured PHIRedactor instance.
        redact_attributes: Set of attribute keys to redact. Defaults to
                           _DEFAULT_REDACT_ATTRIBUTES (common LLM span attrs).
                           Pass a custom set to override entirely, or use
                           add_attributes to extend the defaults.
        redact_all_strings: If True, redact ALL string-valued attributes,
                            not just those in redact_attributes. Use with
                            caution — this includes structural attributes like
                            service.name and http.method.
    """

    def __init__(
        self,
        redactor: PHIRedactor,
        redact_attributes: Optional[Set[str]] = None,
        redact_all_strings: bool = False,
    ) -> None:
        self._redactor = redactor
        self._redact_attrs = redact_attributes or set(_DEFAULT_REDACT_ATTRIBUTES)
        self._redact_all = redact_all_strings

    def add_attributes(self, *attribute_keys: str) -> "PHIRedactionSpanProcessor":
        """Add attribute keys to the redaction set. Mutates in place."""
        self._redact_attrs.update(attribute_keys)
        return self

    def on_start(self, span: object, parent_context: object = None) -> None:
        """No-op on span start."""
        pass

    def on_end(self, span: object) -> None:
        """Redact PHI from span attributes before forwarding to export."""
        try:
            self._redact_span(span)
        except Exception:
            # Never let redaction failures break the tracing pipeline
            pass

    def _redact_span(self, span: object) -> None:
        """Mutate span attributes in place to redact PHI."""
        # Access span attributes via the OTel SDK internal API
        # This is intentionally accessing protected members because the
        # OTel SDK doesn't provide a public API for mutating attributes.
        attrs = getattr(span, "attributes", None) or getattr(span, "_attributes", None)
        if attrs is None:
            return

        # Collect keys to redact
        keys_to_redact = (
            set(attrs.keys()) if self._redact_all
            else self._redact_attrs & set(attrs.keys())
        )

        for key in keys_to_redact:
            value = attrs.get(key)
            if isinstance(value, str):
                result = self._redactor.redact(value)
                if result.was_modified:
                    # OTel span attributes are typically a MappingProxy or dict
                    # Try direct assignment; fall back to __setitem__
                    try:
                        attrs[key] = result.redacted
                    except TypeError:
                        # MappingProxy — need to use span.set_attribute
                        if hasattr(span, "set_attribute"):
                            span.set_attribute(key, result.redacted)  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
