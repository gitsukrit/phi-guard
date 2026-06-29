"""
Langfuse adapter for phi-guard.

Wraps a Langfuse client so that PHI is redacted from inputs and outputs
before they are transmitted to Langfuse's servers. The wrapper is a
transparent proxy: all non-intercepted methods pass through unchanged.

Requires: pip install phi-guard[langfuse]

Usage:
    from langfuse import Langfuse
    from phi_guard import PHIRedactor
    from phi_guard.adapters.langfuse_adapter import PHIGuardLangfuse

    langfuse = Langfuse(public_key="pk-...", secret_key="sk-...")
    redactor = PHIRedactor()

    safe_langfuse = PHIGuardLangfuse(langfuse, redactor)

    # Use exactly like the regular Langfuse client
    trace = safe_langfuse.trace(name="patient-query")
    generation = trace.generation(
        name="triage",
        model="claude-sonnet-4-6",
        input=[{"role": "user", "content": "My SSN is 123-45-6789"}],
        output="I can help with that. Please visit...",
    )
    # ↑ The SSN never reaches Langfuse servers.

LangChain callback handler:
    If you use LangChain, use PHIGuardLangfuseCallbackHandler instead:

    from phi_guard.adapters.langfuse_adapter import PHIGuardLangfuseCallbackHandler
    handler = PHIGuardLangfuseCallbackHandler(langfuse=langfuse, redactor=redactor)
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from phi_guard.redactor import PHIRedactor

logger = logging.getLogger(__name__)

# Fields in Langfuse generation/span payloads that typically contain LLM I/O
_REDACT_FIELDS = {"input", "output", "prompt", "completion", "metadata"}
_SAFE_FIELDS = {"name", "model", "usage", "id", "trace_id", "start_time", "end_time",
                "level", "status_message", "version", "release"}


class PHIGuardLangfuse:
    """
    A Langfuse client wrapper that redacts PHI before transmission.

    Intercepts `trace()`, `generation()`, `span()`, and `event()` calls,
    redacting PHI from `input`, `output`, `prompt`, and `metadata` fields.
    All other fields (model name, timestamps, usage counts) are passed through.

    Args:
        langfuse: A configured Langfuse client instance.
        redactor: A PHIRedactor instance.
        log_redaction_summary: If True, log (at DEBUG level) a summary of
                               what was redacted per call. The summary contains
                               pattern type counts only — not original values.
    """

    def __init__(
        self,
        langfuse: Any,
        redactor: PHIRedactor,
        log_redaction_summary: bool = False,
    ) -> None:
        self._langfuse = langfuse
        self._redactor = redactor
        self._log_summary = log_redaction_summary

    def __getattr__(self, name: str) -> Any:
        """Proxy all unintercepted attributes to the underlying client."""
        return getattr(self._langfuse, name)

    def trace(self, **kwargs: Any) -> Any:
        """Create a Langfuse trace with PHI-safe metadata."""
        kwargs = self._redact_kwargs(kwargs, context="trace")
        return _PHIGuardStatefulClient(
            self._langfuse.trace(**kwargs),
            redactor=self._redactor,
            log_summary=self._log_summary,
        )

    def generation(self, **kwargs: Any) -> Any:
        """Create a top-level generation with redacted input/output."""
        kwargs = self._redact_kwargs(kwargs, context="generation")
        return self._langfuse.generation(**kwargs)

    def span(self, **kwargs: Any) -> Any:
        """Create a top-level span with redacted metadata."""
        kwargs = self._redact_kwargs(kwargs, context="span")
        return self._langfuse.span(**kwargs)

    def event(self, **kwargs: Any) -> Any:
        """Create a top-level event with redacted payload."""
        kwargs = self._redact_kwargs(kwargs, context="event")
        return self._langfuse.event(**kwargs)

    def _redact_kwargs(self, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Redact PHI from Langfuse observation keyword arguments."""
        redacted = dict(kwargs)
        for field_name in _REDACT_FIELDS:
            if field_name not in redacted:
                continue
            value = redacted[field_name]
            if isinstance(value, str):
                result = self._redactor.redact(value)
                redacted[field_name] = result.redacted
                if self._log_summary and result.was_modified:
                    logger.debug(
                        "phi-guard [%s.%s]: %s",
                        context, field_name, result.match_summary
                    )
            elif isinstance(value, list):
                # OpenAI/Anthropic messages list
                redacted[field_name] = self._redactor.redact_messages(value)
            elif isinstance(value, dict):
                redacted[field_name] = self._redactor.redact_dict(value)

        return redacted


class _PHIGuardStatefulClient:
    """
    Proxy for Langfuse StatefulClient (trace/span returned objects) that
    intercepts generation/span/event/update calls on the returned objects.
    """

    def __init__(self, client: Any, redactor: PHIRedactor, log_summary: bool) -> None:
        self._client = client
        self._redactor = redactor
        self._log_summary = log_summary

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def generation(self, **kwargs: Any) -> Any:
        kwargs = self._redact_kwargs(kwargs, "generation")
        return self._client.generation(**kwargs)

    def span(self, **kwargs: Any) -> Any:
        kwargs = self._redact_kwargs(kwargs, "span")
        return self._client.span(**kwargs)

    def event(self, **kwargs: Any) -> Any:
        kwargs = self._redact_kwargs(kwargs, "event")
        return self._client.event(**kwargs)

    def update(self, **kwargs: Any) -> Any:
        kwargs = self._redact_kwargs(kwargs, "update")
        return self._client.update(**kwargs)

    def _redact_kwargs(self, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
        redacted = dict(kwargs)
        for field_name in _REDACT_FIELDS:
            if field_name not in redacted:
                continue
            value = redacted[field_name]
            if isinstance(value, str):
                result = self._redactor.redact(value)
                redacted[field_name] = result.redacted
                if self._log_summary and result.was_modified:
                    logger.debug(
                        "phi-guard [%s.%s]: %s",
                        context, field_name, result.match_summary
                    )
            elif isinstance(value, list):
                redacted[field_name] = self._redactor.redact_messages(value)
            elif isinstance(value, dict):
                redacted[field_name] = self._redactor.redact_dict(value)
        return redacted


# ── LangChain callback handler ────────────────────────────────────────────────

def _make_phi_guard_callback_handler(langfuse: Any, redactor: PHIRedactor) -> Any:
    """
    Create a LangChain callback handler that redacts PHI before forwarding
    to Langfuse.

    This is a factory function rather than a class definition to avoid
    importing LangChain at module import time. Call this only when you have
    LangChain installed.

    Args:
        langfuse: A configured Langfuse client.
        redactor: A PHIRedactor instance.

    Returns:
        A LangChain BaseCallbackHandler subclass instance.

    Raises:
        ImportError: If langfuse or langchain is not installed.
    """
    try:
        from langfuse.callback import CallbackHandler  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "LangChain callback requires langfuse[langchain]: "
            "pip install 'langfuse[langchain]'"
        ) from exc

    class _PHIGuardCallbackHandler(CallbackHandler):  # type: ignore[misc]
        def __init__(self) -> None:
            # Pass through Langfuse credentials from the existing client
            super().__init__(
                public_key=langfuse.client.public_key,  # type: ignore[attr-defined]
                secret_key=langfuse.client.secret_key,  # type: ignore[attr-defined]
                host=langfuse.client.host,  # type: ignore[attr-defined]
            )
            self._phi_redactor = redactor

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            # Redact outputs before passing to parent handler
            if hasattr(response, "generations"):
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "text"):
                            result = self._phi_redactor.redact(gen.text)
                            gen.text = result.redacted
            super().on_llm_end(response, **kwargs)

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            **kwargs: Any,
        ) -> None:
            # Redact inputs before passing to parent handler
            redacted_prompts = [
                self._phi_redactor.redact(p).redacted for p in prompts
            ]
            super().on_llm_start(serialized, redacted_prompts, **kwargs)

    return _PHIGuardCallbackHandler()


# Public alias
PHIGuardLangfuseCallbackHandler = _make_phi_guard_callback_handler
