"""
LLM-based PHI detection backend.

⚠️  READ BEFORE ENABLING ⚠️

CIRCULAR EXPOSURE RISK:
  This backend sends the text you want to protect — potentially containing PHI —
  to a *second* LLM API call. If that API call is not covered by the same BAA
  (Business Associate Agreement) as your primary LLM, you have created the exact
  HIPAA risk phi-guard is meant to prevent, one step later in the pipeline.

  Only use this backend when:
    1. The LLM you call here is covered by the same BAA as your primary model.
    2. You have confirmed with your compliance team that this two-call pattern
       is acceptable within your organization's data handling policy.
    3. You cannot accept the false-negative rate of the regex backend for
       your specific use case (typically: heavily narrative clinical text where
       names and contextual PHI matter more than structured identifiers).

LATENCY:
  This backend adds an additional LLM call per observation. At typical
  cloud API latencies (300–1000ms), this will meaningfully slow your
  observability pipeline. It is not suitable for synchronous hot paths.

WHEN THIS BACKEND IS WORTH THE TRADE-OFF:
  - Async post-hoc log sanitization (not in the request path)
  - Audit logging for flagged interactions
  - Clinical narrative where names are the primary PHI risk
  - High-compliance environments where missing a name is worse than
    the latency/cost/exposure trade-off

ALTERNATIVES:
  Consider presidio (Microsoft), AWS Comprehend Medical, or spaCy with
  an NER model as non-LLM alternatives with better privacy characteristics.
"""

from __future__ import annotations

import json
import warnings

from phi_guard.backends.base import Backend, RedactionMatch, RedactionResult
from phi_guard.config import RedactionConfig

_SYSTEM_PROMPT = """You are a PHI detection assistant. Your only job is to detect and
de-identify protected health information (PHI) in the text you receive.

Return ONLY a JSON object with this structure:
{
  "redacted_text": "<the input text with all PHI replaced by [REDACTED:<TYPE>]>",
  "found": [
    {"type": "<PHI_TYPE>", "original": "<matched text>", "replacement": "[REDACTED:<PHI_TYPE>]"}
  ]
}

PHI types to detect: PERSON_NAME, DOB, DATE, SSN, MRN, PHONE, EMAIL, ADDRESS, ZIP,
INSURANCE_ID, ACCOUNT_NUMBER, IP_ADDRESS, URL, AGE_OVER_89, DEVICE_ID, BIOMETRIC.

Rules:
- Replace every PHI instance in the text, even partial names or informal references.
- Do not explain your output. Return only the JSON object.
- If no PHI is found, return {"redacted_text": "<original text>", "found": []}.
- Do not redact clinical terminology (diagnoses, medications, procedures) unless
  they appear as part of a direct identifier.
"""


class LLMBackend(Backend):
    """
    PHI detection backend using an LLM as the detector.

    Requires the `anthropic` package: pip install phi-guard[llm-backend]

    Args:
        config: RedactionConfig (pattern settings are ignored; the LLM decides).
        model: Anthropic model ID to use for detection.
        client: Optional pre-configured anthropic.Anthropic client. If None,
                a client is constructed from environment variables.
        warn_on_init: If True (default), emit a warning explaining the
                      circular exposure risk when this backend is instantiated.

    Example:
        import anthropic
        from phi_guard.backends.llm_backend import LLMBackend
        from phi_guard.config import RedactionConfig

        # ⚠️ Only do this if your BAA covers the model you pass here.
        backend = LLMBackend(
            config=RedactionConfig(),
            model="claude-haiku-4-5-20251001",  # cheapest/fastest
            warn_on_init=True,
        )
    """

    def __init__(
        self,
        config: RedactionConfig,
        model: str = "claude-haiku-4-5-20251001",
        client: object | None = None,
        warn_on_init: bool = True,
    ) -> None:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "LLMBackend requires the 'anthropic' package. "
                "Install it with: pip install phi-guard[llm-backend]"
            ) from exc

        if warn_on_init:
            warnings.warn(
                "\n\n"
                "⚠️  phi-guard LLMBackend CIRCULAR EXPOSURE RISK ⚠️\n"
                "This backend sends potentially-PHI text to an LLM API call.\n"
                "Ensure the model you're calling is covered by the same BAA\n"
                "as your primary LLM before using this backend in production.\n"
                "See phi_guard/backends/llm_backend.py for full details.\n",
                stacklevel=2,
                category=UserWarning,
            )

        self.config = config
        self.model = model
        self._client = client or anthropic.Anthropic()

    def redact_text(self, text: str) -> RedactionResult:
        """
        Send text to an LLM for PHI detection and redaction.

        The LLM is instructed to return JSON containing the redacted text
        and a list of matches. If the LLM returns malformed JSON or an
        unexpected structure, falls back to the original text with a warning.
        """
        import anthropic  # type: ignore[import]

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
        except anthropic.APIError as exc:
            warnings.warn(
                f"phi-guard LLMBackend: API call failed ({exc}). "
                "Returning original text unredacted.",
                stacklevel=2,
            )
            return RedactionResult(redacted=text)

        raw_output = response.content[0].text if response.content else ""

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            # LLM returned non-JSON; extract JSON substring if possible
            import re
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None

        if not parsed or "redacted_text" not in parsed:
            warnings.warn(
                "phi-guard LLMBackend: model returned unexpected format. "
                "Returning original text unredacted.",
                stacklevel=2,
            )
            return RedactionResult(redacted=text)

        matches = []
        for item in parsed.get("found", []):
            if not isinstance(item, dict):
                continue
            matches.append(RedactionMatch(
                pattern_type=item.get("type", "UNKNOWN").lower(),
                original=item.get("original", ""),
                placeholder=item.get("replacement", "[REDACTED]"),
                start=-1,  # LLM doesn't return offsets
                end=-1,
            ))

        return RedactionResult(
            redacted=parsed["redacted_text"],
            matches=matches,
        )

    def supports_streaming(self) -> bool:
        return False
