"""
examples/leaky_app.py — Deliberately leaky healthcare chatbot.

This example shows the problem phi-guard solves.

Run without phi-guard first (default: LEAKY=true):
    python examples/leaky_app.py

Then with phi-guard protection:
    LEAKY=false python examples/leaky_app.py

The Anthropic API key is required. Set ANTHROPIC_API_KEY in your environment
or create a .env file. This example does NOT require Langfuse — it uses
the JSON logger adapter to show redaction in plain structured logs.

What this demonstrates:
  - Without phi-guard: patient SSN, DOB, email, and MRN appear in raw logs
  - With phi-guard: logs contain [REDACTED:SSN], [REDACTED:DATE], etc.
  - The LLM response itself is unchanged — only the LOG output is sanitized
  - match_summary shows WHAT was redacted without logging the actual values

DISCLAIMER: This is a developer demonstration. It is not a HIPAA compliance
implementation. See README.md for the full scope statement.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from io import StringIO

# ── Simulated patient intake data ─────────────────────────────────────────────
# In a real system this comes from an EHR or patient portal.
# These are fake identifiers — do not use real patient data in examples.

PATIENT_CONTEXT = {
    "mrn": "MRN: 00847291",
    "name": "Janet Morrison",           # Names require NER — not caught by regex
    "dob": "DOB: 11/22/1948",
    "ssn": "SSN: 289-44-1022",
    "email": "jmorrison1948@example.com",
    "insurance": "Member ID: BCBS-091827364",
    "presenting_complaint": (
        "Patient presents with chest pain radiating to left arm, onset 2 hours ago. "
        "History of hypertension (lisinopril 10mg QD), type 2 diabetes (metformin 500mg BID). "
        "No known drug allergies."
    ),
}

SYSTEM_PROMPT = """You are a clinical triage assistant. 
You will receive patient information and help assess urgency.
Respond concisely with an ESI triage level (1-5) and brief rationale.
"""

def build_user_message(patient: dict) -> str:
    return (
        f"Patient intake:\n"
        f"{patient['mrn']}, {patient['name']}, {patient['dob']}, {patient['ssn']}\n"
        f"Insurance: {patient['insurance']}\n"
        f"Contact: {patient['email']}\n\n"
        f"Presenting complaint: {patient['presenting_complaint']}"
    )


# ── LEAKY version: no phi-guard ───────────────────────────────────────────────

def run_leaky(api_key: str) -> None:
    """
    Simulate an LLM call with standard Python logging — no redaction.
    PHI will appear in log output exactly as provided.
    """
    import anthropic  # type: ignore[import]

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger("leaky_app")

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": build_user_message(PATIENT_CONTEXT)}]

    logger.info("=== LEAKY MODE: PHI appears in logs ===")
    logger.info("[REQUEST] Sending to LLM: %s", json.dumps(messages, indent=2))

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    output = response.content[0].text
    logger.info("[RESPONSE] LLM output: %s", output)
    logger.info(
        "[USAGE] input_tokens=%d output_tokens=%d",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    logger.info("--- End of leaky example ---")


# ── SAFE version: phi-guard enabled ──────────────────────────────────────────

def run_safe(api_key: str) -> None:
    """
    Same LLM call, but logs are routed through phi-guard's JSON logger adapter.
    PHI is redacted from log output before emission. The LLM receives the full
    context — only the observability layer sees the redacted version.
    """
    import anthropic  # type: ignore[import]
    from phi_guard import PHIRedactor, RedactionConfig
    from phi_guard.adapters.json_logger import (
        PHIRedactionFilter,
        PHIAwareJSONFormatter,
    )

    # Set up phi-guard
    redactor = PHIRedactor()

    # Set up structured logging with PHI redaction
    safe_logger = logging.getLogger("safe_app")
    safe_logger.setLevel(logging.INFO)
    safe_logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        PHIAwareJSONFormatter(
            redactor,
            extra_fields={"service": "triage-demo", "mode": "phi-guard-enabled"},
        )
    )
    handler.addFilter(PHIRedactionFilter(redactor))
    safe_logger.addHandler(handler)

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": build_user_message(PATIENT_CONTEXT)}]

    safe_logger.info("=== SAFE MODE: phi-guard redacting PHI from logs ===")

    # Redact the request payload before logging it
    safe_messages = redactor.redact_messages(messages)
    safe_logger.info(
        "[REQUEST] Sending to LLM (redacted): %s",
        json.dumps(safe_messages, indent=2),
    )

    # The LLM itself receives the full unredacted context
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=messages,  # full context to the model
    )

    output = response.content[0].text

    # Redact the response before logging
    output_result = redactor.redact(output)
    safe_logger.info("[RESPONSE] LLM output (redacted): %s", output_result.redacted)

    # Log the redaction summary — tells you WHAT was caught, not the original values
    if output_result.was_modified:
        safe_logger.info(
            "[PHI-GUARD] Output contained PHI. Redaction summary: %s",
            json.dumps(output_result.match_summary),
        )
    else:
        safe_logger.info("[PHI-GUARD] No PHI detected in LLM output.")

    safe_logger.info(
        "[USAGE] input_tokens=%d output_tokens=%d",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    safe_logger.info("--- End of safe example ---")


# ── Side-by-side demo without an API call ─────────────────────────────────────

def run_demo_no_api() -> None:
    """
    Show redaction behavior without making an LLM API call.
    Uses a canned LLM response to demonstrate the before/after in logs.
    """
    from phi_guard import PHIRedactor
    from phi_guard.adapters.json_logger import PHIRedactionFilter, PHIAwareJSONFormatter

    redactor = PHIRedactor()

    # Simulated LLM response that echoes back PHI (a common failure mode)
    simulated_response = (
        "Based on the intake, patient Janet Morrison (DOB: 11/22/1948, "
        "SSN: 289-44-1022) presents with chest pain. ESI Level 2 — "
        "high acuity, should be seen within 15 minutes. "
        "Contact: jmorrison1948@example.com for follow-up."
    )

    print("\n" + "=" * 60)
    print("BEFORE phi-guard (raw LLM output in logs):")
    print("=" * 60)
    print(simulated_response)

    print("\n" + "=" * 60)
    print("AFTER phi-guard (what gets logged):")
    print("=" * 60)
    result = redactor.redact(simulated_response)
    print(result.redacted)

    print("\n" + "=" * 60)
    print("phi-guard audit summary (safe to log, no PHI):")
    print("=" * 60)
    print(json.dumps(result.match_summary, indent=2))
    print(f"\nTotal redactions: {len(result.matches)}")

    print("\n" + "=" * 60)
    print("Note: 'Janet Morrison' was NOT redacted.")
    print("Names require NER (spaCy, AWS Comprehend Medical, Presidio).")
    print("The regex baseline catches structured identifiers only.")
    print("See README.md §Known Limitations.")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    leaky = os.environ.get("LEAKY", "demo").lower()

    if leaky == "demo" or not api_key:
        if not api_key:
            print("No ANTHROPIC_API_KEY set. Running offline demo.\n")
        run_demo_no_api()
    elif leaky == "true":
        print("Running LEAKY mode — PHI will appear in logs.\n")
        run_leaky(api_key)
    else:
        print("Running SAFE mode — PHI redacted from logs.\n")
        run_safe(api_key)
