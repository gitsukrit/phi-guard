# phi-guard

**PHI-aware observability middleware for healthcare AI.**

Redact protected health information from LLM traces, logs, and telemetry before sensitive data leaves your application.

---

## The problem

Observability tools are designed to capture everything. That's their entire value proposition: see what went in, see what came out, replay it, search it, alert on it. In healthcare AI, that design goal is a HIPAA liability by default.

The moment you instrument your LLM calls with Langfuse, LangSmith, Datadog, or any structured logger, patient SSNs, dates of birth, MRNs, and insurance IDs start appearing in your observability platform. Not because you made a mistake. Because the tool is working exactly as intended.

phi-guard is the shim between your LLM calls and your observability layer. It redacts PHI from what gets logged and traced, while leaving the full context intact for the model itself.

```
Without phi-guard:
  LLM input → Langfuse → "SSN: 289-44-1022, DOB: 11/22/1948, MRN: 00847291"

With phi-guard:
  LLM input → phi-guard → Langfuse → "SSN: [REDACTED:SSN], DOB: [REDACTED:DATE], MRN: [REDACTED:MRN]"
```

The model receives the full context. Your observability platform does not.

---

## What this is not

**phi-guard is a developer tool, not a HIPAA compliance product.**

It catches careless mistakes early — specifically, the class of mistake where PHI leaks into observability infrastructure because no one thought about it. It does not:

- Guarantee HIPAA compliance for your application or infrastructure
- Replace a BAA with your LLM provider, observability vendor, or cloud host
- Catch every form of PHI (names in free text require the Presidio backend — see below)
- Validate that your data handling policies are correct
- Substitute for a formal risk assessment or security review

This is the same relationship that `helmet.js` has to web security: it makes the obvious mistakes harder, but it is not a penetration test.

---

## Quickstart

```bash
pip install phi-guard
```

No dependencies required for the core library. Extras for adapters and advanced backends:

```bash
pip install "phi-guard[langfuse]"      # Langfuse adapter
pip install "phi-guard[otel]"           # OpenTelemetry adapter
pip install "phi-guard[presidio]"       # NER-based PHI detection (catches names)
pip install "phi-guard[llm-backend]"    # Experimental LLM-based detection
```

After installing the Presidio extra, download a spaCy model:

```bash
python -m spacy download en_core_web_lg   # best accuracy
# or: python -m spacy download en_core_web_sm  (faster, less accurate)
```

### Basic redaction

```python
from phi_guard import PHIRedactor

redactor = PHIRedactor()

result = redactor.redact(
    "Patient MRN: 00847291, DOB: 11/22/1948, SSN: 289-44-1022, "
    "contact: jmorrison@example.com"
)

print(result.redacted)
# → "Patient MRN: [REDACTED:MRN], DOB: [REDACTED:DATE], SSN: [REDACTED:SSN], "
#   "contact: [REDACTED:EMAIL]"

# Safe to log — shows WHAT was redacted without the original values
print(result.match_summary)
# → {'mrn': 1, 'date': 1, 'ssn': 1, 'email': 1}
```

### Dict redaction (LLM API payloads)

```python
response = client.messages.create(...)

# Redact the full response payload before logging
safe_payload = redactor.redact_dict(response.model_dump())
logger.info("LLM response", extra=safe_payload)
```

### Messages list redaction

```python
messages = [{"role": "user", "content": "My SSN is 289-44-1022"}]

# Redact before logging — the model still receives the full input
safe_messages = redactor.redact_messages(messages)
logger.info("Request: %s", safe_messages)
```

---

## Backends

phi-guard ships three backends. All implement the same `Backend` interface and
return `RedactionResult` objects with the same `redacted` / `match_summary` fields.

### RegexBackend (default)

Fast, zero-dependency, good for structured PHI: SSN, MRN, NPI, dates, phone,
email, insurance IDs, IP addresses. Does not catch person names.

```python
from phi_guard import PHIRedactor  # uses RegexBackend by default

redactor = PHIRedactor()
```

### PresidioBackend (NER — catches names)

Uses Microsoft Presidio + spaCy Named Entity Recognition. Catches what regex
cannot: person names, contextual addresses, informal date references. Higher
recall on clinical narrative text; 10–50x slower than RegexBackend.

```python
from phi_guard import PHIRedactor
from phi_guard.backends.presidio_backend import PresidioBackend
from phi_guard.config import RedactionConfig

config = RedactionConfig()
backend = PresidioBackend(config, model="en_core_web_lg", score_threshold=0.5)
redactor = PHIRedactor(backend=backend)

result = redactor.redact("Dr. Janet Morrison presents with chest pain.")
# → "Dr. [REDACTED:PERSON_NAME] presents with chest pain."
```

Available Presidio entities: `PERSON`, `DATE_TIME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`,
`US_SSN`, `LOCATION`, `IP_ADDRESS`, `MEDICAL_LICENSE`, and more. See
`phi_guard/backends/presidio_backend.py` for the full entity → phi-guard type mapping.

### CompositeBackend (recommended for production)

Chains multiple backends in sequence. RegexBackend runs first (fast, zero-cost
for structured identifiers), PresidioBackend runs on the residual (catching names
and contextual PHI that regex missed). Placeholders from the first backend are
not re-processed by the second.

```python
from phi_guard import PHIRedactor
from phi_guard.backends.presidio_backend import PresidioBackend, CompositeBackend
from phi_guard.backends.regex_backend import RegexBackend
from phi_guard.config import RedactionConfig

config = RedactionConfig()
redactor = PHIRedactor(backend=CompositeBackend([
    RegexBackend(config),        # fast — catches SSN, MRN, email, phone, NPI…
    PresidioBackend(config),     # NER — catches "Janet Morrison", contextual dates
]))

result = redactor.redact(
    "Patient Janet Morrison (DOB: 11/22/1948, SSN: 289-44-1022) presents with chest pain."
)
# → "Patient [REDACTED:PERSON_NAME] (DOB: [REDACTED:DATE], SSN: [REDACTED:SSN]) presents with chest pain."
```

### LLMBackend (experimental)

Uses a second LLM API call to detect PHI. High recall on clinical narrative text,
but has a circular exposure risk: the text you are trying to protect is sent to
another model.

**Read `phi_guard/backends/llm_backend.py` before using this backend.** Only use
it if the model you call here is covered by the same BAA as your primary model.

---

## Streaming

phi-guard provides streaming-safe redaction for real-time LLM output.

The core problem: PHI can be split across chunk boundaries. If the model streams
`"SSN: 123-45"` in one chunk and `"6789"` in the next, simple per-chunk redaction
misses it. phi-guard solves this with a rolling buffer that holds the last 256
characters and only emits text that is far enough back for any PHI pattern to have
been fully received.

### Sync streaming (Anthropic SDK)

```python
from phi_guard import PHIRedactor
from phi_guard.streaming import StreamingRedactor

redactor = PHIRedactor()
stream_redactor = StreamingRedactor(redactor)

with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=messages,
) as stream:
    for text_chunk in stream.text_stream:
        safe = stream_redactor.process_chunk(text_chunk)
        if safe:
            print(safe, end="", flush=True)
    # Always call flush() — emits the remaining buffer
    final = stream_redactor.flush()
    if final:
        print(final, end="", flush=True)

# After streaming: log the redaction summary (safe — no original values)
print(stream_redactor.match_summary)
# → {'ssn': 1, 'date': 2}
```

### Sync generator wrapper

```python
from phi_guard.streaming import redact_stream_sync

with client.messages.stream(model=..., messages=...) as stream:
    for safe_chunk in redact_stream_sync(stream.text_stream, redactor):
        print(safe_chunk, end="", flush=True)
```

### Async generator (FastAPI / asyncio)

```python
from phi_guard.streaming import redact_stream

async def generate_safe(messages):
    async with client.messages.stream(model=..., messages=messages) as stream:
        async for safe_chunk in redact_stream(stream.text_stream, redactor):
            yield safe_chunk
```

### Latency trade-off

The buffer means the last `buffer_size` characters (default: 256) arrive slightly
late — they are held until the stream ends and `flush()` is called. For user-facing
streaming where responsiveness matters more than marginal safety, reduce the buffer:

```python
# Smaller buffer = snappier UX, slightly lower protection for long PHI spans
stream_redactor = StreamingRedactor(redactor, buffer_size=64)
```

---

## Adapters

### Langfuse

```python
from langfuse import Langfuse
from phi_guard import PHIRedactor
from phi_guard.adapters.langfuse_adapter import PHIGuardLangfuse

langfuse = Langfuse(public_key="pk-...", secret_key="sk-...")
redactor = PHIRedactor()

# Wrap the client — use exactly like the standard Langfuse client
safe_langfuse = PHIGuardLangfuse(langfuse, redactor, log_redaction_summary=True)

trace = safe_langfuse.trace(name="patient-triage")
generation = trace.generation(
    name="triage-assessment",
    model="claude-haiku-4-5-20251001",
    input=[{"role": "user", "content": "Patient SSN: 289-44-1022, DOB: 11/22/1948"}],
    output="ESI Level 2. Chest pain with radiation requires immediate evaluation.",
)
# SSN and DOB never reach Langfuse servers.
```

### Standard Python logging

```python
import logging
from phi_guard import PHIRedactor
from phi_guard.adapters.json_logger import PHIRedactionFilter, PHIAwareJSONFormatter

redactor = PHIRedactor()

handler = logging.StreamHandler()
handler.addFilter(PHIRedactionFilter(redactor))
handler.setFormatter(PHIAwareJSONFormatter(
    redactor,
    extra_fields={"service": "triage-api", "env": "prod"},
))

logger = logging.getLogger("my_app")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Processing intake for MRN: 00847291, DOB: 11/22/1948")
# Emits: {"timestamp": "...", "level": "INFO", "message":
#         "Processing intake for MRN: [REDACTED:MRN], DOB: [REDACTED:DATE]", ...}
```

Or use the convenience factory:

```python
from phi_guard.adapters.json_logger import build_phi_safe_logger

logger = build_phi_safe_logger("my_app", structured=True)
```

### OpenTelemetry

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from phi_guard import PHIRedactor
from phi_guard.adapters.otel import PHIRedactionSpanProcessor

redactor = PHIRedactor()
provider = TracerProvider()

# Add BEFORE your export processor — order matters
provider.add_span_processor(PHIRedactionSpanProcessor(redactor))
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
```

---

## Configuration

### Enabling/disabling patterns

```python
from phi_guard import PHIRedactor, RedactionConfig

# Add ZIP codes (disabled by default — high false positive rate)
config = RedactionConfig().with_pattern("zip_code")

# Remove date redaction (e.g., scheduling system where dates are not PHI)
config = RedactionConfig().without_pattern("date")

# Strict: all patterns including ZIP and URL
config = RedactionConfig.strict()

# Minimal: SSN, NPI, MRN, email only (very low false positive rate)
config = RedactionConfig.minimal()

redactor = PHIRedactor(config=config)
```

### Custom patterns

```python
config = RedactionConfig(
    custom_patterns={
        # Your EHR's patient portal ID format
        "patient_portal_id": r"PP-\d{8}",
        # Internal facility code
        "facility_code": r"FAC-[A-Z]{3}-\d{4}",
    }
)
redactor = PHIRedactor(config=config)
```

### Custom placeholders

```python
# Default: [REDACTED:SSN]
config = RedactionConfig(placeholder_template="***{type}***")
# → ***SSN***

# Flat placeholder (loses type information)
config = RedactionConfig(placeholder_template="[REDACTED]")
```

---

## Pattern reference

| Pattern key | HIPAA identifier | Default | False positive risk | Notes |
|---|---|---|---|---|
| `ssn` | Social Security Number | ✅ | Low | Excludes invalid prefixes (000, 666, 9xx) |
| `mrn` | Medical Record Number | ✅ | Low | Keyword-anchored |
| `npi` | License/Certificate Number | ✅ | Low | 10-digit, keyword-anchored |
| `insurance_id` | Health Plan Beneficiary Number | ✅ | Low | Keyword-anchored, allows hyphens |
| `account_number` | Account Number | ✅ | Low | Keyword-anchored |
| `phone` | Telephone Number | ✅ | Medium | US formats |
| `fax` | Fax Number | ✅ | Low | Keyword-anchored |
| `email` | Email Address | ✅ | Low | Standard format |
| `date` | Dates (except year) | ✅ | Medium | MM/DD/YYYY and Month DD, YYYY |
| `age_over_89` | Ages over 89 | ✅ | Medium | HIPAA aggregation requirement |
| `ip_address` | IP Address | ✅ | Medium | IPv4 only |
| `zip_code` | Geographic — ZIP | ❌ | **High** | Matches any 5-digit number |
| `url` | Web URLs | ❌ | **High** | Often legitimate in healthcare AI |

Person names are not in this table because they require NER (PresidioBackend), not regex.

---

## Known limitations

### Names in the regex baseline

The RegexBackend does not catch person names. Names are HIPAA Safe Harbor
identifier #1 and the hardest to detect with regex — there is no lexical pattern
that distinguishes "Janet Morrison" from any other two-word sequence.

The PresidioBackend addresses this via spaCy NER. For the right trade-offs, see
the Backends section above.

### Free-text clinical narrative

"The patient reports worsening symptoms since her birthday last November" contains
temporal PHI that no pattern-based approach will catch. phi-guard is most effective
at catching structured identifiers that an LLM echoes back verbatim. Open-ended
narrative PHI requires semantic understanding.

### Streaming and NER

The PresidioBackend's NER requires full-sentence context. Per-chunk streaming
degrades accuracy significantly — `PresidioBackend.supports_streaming()` returns
`False` as a result. For streaming with name detection, collect the full response
first and then redact, or use a different NER approach (see below).

### The LLM backend has a circular exposure problem

The optional `LLMBackend` uses a second LLM call to detect PHI. If that model is
not covered by the same BAA as your primary model, you have moved the exposure
problem rather than eliminated it. Read the warning in
`phi_guard/backends/llm_backend.py` before using this backend.

### Regex is not adversarially robust

phi-guard is not designed to catch adversarial inputs — LLM outputs that
deliberately obfuscate PHI using typos, spacing, or encoding variations. It catches
the careless-mistakes category, not the intentional-exfiltration category.

---

## Demo

```bash
# Offline demo — no API key needed, shows before/after with a canned response
python examples/leaky_app.py

# With API key — leaky mode (PHI appears in logs)
ANTHROPIC_API_KEY=sk-... LEAKY=true python examples/leaky_app.py

# With API key — protected mode (PHI redacted from logs)
ANTHROPIC_API_KEY=sk-... LEAKY=false python examples/leaky_app.py
```

An interactive demo is also available as a React artifact — see the project page.
It shows the two-lane split (what the LLM receives vs. what gets logged) live,
with a toggle to switch protection on and off.

---

## Running tests

```bash
pip install "phi-guard[dev]"
pytest tests/ -v
```

115 tests covering: pattern accuracy (true positives and false positive avoidance),
redactor behavior, dict and messages traversal, config permutations, logging adapter,
streaming buffer logic (including split-token PHI across chunk boundaries), and
async generator correctness.

---

## Backend selection guide

| Scenario | Recommended backend |
|---|---|
| High-throughput structured logs | `RegexBackend` (default) |
| Clinical narrative where names matter | `PresidioBackend` |
| Both — standard production setup | `CompositeBackend([RegexBackend, PresidioBackend])` |
| Streaming user-facing responses | `StreamingRedactor` with `RegexBackend` |
| Post-hoc batch log sanitization | `PresidioBackend` or `LLMBackend` (read the BAA warning) |

---

## Contributing

phi-guard is intentionally small. Before adding a pattern, check:

1. Is the false positive rate documented and acceptable?
2. Does the pattern have tests for both detection and false positive avoidance?
3. Is the HIPAA Safe Harbor identifier this maps to documented?

PRs for additional adapters (LangSmith, Datadog LLM Observability) are welcome.
PRs for additional Presidio entity types or custom recognizers are also welcome.

---

## License

MIT. See [LICENSE](LICENSE).

---

*phi-guard does not make your application HIPAA-compliant. It makes one class of careless mistake less likely. Know the difference.*
