"""
Streaming redaction for real-time LLM output.

The problem with streaming + redaction:
  PHI can be split across chunk boundaries. If the model streams "SSN: 123-45"
  in one chunk and "6789" in the next, simple per-chunk redaction misses it.

The solution — rolling buffer with safe zone:
  Maintain a buffer of the last N characters (N = max pattern length).
  Only emit text that is far enough back in the buffer that any PHI spanning
  from the current position would have been fully captured.

  For phi-guard's default pattern set, the longest possible PHI token is
  ~100 characters (a long insurance ID with keyword: "Beneficiary ID: ...").
  The default buffer_size of 256 provides a comfortable margin.

Usage — sync (Anthropic SDK):
    from phi_guard import PHIRedactor
    from phi_guard.streaming import StreamingRedactor

    redactor = PHIRedactor()
    stream_redactor = StreamingRedactor(redactor)

    with client.messages.stream(model=..., messages=...) as stream:
        for text_chunk in stream.text_stream:
            safe = stream_redactor.process_chunk(text_chunk)
            if safe:
                print(safe, end="", flush=True)
        # Always flush — emits the buffered tail
        final = stream_redactor.flush()
        if final:
            print(final, end="", flush=True)

Usage — async generator (FastAPI / asyncio):
    from phi_guard.streaming import redact_stream

    async def generate_safe(messages):
        raw_stream = client.messages.stream(model=..., messages=messages)
        async with raw_stream as stream:
            async for safe_chunk in redact_stream(stream.text_stream, redactor):
                yield safe_chunk

Usage — sync generator wrapper:
    from phi_guard.streaming import redact_stream_sync

    for safe_chunk in redact_stream_sync(some_sync_iterator, redactor):
        print(safe_chunk, end="", flush=True)

Usage — Langfuse integration:
    Wrap your generator before passing to Langfuse's trace.generation():

    chunks = []
    with client.messages.stream(...) as stream:
        for chunk in stream.text_stream:
            safe = stream_redactor.process_chunk(chunk)
            if safe:
                chunks.append(safe)
                yield safe
    final = stream_redactor.flush()
    if final:
        chunks.append(final)
        yield final

    # After streaming, log to Langfuse with the redacted output
    generation.update(output="".join(chunks))
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator, Optional

from phi_guard.backends.base import RedactionResult
from phi_guard.redactor import PHIRedactor


# Maximum length of any PHI pattern match in phi-guard's default pattern set.
# Used as the default buffer size. Increase if you add very long custom patterns.
_DEFAULT_BUFFER_SIZE = 256


class StreamingRedactor:
    """
    Stateful redactor for streaming text.

    Create one instance per stream. Not thread-safe across concurrent streams;
    create separate instances for concurrent requests.

    Args:
        redactor: A configured PHIRedactor instance.
        buffer_size: Number of characters to hold in the trailing buffer.
                     Increase if your custom patterns can match longer spans.
                     Default: 256.

    Attributes:
        total_chunks_processed: Count of process_chunk() calls made.
        total_redacted_count: Total number of PHI matches across all chunks.
    """

    def __init__(
        self,
        redactor: PHIRedactor,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
    ) -> None:
        self.redactor = redactor
        self.buffer_size = buffer_size
        self._buffer = ""
        self._all_matches: list = []
        self.total_chunks_processed = 0
        self.total_redacted_count = 0

    def process_chunk(self, chunk: str) -> str:
        """
        Add a chunk to the buffer and return any safely-emittable redacted text.

        "Safely emittable" means: text far enough back in the buffer that
        no PHI pattern could span from there to the current end.

        Returns empty string if the buffer hasn't filled past buffer_size yet.
        This is expected — the text will be emitted when the buffer fills
        or when flush() is called.

        Args:
            chunk: The raw text chunk from the LLM stream.

        Returns:
            Redacted text safe to emit, or "" if nothing ready yet.
        """
        if not chunk:
            return ""

        self.total_chunks_processed += 1
        self._buffer += chunk

        if len(self._buffer) <= self.buffer_size:
            # Buffer hasn't filled; nothing safe to emit yet
            return ""

        # The safe zone is everything except the trailing buffer_size chars
        safe_end = len(self._buffer) - self.buffer_size
        safe_text = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]

        result = self.redactor.redact(safe_text)
        self._all_matches.extend(result.matches)
        self.total_redacted_count += len(result.matches)
        return result.redacted

    def flush(self) -> str:
        """
        Emit all remaining buffered text.

        MUST be called after the last chunk. Any text still in the buffer
        (up to buffer_size characters) is redacted and returned.

        Returns:
            Redacted text from the remaining buffer, or "" if buffer was empty.
        """
        if not self._buffer:
            return ""

        result = self.redactor.redact(self._buffer)
        self._all_matches.extend(result.matches)
        self.total_redacted_count += len(result.matches)
        self._buffer = ""
        return result.redacted

    def reset(self) -> None:
        """Reset state for reuse with a new stream."""
        self._buffer = ""
        self._all_matches = []
        self.total_chunks_processed = 0
        self.total_redacted_count = 0

    @property
    def match_summary(self) -> dict:
        """
        Aggregate match summary across all chunks processed.
        Safe to log — does not contain original values.
        """
        summary: dict = {}
        for m in self._all_matches:
            summary[m.pattern_type] = summary.get(m.pattern_type, 0) + 1
        return summary

    @property
    def has_buffered_text(self) -> bool:
        """True if there is text in the buffer that hasn't been emitted yet."""
        return len(self._buffer) > 0


# ── Generator wrappers ────────────────────────────────────────────────────────

def redact_stream_sync(
    text_stream: Iterator[str],
    redactor: PHIRedactor,
    buffer_size: int = _DEFAULT_BUFFER_SIZE,
) -> Iterator[str]:
    """
    Sync generator that wraps a text stream with PHI redaction.

    Yields redacted chunks as they become safe to emit. Chunks may be
    smaller or larger than the input chunks depending on buffer flushing.

    Args:
        text_stream: Any iterable of string chunks (e.g., stream.text_stream).
        redactor: A configured PHIRedactor instance.
        buffer_size: Trailing buffer size in characters. Default: 256.

    Yields:
        Redacted text chunks, safe to log or transmit.

    Example:
        with client.messages.stream(model=..., messages=...) as stream:
            for safe_chunk in redact_stream_sync(stream.text_stream, redactor):
                print(safe_chunk, end="", flush=True)
    """
    sr = StreamingRedactor(redactor, buffer_size=buffer_size)

    for chunk in text_stream:
        safe = sr.process_chunk(chunk)
        if safe:
            yield safe

    final = sr.flush()
    if final:
        yield final


async def redact_stream(
    text_stream: AsyncIterator[str],
    redactor: PHIRedactor,
    buffer_size: int = _DEFAULT_BUFFER_SIZE,
) -> AsyncIterator[str]:
    """
    Async generator that wraps an async text stream with PHI redaction.

    Args:
        text_stream: Any async iterable of string chunks.
        redactor: A configured PHIRedactor instance.
        buffer_size: Trailing buffer size in characters. Default: 256.

    Yields:
        Redacted text chunks, safe to log or transmit.

    Example:
        async with client.messages.stream(model=..., messages=...) as stream:
            async for safe_chunk in redact_stream(stream.text_stream, redactor):
                yield safe_chunk
    """
    sr = StreamingRedactor(redactor, buffer_size=buffer_size)

    async for chunk in text_stream:
        safe = sr.process_chunk(chunk)
        if safe:
            yield safe

    final = sr.flush()
    if final:
        yield final
