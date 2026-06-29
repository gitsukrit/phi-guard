"""
Tests for phi_guard.streaming.

Key test: verify that PHI split across chunk boundaries is correctly caught.
"""

from __future__ import annotations

import pytest
from phi_guard import PHIRedactor
from phi_guard.streaming import StreamingRedactor, redact_stream_sync


class TestStreamingRedactor:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def test_complete_phi_in_single_large_chunk(self):
        """Full PHI in one chunk, no splitting."""
        sr = StreamingRedactor(self.redactor, buffer_size=10)
        # Chunk > buffer_size, so some is emitted immediately
        result = sr.process_chunk("Patient SSN: 123-45-6789, seen at clinic." * 5)
        final = sr.flush()
        full = result + final
        assert "123-45-6789" not in full
        assert "[REDACTED:SSN]" in full

    def test_phi_split_across_chunks(self):
        """SSN split at the dash boundary — the classic split-token failure mode."""
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        # Split "123-45-6789" at the middle
        c1 = sr.process_chunk("Patient SSN: 123-")
        c2 = sr.process_chunk("45-6789 was verified.")
        # With a 256-char buffer and short input, nothing emits until flush
        final = sr.flush()
        full = c1 + c2 + final
        assert "123-" not in full or "45-6789" not in full  # not split across output
        # The full SSN should be redacted
        assert "123-45-6789" not in full
        assert "[REDACTED:SSN]" in full

    def test_email_split_at_at_sign(self):
        """Email split across the @ symbol."""
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        c1 = sr.process_chunk("Contact: jane.doe")
        c2 = sr.process_chunk("@hospital.org for follow-up.")
        final = sr.flush()
        full = c1 + c2 + final
        assert "jane.doe@hospital.org" not in full
        assert "[REDACTED:EMAIL]" in full

    def test_clean_text_passes_through(self):
        """Text with no PHI is passed through unchanged."""
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        text = "The patient reported improvement. No concerns noted."
        # Send in one big chunk
        c1 = sr.process_chunk(text)
        final = sr.flush()
        full = c1 + final
        assert "improvement" in full
        assert "No concerns noted" in full

    def test_empty_chunks_ignored(self):
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        assert sr.process_chunk("") == ""
        assert sr.process_chunk("") == ""
        sr.flush()

    def test_flush_must_be_called(self):
        """Without flush, buffered text is not emitted."""
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        result = sr.process_chunk("short text")
        # Buffer hasn't filled past buffer_size — nothing emitted yet
        assert result == ""
        # flush emits the remainder
        final = sr.flush()
        assert "short text" in final

    def test_multiple_phi_types_across_stream(self):
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        chunks = [
            "Patient: MRN: 00123456",
            " DOB: 03/15/1982",
            " SSN: 123-45-6789",
            " email: p@clinic.org",
        ]
        output = ""
        for chunk in chunks:
            output += sr.process_chunk(chunk)
        output += sr.flush()

        assert "00123456" not in output
        assert "03/15/1982" not in output
        assert "123-45-6789" not in output
        assert "p@clinic.org" not in output

    def test_match_summary_after_stream(self):
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        sr.process_chunk("SSN: 123-45-6789 email: p@clinic.org")
        sr.flush()
        summary = sr.match_summary
        assert "ssn" in summary or "email" in summary

    def test_total_chunks_processed_counter(self):
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        sr.process_chunk("chunk 1")
        sr.process_chunk("chunk 2")
        sr.process_chunk("chunk 3")
        sr.flush()
        assert sr.total_chunks_processed == 3

    def test_reset_clears_state(self):
        sr = StreamingRedactor(self.redactor, buffer_size=256)
        sr.process_chunk("SSN: 123-45-6789")
        sr.flush()
        sr.reset()
        assert sr.total_chunks_processed == 0
        assert sr.total_redacted_count == 0
        assert not sr.has_buffered_text

    def test_buffer_overfill_emits_safe_zone(self):
        """When buffer fills past buffer_size, safe-zone text is emitted immediately."""
        sr = StreamingRedactor(self.redactor, buffer_size=50)
        # Send 200 chars of clean text in one chunk
        long_text = "The patient is responding well to treatment. " * 5
        emitted = sr.process_chunk(long_text)
        # Something should be emitted (safe zone ahead of buffer)
        assert len(emitted) > 0
        final = sr.flush()
        full = emitted + final
        assert "patient is responding well" in full


class TestRedactStreamSync:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def _stream(self, chunks):
        """Create a simple sync iterator from a list of chunks."""
        return iter(chunks)

    def test_basic_sync_stream(self):
        chunks = ["Hello, SSN: 123-", "45-6789 confirmed.", " End."]
        output = "".join(redact_stream_sync(self._stream(chunks), self.redactor))
        assert "123-45-6789" not in output
        assert "[REDACTED:SSN]" in output

    def test_clean_sync_stream(self):
        chunks = ["No PHI here.", " All clear.", " Thank you."]
        output = "".join(redact_stream_sync(self._stream(chunks), self.redactor))
        assert "No PHI here" in output

    def test_empty_stream(self):
        output = "".join(redact_stream_sync(iter([]), self.redactor))
        assert output == ""

    def test_single_chunk_stream(self):
        chunks = ["SSN: 123-45-6789"]
        output = "".join(redact_stream_sync(self._stream(chunks), self.redactor))
        assert "123-45-6789" not in output


# ── Async tests ───────────────────────────────────────────────────────────────

import asyncio

async def _async_chunks(chunks):
    for chunk in chunks:
        yield chunk

async def collect_async_stream(chunks, redactor):
    from phi_guard.streaming import redact_stream
    parts = []
    async for chunk in redact_stream(_async_chunks(chunks), redactor):
        parts.append(chunk)
    return "".join(parts)


class TestRedactStreamAsync:
    def setup_method(self):
        self.redactor = PHIRedactor()

    def test_basic_async_stream(self):
        chunks = ["DOB: 03/", "15/1982 confirmed."]
        result = asyncio.run(collect_async_stream(chunks, self.redactor))
        assert "03/15/1982" not in result
        assert "[REDACTED:DATE]" in result

    def test_clean_async_stream(self):
        chunks = ["Clean text.", " No identifiers here."]
        result = asyncio.run(collect_async_stream(chunks, self.redactor))
        assert "Clean text" in result

    def test_empty_async_stream(self):
        result = asyncio.run(collect_async_stream([], self.redactor))
        assert result == ""
