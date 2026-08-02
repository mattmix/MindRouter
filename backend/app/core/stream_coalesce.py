############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# stream_coalesce.py: batches framed streaming events per socket write
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Write-side coalescing for streaming responses.

Per-token socket writes dominate gateway CPU on fast backends: every SSE
``data:`` block (or Ollama NDJSON line) costs an ASGI send + kernel write.
``StreamCoalescer`` buffers framed events and releases them in batches so
the write path pays one send per batch instead of one per token.

Latency-sensitive events are never delayed: the first event of a stream
(TTFT), any event carrying finish_reason/usage, errors, and [DONE] all
flush immediately, joined with whatever is buffered ahead of them so
ordering is preserved.

The max-delay bound is checked only as events arrive — there is no idle
timer, so during a backend stall buffered events flush on the next
arrival or at end of stream, not after ``max_delay_ms``.
"""

import time
from typing import List, Optional


class StreamCoalescer:
    """Batch framed stream events (SSE blocks or NDJSON lines).

    Flush triggers: first event of the stream, ``max_events`` buffered,
    ``max_delay_ms`` elapsed since the last flush (checked as events
    arrive), or ``force=True``.  ``max_events <= 1`` or
    ``max_delay_ms <= 0`` disables coalescing — every event passes
    straight through, restoring the per-event write behavior.
    """

    def __init__(self, max_events: int, max_delay_ms: int):
        self.enabled = max_events > 1 and max_delay_ms > 0
        self._max_events = max_events
        self._max_delay_s = max_delay_ms / 1000.0
        self._buf: List[bytes] = []
        self._first_sent = False
        self._last_flush = time.monotonic()

    def add(self, event: bytes, force: bool = False) -> Optional[bytes]:
        """Buffer one framed event; return bytes to write now, or None."""
        if not self.enabled or force or not self._first_sent:
            self._first_sent = True
            if self._buf:
                self._buf.append(event)
                return self._drain()
            self._last_flush = time.monotonic()
            return event

        self._buf.append(event)
        if (
            len(self._buf) >= self._max_events
            or time.monotonic() - self._last_flush >= self._max_delay_s
        ):
            return self._drain()
        return None

    def flush(self) -> Optional[bytes]:
        """Drain buffered events (end of stream / error paths)."""
        if self._buf:
            return self._drain()
        return None

    def _drain(self) -> bytes:
        out = b"".join(self._buf)
        self._buf.clear()
        self._last_flush = time.monotonic()
        return out
