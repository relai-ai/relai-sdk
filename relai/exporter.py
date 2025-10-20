import threading
import typing

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class InMemorySpanExporter(SpanExporter):
    """Implementation of :class:`.SpanExporter` that stores spans in memory.

    This class can be used for testing purposes. It stores the exported spans
    in a list in memory that can be retrieved using the
    :func:`.get_finished_spans` method. It preserves at most `max_spans` latest spans.
    """

    def __init__(self, max_spans: int = 2000000) -> None:
        self._finished_spans: typing.List[ReadableSpan] = []
        self._stopped = False
        self._lock = threading.Lock()
        self._max_spans = max_spans

    def clear(self) -> None:
        """Clear list of collected spans."""
        with self._lock:
            self._finished_spans.clear()

    def get_finished_spans(self) -> typing.Tuple[ReadableSpan, ...]:
        """Get list of collected spans."""
        with self._lock:
            return tuple(self._finished_spans)

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        """Stores a list of spans in memory, preserving at most `max_spans` latest spans."""
        if self._stopped:
            return SpanExportResult.FAILURE
        with self._lock:
            self._finished_spans.extend(spans)
            # Trim to keep only the latest max_spans
            if len(self._finished_spans) > self._max_spans:
                excess = len(self._finished_spans) - self._max_spans
                self._finished_spans = self._finished_spans[excess:]
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shut downs the exporter.

        Calls to export after the exporter has been shut down will fail.
        """
        self._stopped = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
