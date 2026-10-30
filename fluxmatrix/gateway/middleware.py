"""Distributed tracing, latency accounting, and backpressure middleware for FluxMatrix."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fluxmatrix.common.logging import (
    get_logger,
    request_id_var,
    trace_id_var,
)
from fluxmatrix.common.telemetry import (
    REQUEST_DURATION_SECONDS,
    REQUESTS_TOTAL,
)

logger = get_logger("middleware")


class DistributedTracingMiddleware(BaseHTTPMiddleware):
    """Injects and propagates unique Request and Trace identifiers across all async coroutines."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Extract or generate correlation IDs
        trace_id = request.headers.get("X-Trace-ID") or f"trc-{uuid.uuid4().hex[:16]}"
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"

        # Bind to async contextvars for structured logger auto-injection
        trace_token = trace_id_var.set(trace_id)
        request_token = request_id_var.set(request_id)

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code

            duration = time.perf_counter() - start_time
            duration_ms = duration * 1000.0

            # Attach tracing headers to outbound response
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"

            # Record Prometheus metrics
            endpoint = request.url.path
            REQUEST_DURATION_SECONDS.labels(engine="gateway", status=str(status_code)).observe(duration)
            REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                engine="gateway",
                status=str(status_code),
            ).inc()

            return response

        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.exception("Unhandled gateway exception", error=str(exc), path=request.url.path)
            REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                engine="gateway",
                status="500",
            ).inc()
            raise

        finally:
            trace_id_var.reset(trace_token)
            request_id_var.reset(request_token)

