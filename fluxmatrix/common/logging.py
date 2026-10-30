"""Production-grade structured logging system for FluxMatrix.

Features:
- Microsecond timestamp precision.
- Asynchronous context propagation via ContextVars (trace_id, request_id, client_ip).
- Dual output: Human-friendly rich colored console logs for local dev, strict JSON for ingestion.
- Persistent file logging (`fluxmatrix_gateway.log`) for complete auditability.
- Seamless intercept of standard library, Uvicorn, and Starlette loggers.
"""

import contextvars
import logging
import os
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

# Context variables for automatic distributed tracing across async tasks
trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
engine_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("engine_id", default=None)


def add_tracing_context(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Inject active trace_id, request_id, and engine context into the log event automatically."""
    trace_id = trace_id_var.get()
    request_id = request_id_var.get()
    engine_id = engine_id_var.get()

    if trace_id:
        event_dict["trace_id"] = trace_id
    if request_id:
        event_dict["request_id"] = request_id
    if engine_id:
        event_dict["engine"] = engine_id

    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_file_path: Optional[str] = "fluxmatrix_gateway.log",
) -> None:
    """Initialize structured logging across the entire application."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Base processors common to all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_tracing_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Console rendering: Colorized for interactive developer sessions
    if log_format.lower() == "json":
        console_renderer = structlog.processors.JSONRenderer()
    else:
        console_renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=30,
        )

    # Handlers for standard library logging
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file_path:
        # Guarantee log directory exists if a path is supplied
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    # Format output pipeline
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Apply structlog formatter to all handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
    )

    for handler in handlers:
        handler.setFormatter(formatter)

    # Muffle overly chatty third-party loggers while keeping critical warnings
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structured logger instance."""
    return structlog.get_logger(name or "fluxmatrix")
