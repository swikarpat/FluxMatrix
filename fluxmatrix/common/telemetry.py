"""Prometheus metrics telemetry for FluxMatrix Gateway.

Instruments real-time latency (TTFT, TPOT), streaming throughput,
active concurrency, and MCP tool governance violations.
"""

from prometheus_client import Counter, Gauge, Histogram

# Request and Stream Concurrency
REQUESTS_TOTAL = Counter(
    "fluxmatrix_requests_total",
    "Total requests processed by the gateway",
    ["method", "endpoint", "engine", "status"],
)

ACTIVE_STREAMS = Gauge(
    "fluxmatrix_active_streams",
    "Current active inference streaming connections",
    ["engine"],
)

# Inference Latency SLO Metrics (Key for AI Infra Interviews)
# TTFT: Time to First Token (measures prompt processing + prefill latency)
TTFT_SECONDS = Histogram(
    "fluxmatrix_ttft_seconds",
    "Time to First Token (TTFT) in seconds",
    ["engine", "routing_tier"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# TPOT: Time Per Output Token (measures token generation decode latency)
TPOT_SECONDS = Histogram(
    "fluxmatrix_tpot_seconds",
    "Time Per Output Token (TPOT) in seconds",
    ["engine"],
    buckets=(0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1),
)

# Overall Request Duration
REQUEST_DURATION_SECONDS = Histogram(
    "fluxmatrix_request_duration_seconds",
    "Total end-to-end duration for request in seconds",
    ["engine", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Token Accounting & Throughput
PROMPT_TOKENS_TOTAL = Counter(
    "fluxmatrix_prompt_tokens_total",
    "Total prompt input tokens ingested",
    ["engine"],
)

COMPLETION_TOKENS_TOTAL = Counter(
    "fluxmatrix_completion_tokens_total",
    "Total completion output tokens generated",
    ["engine"],
)

# Routing & FinOps Decisions
ROUTING_DECISIONS_TOTAL = Counter(
    "fluxmatrix_routing_decisions_total",
    "Total routing decisions made by classification tier",
    ["tier", "reason"],
)

# MCP Tool Governance Metrics
MCP_TOOL_CALLS_TOTAL = Counter(
    "fluxmatrix_mcp_tool_calls_total",
    "Total Model Context Protocol tool execution requests intercepted",
    ["tool_name", "status"],
)

MCP_SECURITY_VIOLATIONS_TOTAL = Counter(
    "fluxmatrix_mcp_security_violations_total",
    "Total blocked or unauthorized MCP tool execution attempts",
    ["tool_name", "violation_type"],
)
