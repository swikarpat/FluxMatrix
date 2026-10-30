# FluxMatrix: High-Throughput AI Inference Gateway & Execution Control Plane

> **AI / Superintelligence (ASI) Systems Infrastructure**  
> Architected for high-concurrency LLM serving, dynamic semantic routing, Model Context Protocol (MCP) tool governance, and zero-overhead observability.

---

## 1. System Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │  Client Applications / ADK Agents       │
                      │  (OpenAI SDK, Google ADK, HTTP/SSE)    │
                      └──────────────────┬─────────────────────┘
                                         │
                             HTTP/2 POST /v1/chat/completions
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ FLUXMATRIX CONTROL PLANE & INFERENCE GATEWAY                                     │
│                                                                                  │
│   ┌───────────────────────────────┐     ┌────────────────────────────────────┐   │
│   │ Distributed Tracing & Logging │     │ Prometheus Telemetry               │   │
│   │ (ContextVars, X-Trace-ID, MDC)│     │ (TTFT, TPOT, Active Concurrency)   │   │
│   └───────────────┬───────────────┘     └────────────────────────────────────┘   │
│                   │                                                              │
│                   ▼                                                              │
│   ┌───────────────────────────────┐                                              │
│   │ MCP Tool Governance Engine    │  ──► Blocks forbidden tools (e.g. exec_shell) │
│   │ (RBAC & Credential Redaction) │  ──► Masks sensitive tokens/passwords        │
│   └───────────────┬───────────────┘                                              │
│                   │                                                              │
│                   ▼                                                              │
│   ┌───────────────────────────────┐                                              │
│   │ Semantic & Complexity Router  │                                              │
│   │ (Zero-GPU Intent Classifier)  │                                              │
│   └───────┬───────────────┬───────┘                                              │
└───────────┼───────────────┼──────────────────────────────────────────────────────┘
            │               │
     Low Complexity   High Complexity / Code / Reasoning
            │               │
            ▼               ▼
┌───────────────────┐   ┌───────────────────────────────────┐
│ FAST QUANTIZED    │   │ REASONING HEAVY TIER              │
│ TIER              │   │ (DeepSeek-R1-70B / Llama-3.3-70B) │
│ (Qwen-2.5-7B-AWQ) │   │ Deep multi-step reasoning         │
│ Low TTFT (<30ms)  │   │ High memory capacity              │
└───────────────────┘   └───────────────────────────────────┘
```

---

## 2. Key Systems Concepts (Interview Talking Points)

When hiring managers or Staff Engineers interview you about AI Infrastructure, these are the exact questions and systems concepts you can speak to with authority:

### Q1: "Why put a custom Gateway in front of vLLM or Triton instead of sending traffic directly?"
* **Answer:** Serving engines like vLLM are optimized for **tensor operations and continuous batching**, not edge traffic policy. Directly exposing vLLM creates three critical vulnerabilities:
  1. **GPU OOM crashes:** Spike traffic without admission control / backpressure saturates KV-cache VRAM and crashes the worker.
  2. **Compute Inefficiency (FinOps):** If simple requests ("Summarize this date") hit expensive 70B models, you waste \$100k/month in GPU compute. The Gateway performs **zero-GPU semantic routing** to offload 70% of volume to fast 7B/AWQ models.
  3. **Security:** Autonomous agents executing tools via the Model Context Protocol (MCP) need an external governance proxy to enforce RBAC, audit trails, and secret sanitization before requests reach the execution engines.

### Q2: "What is TTFT vs. TPOT, and why do they dictate LLM architecture?"
* **TTFT (Time to First Token):** The latency from when the user submits a prompt until the first token stream starts. TTFT is dominated by the **Prefill Phase** (computing attention across all prompt tokens in parallel).
* **TPOT (Time Per Output Token):** The incremental latency of generating each subsequent token during the **Decode Phase** (memory-bandwidth bound, generating one token at a time autoregressively).
* **System Design Implication:** FluxMatrix instruments separate Prometheus histograms for both `TTFT_SECONDS` and `TPOT_SECONDS` because fixing high TTFT requires **Prefix Caching & Prompt Optimization**, whereas fixing high TPOT requires **Quantization (AWQ/FP8) & Speculative Decoding**.

### Q3: "How does FluxMatrix handle distributed logging across concurrent async requests?"
* **Answer:** In high-concurrency Python ASGI servers, naive global variables cause race conditions between interleaved `async` coroutines. FluxMatrix uses **Python `contextvars` (Mapped Diagnostic Context)**:
  * When a request arrives, `DistributedTracingMiddleware` generates a unique `X-Trace-ID` and `X-Request-ID`.
  * These IDs are bound to async task-local storage.
  * Every log emitted anywhere in the call stack (router, MCP governor, engine streaming loop) automatically inherits the active `trace_id` and `engine_id` in structured JSON without passing variables manually down functions.

---

## 3. Directory Layout

```
FluxMatrix/
├── README.md                           # Architectural documentation & interview guide
├── requirements.txt                    # Production dependencies
├── pytest.ini                          # Test configuration
├── fluxmatrix/
│   ├── common/
│   │   ├── config.py                   # Pydantic v2 settings (env prefix FLUXMATRIX_)
│   │   ├── logging.py                  # Structured JSON & console logger with ContextVars
│   │   └── telemetry.py                # Prometheus metrics (TTFT, TPOT, Active Streams)
│   ├── core/
│   │   ├── models.py                   # Wire schemas (OpenAI / vLLM / Google ADK compatible)
│   │   ├── token_estimator.py          # Microsecond prompt tokenizer & complexity classifier
│   │   ├── router.py                   # Semantic routing engine (Fast vs Reasoning tiers)
│   │   └── mcp_governor.py             # Model Context Protocol security & RBAC governor
│   ├── engines/
│   │   ├── base.py                     # Abstract engine protocol
│   │   ├── mock_engine.py              # High-fidelity simulator (realistic TTFT/TPOT on Apple Silicon)
│   │   └── vllm_engine.py              # Async HTTP client for production vLLM workers
│   └── gateway/
│       ├── server.py                   # FastAPI / ASGI application factory
│       ├── routes.py                   # /v1/chat/completions (SSE stream), /health, /metrics
│       └── middleware.py               # Distributed tracing, latency timer, backpressure
├── examples/
│   └── stream_client_demo.py           # Live interactive client measuring real-time TTFT/TPOT
└── tests/
    ├── test_router.py                  # Routing classification unit tests
    ├── test_mcp_governor.py            # Security & tool authorization tests
    └── test_gateway_streaming.py       # Full HTTP & SSE streaming integration tests
```

---

## 4. Quick Start (Running on MacBook Pro M5 Pro)

### 1. Run the Gateway Server
```bash
source .venv/bin/activate
python -m fluxmatrix.gateway.server
```
Output:
```
2026-10-30T06:20:00Z [info ] Starting FluxMatrix AI Inference Gateway engine_mode=mock environment=development fast_tier=qwen-2.5-7b-instruct-awq heavy_tier=deepseek-r1-70b-instruct host=0.0.0.0 port=8080
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### 2. Run the Interactive Streaming Client
In a separate terminal:
```bash
source .venv/bin/activate
python examples/stream_client_demo.py
```
This tests:
1. **Simple query:** Automatically classified as low complexity and routed to `qwen-2.5-7b-instruct-awq`.
2. **Complex algorithmic reasoning:** Automatically classified as high complexity and routed to `deepseek-r1-70b-instruct`.
3. Displays a rich latency telemetry table with measured **TTFT** and **TPOT**.

### 3. Run the Test Suite
```bash
pytest -v
```
All 12 unit and integration tests pass verifying wire compatibility, MCP authorization, and async SSE streaming.

