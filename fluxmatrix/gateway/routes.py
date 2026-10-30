"""API endpoints for FluxMatrix Gateway.

Exposes:
- POST /v1/chat/completions (OpenAI/vLLM wire-compatible streaming & batch endpoint)
- POST /v1/mcp/authorize (Model Context Protocol authorization check)
- GET  /v1/models (Model list and tier capabilities)
- GET  /health (Gateway and engine health status)
- GET  /metrics (Prometheus scrape endpoint)
"""

import json
from typing import AsyncIterator, List
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fluxmatrix.common.config import GatewaySettings, settings
from fluxmatrix.common.logging import get_logger
from fluxmatrix.core.mcp_governor import MCPGovernor
from fluxmatrix.core.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    MCPAuthorizationDecision,
    ToolCall,
)
from fluxmatrix.core.router import SemanticRouter
from fluxmatrix.engines.base import BaseInferenceEngine
from fluxmatrix.engines.mock_engine import MockInferenceEngine
from fluxmatrix.engines.vllm_engine import VLLMInferenceEngine

logger = get_logger("routes")
router = APIRouter()

# Global engine and governance singletons (initialized on startup)
_engine: BaseInferenceEngine = None
_semantic_router: SemanticRouter = None
_mcp_governor: MCPGovernor = None


def init_dependencies(custom_settings: GatewaySettings = settings):
    """Initialize core gateway dependencies based on configuration settings."""
    global _engine, _semantic_router, _mcp_governor

    _semantic_router = SemanticRouter(settings=custom_settings)
    _mcp_governor = MCPGovernor(settings=custom_settings)

    if custom_settings.engine_mode == "vllm":
        _engine = VLLMInferenceEngine(
            engine_id="vllm-primary",
            base_url=custom_settings.vllm_base_url,
            api_key=custom_settings.vllm_api_key,
            timeout_seconds=custom_settings.engine_timeout_seconds,
        )
    else:
        _engine = MockInferenceEngine(
            engine_id="fluxmatrix-mock-vllm",
            base_ttft_ms=25.0,
            base_tpot_ms=10.0,
        )


def get_engine() -> BaseInferenceEngine:
    global _engine
    if _engine is None:
        init_dependencies()
    return _engine


def get_router() -> SemanticRouter:
    global _semantic_router
    if _semantic_router is None:
        init_dependencies()
    return _semantic_router


def get_mcp_governor() -> MCPGovernor:
    global _mcp_governor
    if _mcp_governor is None:
        init_dependencies()
    return _mcp_governor


async def sse_event_stream(
    chunk_iterator: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[str]:
    """Format ChatCompletionChunk objects into standard Server-Sent Events (SSE) wire protocol."""
    try:
        async for chunk in chunk_iterator:
            chunk_json = chunk.model_dump_json(exclude_none=True)
            yield f"data: {chunk_json}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("Error during SSE token streaming", error=str(e))
        error_payload = json.dumps({"error": {"message": str(e), "type": "stream_error"}})
        yield f"data: {error_payload}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    engine: BaseInferenceEngine = Depends(get_engine),
    semantic_router: SemanticRouter = Depends(get_router),
    mcp_governor: MCPGovernor = Depends(get_mcp_governor),
):
    """Unified OpenAI/vLLM wire-compatible chat completion endpoint.

    Performs:
    1. Tool validation under MCP security policies.
    2. Dynamic semantic complexity routing.
    3. Asynchronous execution or SSE streaming.
    """
    # 1. MCP Tool Governance: Validate tool schemas
    if request.tools:
        request.tools = mcp_governor.validate_tool_definitions(request.tools)

    # 2. Dynamic Routing: Determine target engine & model
    routing_decision = semantic_router.route_request(request)
    target_model = routing_decision.selected_model

    # 3. Handle Streaming vs Non-Streaming
    if request.stream:
        stream_generator = engine.generate_stream(request, model_override=target_model)
        return StreamingResponse(
            sse_event_stream(stream_generator),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-FluxMatrix-Tier": routing_decision.tier.value,
                "X-FluxMatrix-Model": target_model,
                "X-FluxMatrix-Complexity": f"{routing_decision.complexity_score:.2f}",
            },
        )

    # Non-streaming batch response
    response: ChatCompletionResponse = await engine.generate_complete(request, model_override=target_model)
    return response


@router.post("/v1/mcp/authorize", response_model=MCPAuthorizationDecision)
async def authorize_mcp_tool(
    tool_call: ToolCall,
    agent_id: str = "default-agent",
    user_role: str = "standard_user",
    mcp_governor: MCPGovernor = Depends(get_mcp_governor),
):
    """Direct verification endpoint for Google ADK agents to authorize an MCP tool call."""
    decision = mcp_governor.authorize_tool_call(
        tool_call=tool_call,
        agent_id=agent_id,
        user_role=user_role,
    )
    return decision


@router.get("/v1/models")
async def list_models():
    """List available model endpoints, tiers, and routing thresholds."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.fast_tier_model,
                "object": "model",
                "tier": "fast_quantized",
                "target_ttft_slo_ms": settings.target_ttft_slo_ms,
            },
            {
                "id": settings.heavy_tier_model,
                "object": "model",
                "tier": "reasoning_heavy",
                "target_ttft_slo_ms": settings.target_ttft_slo_ms * 2.5,
            },
        ],
    }


@router.get("/health")
async def health_check(engine: BaseInferenceEngine = Depends(get_engine)):
    """Health check verifying gateway and downstream engine connectivity."""
    engine_healthy = await engine.health()
    status = "healthy" if engine_healthy else "degraded"
    return {
        "status": status,
        "gateway": "online",
        "engine": engine.get_stats(),
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics scrape target."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

