"""Standard data protocols and wire schemas for FluxMatrix.

Supports wire-compatibility with OpenAI/vLLM standards and integrates Google ADK / MCP
metadata for unified agentic governance.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    """Client request schema wire-compatible with standard vLLM / OpenAI endpoints."""

    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # FluxMatrix Extensions:
    routing_override: Optional[str] = Field(
        default=None,
        description="Explicit engine override to bypass automatic semantic routing if needed.",
    )
    user_id: Optional[str] = "anonymous"
    agent_id: Optional[str] = "adk-agent-default"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    # System metadata injected by FluxMatrix control plane
    fluxmatrix_engine: Optional[str] = None
    fluxmatrix_ttft_ms: Optional[float] = None
    fluxmatrix_tpot_ms: Optional[float] = None


class StreamChunkDelta(BaseModel):
    role: Optional[MessageRole] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: StreamChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    # Optional stream usage stats
    usage: Optional[UsageInfo] = None


# --- Internal Systems & Governance Types ---

class RoutingTier(str, Enum):
    FAST_QUANTIZED = "fast_quantized"       # e.g., Qwen-2.5-7B-AWQ, lower cost, ultra-low TTFT
    REASONING_HEAVY = "reasoning_heavy"     # e.g., DeepSeek-R1-70B / Qwen-72B, high depth
    DIRECT_OVERRIDE = "direct_override"     # Explicit client override


class RoutingDecision(BaseModel):
    tier: RoutingTier
    selected_model: str
    complexity_score: float
    estimated_prompt_tokens: int
    requires_tools: bool
    reason: str


class MCPAuthorizationDecision(BaseModel):
    is_allowed: bool
    tool_name: str
    reason: str
    sanitized_arguments: Optional[Dict[str, Any]] = None

