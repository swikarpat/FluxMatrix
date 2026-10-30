"""Central configuration management for FluxMatrix Gateway using Pydantic Settings."""

from typing import List, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Runtime configuration for FluxMatrix control plane."""

    model_config = SettingsConfigDict(
        env_prefix="FLUXMATRIX_",
        env_file=".env",
        extra="ignore",
    )

    # Environment & Network
    environment: Literal["development", "production", "testing"] = "development"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    debug: bool = False

    # Inference Engine & Serving Backend
    engine_mode: Literal["mock", "vllm", "metal"] = "mock"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-fluxmatrix-internal"
    engine_timeout_seconds: float = 60.0
    max_concurrent_requests: int = 1000

    # Dynamic Routing & FinOps Tiering
    fast_tier_model: str = "qwen-2.5-7b-instruct-awq"
    heavy_tier_model: str = "deepseek-r1-70b-instruct"
    default_model: str = "qwen-2.5-7b-instruct-awq"
    complexity_threshold: float = Field(
        default=0.60,
        description="Threshold [0.0 - 1.0] above which queries route to heavy reasoning engine.",
    )
    long_context_token_threshold: int = Field(
        default=4096,
        description="Prompt token count that triggers long-context prefix routing.",
    )

    # Model Context Protocol (MCP) Security & Tool Governance
    mcp_enforce_rbac: bool = True
    mcp_audit_logging: bool = True
    allowed_mcp_namespaces: List[str] = [
        "filesystem:read",
        "database:query",
        "search:read",
        "calculator",
    ]
    blocked_mcp_tools: List[str] = [
        "filesystem:delete",
        "system:execute_shell",
        "system:eval",
        "network:raw_socket",
    ]

    # Structured Logging & Observability
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_file_enabled: bool = True
    log_file_path: str = "fluxmatrix_gateway.log"

    # Telemetry & SLO Tracking
    enable_metrics: bool = True
    target_ttft_slo_ms: float = 80.0  # Time to First Token SLO target in ms
    target_tpot_slo_ms: float = 25.0  # Time Per Output Token SLO target in ms


settings = GatewaySettings()

