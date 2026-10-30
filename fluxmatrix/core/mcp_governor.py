"""Model Context Protocol (MCP) Security & Tool Governance Engine.

Provides:
- Tool-level Role Based Access Control (RBAC).
- Parameter sanitization (stripping API keys, passwords, credentials).
- Dangerous tool blocking (e.g., shell execution, arbitrary file deletions).
- Audit trail logging for enterprise agent compliance.
"""

import json
import re
from typing import Any, Dict, List, Optional
from fluxmatrix.common.config import GatewaySettings
from fluxmatrix.common.logging import get_logger
from fluxmatrix.common.telemetry import (
    MCP_SECURITY_VIOLATIONS_TOTAL,
    MCP_TOOL_CALLS_TOTAL,
)
from fluxmatrix.core.models import MCPAuthorizationDecision, ToolCall, ToolDefinition

logger = get_logger("mcp_governor")

# Sensitive parameter keys to automatically redact
SENSITIVE_KEY_PATTERNS = [
    re.compile(r"(api[_-]?key|secret|token|password|auth|private[_-]?key)", re.IGNORECASE)
]


class MCPGovernor:
    """Enterprise policy engine for inspecting, authorizing, and sanitizing MCP tool interactions."""

    def __init__(self, settings: GatewaySettings):
        self.settings = settings
        self.allowed_namespaces = set(settings.allowed_mcp_namespaces)
        self.blocked_tools = set(settings.blocked_mcp_tools)

    def validate_tool_definitions(self, tools: Optional[List[ToolDefinition]]) -> List[ToolDefinition]:
        """Filter out blocked or unauthorized tools from the request schema before passing to the model."""
        if not tools:
            return []

        filtered_tools: List[ToolDefinition] = []
        for tool in tools:
            fn = tool.function or {}
            tool_name = fn.get("name", "")

            # Check explicit blocklist
            if tool_name in self.blocked_tools:
                MCP_SECURITY_VIOLATIONS_TOTAL.labels(
                    tool_name=tool_name, violation_type="explicit_blocklist"
                ).inc()
                logger.warning(
                    "Blocked forbidden tool schema declaration",
                    tool_name=tool_name,
                    violation="explicit_blocklist",
                )
                continue

            filtered_tools.append(tool)

        return filtered_tools

    def authorize_tool_call(
        self,
        tool_call: ToolCall,
        agent_id: str = "default_agent",
        user_role: str = "standard_user",
    ) -> MCPAuthorizationDecision:
        """Enforce real-time authorization on an agent's tool execution attempt."""
        tool_name = tool_call.function.name

        MCP_TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status="evaluated").inc()

        # 1. Check explicit blocklist
        if tool_name in self.blocked_tools:
            MCP_SECURITY_VIOLATIONS_TOTAL.labels(
                tool_name=tool_name, violation_type="blocked_tool"
            ).inc()
            logger.error(
                "Agent attempted to execute blocked MCP tool",
                agent_id=agent_id,
                tool_name=tool_name,
                user_role=user_role,
            )
            return MCPAuthorizationDecision(
                is_allowed=False,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is blocked by security policy.",
                sanitized_arguments=None,
            )

        # 2. Check namespace authorization
        if self.settings.mcp_enforce_rbac:
            namespace = tool_name.split(":")[0] if ":" in tool_name else tool_name
            is_permitted = any(
                allowed == tool_name or allowed == namespace or allowed.startswith(f"{namespace}:")
                for allowed in self.allowed_namespaces
            )
            if not is_permitted:
                MCP_SECURITY_VIOLATIONS_TOTAL.labels(
                    tool_name=tool_name, violation_type="unauthorized_namespace"
                ).inc()
                logger.warning(
                    "MCP tool namespace not in allowed policy",
                    tool_name=tool_name,
                    namespace=namespace,
                    agent_id=agent_id,
                )
                return MCPAuthorizationDecision(
                    is_allowed=False,
                    tool_name=tool_name,
                    reason=f"Namespace '{namespace}' is not authorized for agent '{agent_id}'.",
                    sanitized_arguments=None,
                )

        # 3. Parse and sanitize tool arguments
        try:
            raw_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except Exception:
            raw_args = {"raw": tool_call.function.arguments}

        sanitized_args = self._sanitize_parameters(raw_args)

        if self.settings.mcp_audit_logging:
            logger.info(
                "Authorized MCP tool execution",
                agent_id=agent_id,
                tool_name=tool_name,
                arguments=sanitized_args,
            )

        return MCPAuthorizationDecision(
            is_allowed=True,
            tool_name=tool_name,
            reason="Tool call authorized by policy.",
            sanitized_arguments=sanitized_args,
        )

    def _sanitize_parameters(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive authentication tokens or credentials in tool payloads."""
        sanitized: Dict[str, Any] = {}
        for key, value in arguments.items():
            is_sensitive = any(pattern.search(key) for pattern in SENSITIVE_KEY_PATTERNS)
            if is_sensitive and isinstance(value, str):
                sanitized[key] = "[REDACTED_BY_FLUXMATRIX_MCP_GOVERNOR]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_parameters(value)
            else:
                sanitized[key] = value
        return sanitized

