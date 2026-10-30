"""Unit tests for the Model Context Protocol (MCP) Security & Tool Governance Engine."""

import json
import pytest
from fluxmatrix.common.config import GatewaySettings
from fluxmatrix.core.mcp_governor import MCPGovernor
from fluxmatrix.core.models import FunctionCall, ToolCall, ToolDefinition


@pytest.fixture
def governor():
    settings = GatewaySettings(
        mcp_enforce_rbac=True,
        allowed_mcp_namespaces=["database:query", "filesystem:read", "calculator"],
        blocked_mcp_tools=["system:execute_shell", "filesystem:delete"],
    )
    return MCPGovernor(settings=settings)


def test_authorized_tool_call(governor):
    """Permitted tools with harmless parameters should be approved."""
    tool_call = ToolCall(
        id="call_123",
        function=FunctionCall(
            name="database:query",
            arguments=json.dumps({"sql": "SELECT id, name FROM users LIMIT 10"}),
        ),
    )
    decision = governor.authorize_tool_call(tool_call, agent_id="sql-agent")
    assert decision.is_allowed is True
    assert decision.tool_name == "database:query"
    assert decision.sanitized_arguments["sql"] == "SELECT id, name FROM users LIMIT 10"


def test_blocked_dangerous_tool(governor):
    """Explicitly blocked dangerous tools must be rejected."""
    tool_call = ToolCall(
        id="call_456",
        function=FunctionCall(
            name="system:execute_shell",
            arguments=json.dumps({"command": "rm -rf /"}),
        ),
    )
    decision = governor.authorize_tool_call(tool_call, agent_id="rogue-agent")
    assert decision.is_allowed is False
    assert "blocked by security policy" in decision.reason


def test_unauthorized_namespace(governor):
    """Tools outside permitted namespaces should be rejected under RBAC."""
    tool_call = ToolCall(
        id="call_789",
        function=FunctionCall(
            name="payment:charge_credit_card",
            arguments=json.dumps({"amount": 100}),
        ),
    )
    decision = governor.authorize_tool_call(tool_call, agent_id="guest-agent")
    assert decision.is_allowed is False
    assert "not authorized" in decision.reason


def test_sensitive_credential_sanitization(governor):
    """Sensitive parameters (e.g., API keys, passwords) must be masked in audit logs."""
    tool_call = ToolCall(
        id="call_999",
        function=FunctionCall(
            name="database:query",
            arguments=json.dumps({
                "query": "SELECT *",
                "api_key": "sk-secret-live-credential-12345",
                "db_password": "super-secret-password",
            }),
        ),
    )
    decision = governor.authorize_tool_call(tool_call)
    assert decision.is_allowed is True
    sanitized = decision.sanitized_arguments
    assert sanitized["api_key"] == "[REDACTED_BY_FLUXMATRIX_MCP_GOVERNOR]"
    assert sanitized["db_password"] == "[REDACTED_BY_FLUXMATRIX_MCP_GOVERNOR]"

