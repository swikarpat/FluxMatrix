"""Unit tests for the Semantic & Complexity-based Dynamic Router."""

import pytest
from fluxmatrix.common.config import GatewaySettings
from fluxmatrix.core.models import ChatCompletionRequest, ChatMessage, MessageRole, RoutingTier
from fluxmatrix.core.router import SemanticRouter


@pytest.fixture
def test_settings():
    return GatewaySettings(
        complexity_threshold=0.55,
        fast_tier_model="qwen-2.5-7b-instruct-awq",
        heavy_tier_model="deepseek-r1-70b-instruct",
        long_context_token_threshold=1000,
    )


@pytest.fixture
def router(test_settings):
    return SemanticRouter(settings=test_settings)


def test_simple_prompt_routes_to_fast_tier(router):
    """Simple extraction / summarization prompts should route to the fast quantized tier."""
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role=MessageRole.USER, content="Summarize this brief note: meetings are moved to 3pm."),
        ]
    )
    decision = router.route_request(req)
    assert decision.tier == RoutingTier.FAST_QUANTIZED
    assert decision.selected_model == "qwen-2.5-7b-instruct-awq"
    assert decision.complexity_score < 0.55


def test_complex_code_and_reasoning_routes_to_heavy_tier(router):
    """Deep code optimization and chain-of-thought prompts should route to the heavy reasoning tier."""
    complex_prompt = (
        "Prove the time complexity of the following lock-free queue algorithm step-by-step. "
        "Here is the implementation:\n"
        "```python\n"
        "class LockFreeQueue:\n"
        "    def __init__(self):\n"
        "        self.head = AtomicReference(None)\n"
        "```\n"
        "Analyze the cache coherency and derive O(1) bounds."
    )
    req = ChatCompletionRequest(
        messages=[ChatMessage(role=MessageRole.USER, content=complex_prompt)]
    )
    decision = router.route_request(req)
    assert decision.tier == RoutingTier.REASONING_HEAVY
    assert decision.selected_model == "deepseek-r1-70b-instruct"
    assert decision.complexity_score >= 0.55


def test_client_override_bypasses_classifier(router):
    """Explicit client routing_override should be honored immediately."""
    req = ChatCompletionRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
        routing_override="custom-finetuned-llama3",
    )
    decision = router.route_request(req)
    assert decision.tier == RoutingTier.DIRECT_OVERRIDE
    assert decision.selected_model == "custom-finetuned-llama3"

