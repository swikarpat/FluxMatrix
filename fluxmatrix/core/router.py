"""Semantic and Complexity-based Dynamic Router for FluxMatrix.

Routes queries intelligently between:
1. Fast Quantized Tier (e.g., Qwen-2.5-7B-AWQ) for low-latency, high-throughput queries.
2. Heavy Reasoning Tier (e.g., DeepSeek-R1-70B) for complex analysis, deep code, and multi-step synthesis.
"""

from typing import Optional
from fluxmatrix.common.config import GatewaySettings
from fluxmatrix.common.logging import get_logger
from fluxmatrix.common.telemetry import ROUTING_DECISIONS_TOTAL
from fluxmatrix.core.models import (
    ChatCompletionRequest,
    RoutingDecision,
    RoutingTier,
)
from fluxmatrix.core.token_estimator import PromptAnalyzer

logger = get_logger("router")


class SemanticRouter:
    """Dynamic routing engine optimizing cost, TTFT, and model capacity."""

    def __init__(self, settings: GatewaySettings, analyzer: Optional[PromptAnalyzer] = None):
        self.settings = settings
        self.analyzer = analyzer or PromptAnalyzer()

    def route_request(self, request: ChatCompletionRequest) -> RoutingDecision:
        """Analyze request characteristics and determine the optimal backend inference engine."""

        # 1. Check for manual client-specified override
        if request.routing_override:
            decision = RoutingDecision(
                tier=RoutingTier.DIRECT_OVERRIDE,
                selected_model=request.routing_override,
                complexity_score=1.0,
                estimated_prompt_tokens=self.analyzer.estimate_tokens(request.messages),
                requires_tools=bool(request.tools),
                reason=f"Client specified explicit routing override: {request.routing_override}",
            )
            ROUTING_DECISIONS_TOTAL.labels(tier="direct_override", reason="client_override").inc()
            logger.info("Direct model override routed", model=decision.selected_model)
            return decision

        # 2. Estimate token count and complexity
        estimated_tokens = self.analyzer.estimate_tokens(request.messages)
        complexity_score, signals = self.analyzer.compute_complexity_score(request.messages)
        has_tools = bool(request.tools)

        # 3. Decision Logic:
        # Route to Heavy Tier if complexity score exceeds threshold OR if complex tool definitions present
        if complexity_score >= self.settings.complexity_threshold:
            selected_model = self.settings.heavy_tier_model
            tier = RoutingTier.REASONING_HEAVY
            reason = f"High complexity score ({complexity_score:.2f} >= {self.settings.complexity_threshold:.2f}); signals: {', '.join(signals)}"
        elif estimated_tokens >= self.settings.long_context_token_threshold:
            selected_model = self.settings.heavy_tier_model
            tier = RoutingTier.REASONING_HEAVY
            reason = f"Long context ({estimated_tokens} tokens >= {self.settings.long_context_token_threshold})"
        else:
            selected_model = self.settings.fast_tier_model
            tier = RoutingTier.FAST_QUANTIZED
            reason = f"Low complexity score ({complexity_score:.2f} < {self.settings.complexity_threshold:.2f}); signals: {', '.join(signals)}"

        ROUTING_DECISIONS_TOTAL.labels(tier=tier.value, reason="complexity_scoring").inc()

        decision = RoutingDecision(
            tier=tier,
            selected_model=selected_model,
            complexity_score=complexity_score,
            estimated_prompt_tokens=estimated_tokens,
            requires_tools=has_tools,
            reason=reason,
        )

        logger.info(
            "Routing decision computed",
            tier=decision.tier.value,
            selected_model=decision.selected_model,
            complexity_score=decision.complexity_score,
            tokens=decision.estimated_prompt_tokens,
            reason=decision.reason,
        )

        return decision

