"""High-fidelity simulated inference engine for FluxMatrix.

Provides realistic streaming tokens, prefill TTFT delays, and decode TPOT cadence.
Enables full development, load-testing, and CI on Apple Silicon (MacBook Pro M5 Pro)
without needing immediate cloud GPU instances.
"""

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional
from fluxmatrix.common.logging import get_logger
from fluxmatrix.common.telemetry import (
    ACTIVE_STREAMS,
    COMPLETION_TOKENS_TOTAL,
    PROMPT_TOKENS_TOTAL,
    TPOT_SECONDS,
    TTFT_SECONDS,
)
from fluxmatrix.core.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    MessageRole,
    StreamChunkDelta,
    UsageInfo,
)
from fluxmatrix.engines.base import BaseInferenceEngine

logger = get_logger("mock_engine")


SAMPLE_RESPONSES = [
    "FluxMatrix has analyzed your prompt using its low-latency control plane. ",
    "The query was evaluated against dynamic complexity thresholds and routed to the optimal inference worker. ",
    "Continuous batching and prefix caching ensure minimal Time to First Token (TTFT). ",
    "All agentic tool executions are audited under Model Context Protocol (MCP) governance.",
]


class MockInferenceEngine(BaseInferenceEngine):
    """Simulated inference worker reproducing realistic distributed serving dynamics."""

    def __init__(
        self,
        engine_id: str = "mock-vllm-worker-0",
        base_ttft_ms: float = 25.0,
        base_tpot_ms: float = 12.0,
    ):
        super().__init__(engine_id=engine_id)
        self.base_ttft_ms = base_ttft_ms
        self.base_tpot_ms = base_tpot_ms
        self.total_served = 0

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
        model_override: Optional[str] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream simulated token deltas with realistic latency dynamics."""
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        model = model_override or request.model or "fluxmatrix-mock-model"
        created_time = int(time.time())

        # Account for input prompt tokens
        approx_prompt_tokens = sum(len(m.content or "") // 4 for m in request.messages) + 8
        PROMPT_TOKENS_TOTAL.labels(engine=self.engine_id).inc(approx_prompt_tokens)

        # Track active stream concurrency
        ACTIVE_STREAMS.labels(engine=self.engine_id).inc()
        start_time = time.perf_counter()

        # Choose response tokens to emit
        tokens_to_emit: List[str] = []
        for sentence in SAMPLE_RESPONSES:
            for word in sentence.split(" "):
                if word:
                    tokens_to_emit.append(word + " ")

        if request.max_tokens:
            tokens_to_emit = tokens_to_emit[: request.max_tokens]

        try:
            # 1. Simulate Prefill / TTFT Phase
            # Higher TTFT for longer prompts
            prefill_delay = (self.base_ttft_ms + (approx_prompt_tokens * 0.015)) / 1000.0
            await asyncio.sleep(prefill_delay)

            ttft_measured = time.perf_counter() - start_time
            TTFT_SECONDS.labels(engine=self.engine_id, routing_tier="dynamic").observe(ttft_measured)

            # Emit initial chunk (role declaration)
            yield ChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=StreamChunkDelta(role=MessageRole.ASSISTANT),
                        finish_reason=None,
                    )
                ],
            )

            # 2. Simulate Token Generation / Decode Phase (TPOT)
            last_token_time = time.perf_counter()
            for idx, token in enumerate(tokens_to_emit):
                # Simulated token inter-arrival time
                step_start = time.perf_counter()
                await asyncio.sleep(self.base_tpot_ms / 1000.0)

                tpot_measured = time.perf_counter() - step_start
                TPOT_SECONDS.labels(engine=self.engine_id).observe(tpot_measured)

                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model=model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=StreamChunkDelta(content=token),
                            finish_reason=None,
                        )
                    ],
                )

            # 3. Emit Final Stop Chunk with usage stats
            COMPLETION_TOKENS_TOTAL.labels(engine=self.engine_id).inc(len(tokens_to_emit))
            yield ChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=StreamChunkDelta(),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=approx_prompt_tokens,
                    completion_tokens=len(tokens_to_emit),
                    total_tokens=approx_prompt_tokens + len(tokens_to_emit),
                ),
            )

            self.total_served += 1

        except asyncio.CancelledError:
            logger.warning("Stream connection cancelled by client (backpressure/disconnect)", completion_id=completion_id)
            raise
        finally:
            ACTIVE_STREAMS.labels(engine=self.engine_id).dec()

    async def generate_complete(
        self,
        request: ChatCompletionRequest,
        model_override: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Produce a complete non-streaming chat response."""
        chunks: List[str] = []
        async for chunk in self.generate_stream(request, model_override):
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        full_content = "".join(chunks)
        approx_prompt_tokens = sum(len(m.content or "") // 4 for m in request.messages) + 8
        completion_tokens = len(full_content.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_override or request.model or "fluxmatrix-mock-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=full_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=approx_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=approx_prompt_tokens + completion_tokens,
            ),
            fluxmatrix_engine=self.engine_id,
        )

    async def health(self) -> bool:
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "status": "healthy",
            "type": "high_fidelity_simulator",
            "total_requests_served": self.total_served,
        }

