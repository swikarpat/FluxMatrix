"""Production vLLM Inference Engine Client for FluxMatrix.

Connects to self-hosted vLLM instances running with Continuous Batching,
PagedAttention, and OpenAI-compatible API schemas.
"""

import json
import time
from typing import Any, AsyncIterator, Dict, Optional
import httpx
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
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from fluxmatrix.engines.base import BaseInferenceEngine

logger = get_logger("vllm_engine")


class VLLMInferenceEngine(BaseInferenceEngine):
    """Async client communicating with a downstream vLLM engine instance."""

    def __init__(
        self,
        engine_id: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 60.0,
    ):
        super().__init__(engine_id=engine_id)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
        model_override: Optional[str] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream SSE chunks from the vLLM instance with latency telemetry."""
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True
        if model_override:
            payload["model"] = model_override

        ACTIVE_STREAMS.labels(engine=self.engine_id).inc()
        start_time = time.perf_counter()
        first_token_received = False
        last_token_time = start_time
        token_count = 0

        try:
            async with self.client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_dict = json.loads(data_str)
                            chunk = ChatCompletionChunk.model_validate(chunk_dict)

                            now = time.perf_counter()
                            if not first_token_received:
                                first_token_received = True
                                ttft = now - start_time
                                TTFT_SECONDS.labels(engine=self.engine_id, routing_tier="vllm").observe(ttft)
                            else:
                                tpot = now - last_token_time
                                TPOT_SECONDS.labels(engine=self.engine_id).observe(tpot)

                            last_token_time = now
                            token_count += 1
                            yield chunk

                        except Exception as parse_err:
                            logger.error("Failed to parse vLLM streaming chunk", error=str(parse_err), line=line)

            COMPLETION_TOKENS_TOTAL.labels(engine=self.engine_id).inc(token_count)

        except httpx.HTTPStatusError as http_err:
            logger.error("vLLM upstream HTTP error", status_code=http_err.response.status_code, error=str(http_err))
            raise
        except Exception as e:
            logger.error("vLLM request connection error", error=str(e))
            raise
        finally:
            ACTIVE_STREAMS.labels(engine=self.engine_id).dec()

    async def generate_complete(
        self,
        request: ChatCompletionRequest,
        model_override: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Produce non-streaming response from vLLM."""
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = False
        if model_override:
            payload["model"] = model_override

        start_time = time.perf_counter()
        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        duration = time.perf_counter() - start_time

        data = resp.json()
        parsed = ChatCompletionResponse.model_validate(data)
        parsed.fluxmatrix_engine = self.engine_id
        parsed.fluxmatrix_ttft_ms = duration * 1000.0

        if parsed.usage:
            PROMPT_TOKENS_TOTAL.labels(engine=self.engine_id).inc(parsed.usage.prompt_tokens)
            COMPLETION_TOKENS_TOTAL.labels(engine=self.engine_id).inc(parsed.usage.completion_tokens)

        return parsed

    async def health(self) -> bool:
        """Query vLLM health endpoint."""
        try:
            resp = await self.client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "backend": "vLLM",
            "base_url": self.base_url,
        }

    async def close(self):
        await self.client.aclose()

