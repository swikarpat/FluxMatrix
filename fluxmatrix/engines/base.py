"""Abstract Base Class for FluxMatrix Inference Engines."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
from fluxmatrix.core.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class BaseInferenceEngine(ABC):
    """Protocol for interacting with downstream inference servers."""

    def __init__(self, engine_id: str):
        self.engine_id = engine_id

    @abstractmethod
    async def generate_stream(
        self,
        request: ChatCompletionRequest,
        model_override: Optional[str] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream token chunks asynchronously via Server-Sent Events (SSE)."""
        pass

    @abstractmethod
    async def generate_complete(
        self,
        request: ChatCompletionRequest,
        model_override: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Produce a complete non-streaming chat completion."""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Check if the serving backend is ready and responding."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics (e.g. active connections, requests completed)."""
        pass

