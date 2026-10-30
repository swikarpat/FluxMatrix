"""Integration tests for FluxMatrix Gateway: HTTP endpoints, SSE streaming, and tracing."""

import json
import pytest
from httpx import ASGITransport, AsyncClient
from fluxmatrix.gateway.server import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_health_endpoint(app):
    """Verify /health reports healthy status and engine metadata."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["gateway"] == "online"
        assert "engine" in data


@pytest.mark.asyncio
async def test_models_endpoint(app):
    """Verify /v1/models returns active tiers and SLOs."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 2


@pytest.mark.asyncio
async def test_metrics_endpoint(app):
    """Verify /metrics returns standard Prometheus scrape format."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert "fluxmatrix_requests_total" in resp.text


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(app):
    """Verify standard non-streaming /v1/chat/completions response."""
    payload = {
        "messages": [
            {"role": "user", "content": "Explain continuous batching in distributed inference."}
        ],
        "stream": False,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        # Verify distributed tracing headers
        assert "x-trace-id" in resp.headers
        assert "x-request-id" in resp.headers
        assert "x-response-time-ms" in resp.headers

        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_chat_completions_streaming_sse(app):
    """Verify SSE streaming delivers tokens with proper chunk structures and [DONE] signal."""
    payload = {
        "messages": [
            {"role": "user", "content": "Stream token generation benchmark."}
        ],
        "stream": True,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            assert "x-fluxmatrix-tier" in resp.headers

            received_chunks = []
            done_received = False

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    content_str = line[6:].strip()
                    if content_str == "[DONE]":
                        done_received = True
                        break

                    chunk_data = json.loads(content_str)
                    assert chunk_data["object"] == "chat.completion.chunk"
                    received_chunks.append(chunk_data)

            assert done_received is True
            assert len(received_chunks) > 0
            # Ensure at least one token chunk contained generated content
            has_content = any(
                c["choices"][0]["delta"].get("content") for c in received_chunks if c["choices"]
            )
            assert has_content is True

