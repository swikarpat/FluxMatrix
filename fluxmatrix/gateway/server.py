"""FastAPI Application Factory and Server Lifecycle for FluxMatrix Gateway."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fluxmatrix.common.config import settings
from fluxmatrix.common.logging import get_logger, setup_logging
from fluxmatrix.gateway.middleware import DistributedTracingMiddleware
from fluxmatrix.gateway.routes import init_dependencies, router

logger = get_logger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle gateway lifecycle startup and shutdown routines."""
    # 1. Initialize structured logging
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file_path=settings.log_file_path if settings.log_file_enabled else None,
    )

    logger.info(
        "Starting FluxMatrix AI Inference Gateway",
        environment=settings.environment,
        engine_mode=settings.engine_mode,
        fast_tier=settings.fast_tier_model,
        heavy_tier=settings.heavy_tier_model,
        host=settings.host,
        port=settings.port,
    )

    # 2. Initialize engines and routing policy
    init_dependencies(settings)

    yield

    # 3. Graceful shutdown
    logger.info("FluxMatrix Gateway shutting down cleanly")


def create_app() -> FastAPI:
    """Create and configure the production FastAPI application."""
    app = FastAPI(
        title="FluxMatrix AI Inference Gateway",
        description="High-Throughput Distributed AI Inference Gateway & Execution Control Plane",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS for web dashboards, developer portals, and UI clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach Distributed Tracing Middleware (injects correlation IDs and measures latency)
    app.add_middleware(DistributedTracingMiddleware)

    # Register API routes
    app.include_router(router)

    return app


app = create_app()


def run():
    """CLI entrypoint to launch the server."""
    uvicorn.run(
        "fluxmatrix.gateway.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # Handled directly by structlog
    )


if __name__ == "__main__":
    run()

