"""
QuantumFlow - Algorithmic Options Trading Terminal

FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import get_settings
from .core.logging import setup_logging, get_logger
from .api.v1.routes import router, market_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    setup_logging(debug=settings.DEBUG)
    log = get_logger("quantumflow")

    log.info("starting_quantumflow", version=settings.APP_VERSION)

    # Initialize services
    await market_data.initialize()

    yield

    # Cleanup
    await market_data.close()
    log.info("quantumflow_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="QuantumFlow API",
        description="Algorithmic Options Trading Terminal - Quantitative Engine",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
