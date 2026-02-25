"""Application configuration with environment variable support."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "QuantumFlow"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://quantumflow:quantumflow@localhost:5432/quantumflow"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Market Data
    POLYGON_API_KEY: str = ""
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # Auth
    SECRET_KEY: str = "quantumflow-dev-secret-change-in-production"
    API_KEY_HEADER: str = "X-API-Key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 15
    WS_MAX_CONNECTIONS: int = 100

    # Cache TTLs (seconds)
    OPTION_CHAIN_CACHE_TTL: int = 60
    VOLATILITY_SURFACE_CACHE_TTL: int = 300
    PREDICTION_CACHE_TTL: int = 300

    # Risk
    MAX_POSITION_SIZE: float = 0.05
    MAX_PORTFOLIO_DELTA: float = 500.0
    MARGIN_CALL_THRESHOLD: float = 0.8

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    model_config = {"env_file": ".env", "env_prefix": "QF_", "case_sensitive": True}


@lru_cache
def get_settings() -> Settings:
    return Settings()
