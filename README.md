# QuantumFlow - Algorithmic Options Trading Terminal

Production-grade algorithmic options trading platform with institutional-quality quantitative backend and neo-brutalist frontend terminal.

## Architecture

```
options-trading-algorithm/
├── backend/                        # Python FastAPI quantitative engine
│   ├── app/
│   │   ├── api/v1/routes.py        # REST + WebSocket endpoints
│   │   ├── core/                   # Config, logging
│   │   ├── models/options.py       # Pydantic schemas (30+ models)
│   │   └── services/
│   │       ├── data/               # Market data ingestion + mock generator
│   │       ├── pricing/            # Black-Scholes, Greeks, vol surface, strategy analysis
│   │       ├── ml/                 # XGBoost regime classifier, feature engineering
│   │       └── execution/          # Event-driven backtester
│   ├── tests/                      # Pricing engine test suite
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                       # Next.js 16 TypeScript terminal UI
│   ├── src/
│   │   ├── app/                    # App router pages + globals.css
│   │   ├── components/
│   │   │   ├── market/             # Market Pulse header (live prices)
│   │   │   ├── options/            # Option chain ladder view
│   │   │   ├── strategy/           # Payoff Architect + time slider
│   │   │   ├── volatility/         # IV surface heatmap
│   │   │   ├── portfolio/          # Position monitor + Greeks gauges
│   │   │   ├── layout/             # Mission Control grid layout
│   │   │   └── ui/                 # shadcn/ui primitives
│   │   ├── stores/                 # Zustand state management
│   │   ├── hooks/                  # WebSocket stream, TanStack Query
│   │   └── lib/                    # API client, types, utils
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml              # Full stack: API + UI + TimescaleDB + Redis
└── src/                            # Legacy analysis scripts
```

## Quick Start

### 1. Backend (API)

```bash
# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the API server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at http://localhost:8000/docs

### 2. Frontend (Terminal UI)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### 3. Docker (Full Stack)

```bash
docker-compose up -d
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- TimescaleDB: localhost:5432
- Redis: localhost:6379

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/symbols` | Available symbols with prices |
| GET | `/api/v1/options/chain?symbol=SPY` | Full option chain with Greeks |
| GET | `/api/v1/quote/{symbol}` | Current quote |
| GET | `/api/v1/historical/{symbol}?days=252` | OHLCV history |
| POST | `/api/v1/strategy/analyze` | Multi-leg payoff analysis |
| GET | `/api/v1/strategy/templates?symbol=SPY` | Pre-built strategies |
| GET | `/api/v1/greeks/portfolio` | Aggregate portfolio Greeks |
| GET | `/api/v1/volatility/surface?symbol=SPY` | IV surface data |
| POST | `/api/v1/predict/volatility` | ML regime prediction |
| POST | `/api/v1/backtest/run` | Execute backtest |
| WS | `/api/v1/stream/prices` | Real-time price stream |

## Quantitative Engine

### Black-Scholes Pricing
- Numba JIT-compiled for performance
- European call/put pricing with full Greeks (delta, gamma, theta, vega, rho)
- Newton-Raphson implied volatility solver
- Cox-Ross-Rubinstein binomial tree for American options

### Volatility Surface
- IV surface construction from option chain data
- IV rank and percentile calculation
- Skew metrics (25-delta risk reversal, butterfly)
- Term structure analysis (contango/backwardation detection)

### ML Pipeline
- XGBoost volatility regime classifier (Low Vol / High Vol / Crisis)
- 30+ features: RSI, MACD, Bollinger, realized vol, Parkinson, Garman-Klass
- Walk-forward time-series cross-validation

### Event-Driven Backtester
- Fills at BID (sells) and ASK (buys), never mid-price
- Configurable slippage and commissions
- Margin tracking and stop-loss/take-profit
- Performance metrics: Sharpe, Sortino, max drawdown, win rate

## Frontend Design

"Cyber-Industrial Terminal" aesthetic:
- Dark void background (#0a0a0a) with zinc-900 panels
- JetBrains Mono for all numerical data (non-negotiable for alignment)
- Space Grotesk headings, uppercase with tracking
- Sharp corners (border-radius: 2px max)
- 1px borders everywhere (visible grid structure)
- Framer Motion with easeOutQuart, 0.15s max duration
- Resizable panel layout (Mission Control grid)

## Strategy Example: Iron Condor

```bash
curl -X POST http://localhost:8000/api/v1/strategy/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "legs": [
      {"option_type":"put","strike":565,"expiry":"2026-03-20","side":"buy","quantity":1,"premium":1.20},
      {"option_type":"put","strike":575,"expiry":"2026-03-20","side":"sell","quantity":1,"premium":2.50},
      {"option_type":"call","strike":600,"expiry":"2026-03-20","side":"sell","quantity":1,"premium":2.30},
      {"option_type":"call","strike":610,"expiry":"2026-03-20","side":"buy","quantity":1,"premium":1.10}
    ]
  }'
```

## Tests

```bash
cd backend
NUMBA_DISABLE_JIT=1 python -m pytest tests/ -v
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.9+, FastAPI, asyncio, WebSockets |
| Quant | NumPy, SciPy, Numba, pandas |
| ML | XGBoost, scikit-learn |
| Frontend | Next.js 16, TypeScript, Tailwind CSS |
| UI | shadcn/ui, Recharts, Framer Motion, react-resizable-panels |
| State | Zustand + TanStack Query |
| Infra | Docker, TimescaleDB, Redis |

## Risk Disclaimer

This platform is for educational and research purposes only. Options trading involves significant risk of loss. Always paper trade before using real capital.
