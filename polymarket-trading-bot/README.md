# Polymarket Trading Bot

A Python limit-order trading bot for [Polymarket](https://polymarket.com) built on the [`py-clob-client`](https://github.com/Polymarket/py-clob-client) SDK.

## Project Structure

```
polymarket-trading-bot/
├── strategies/
│   ├── base_strategy.py    # Abstract base class + Signal dataclass
│   ├── macd_strategy.py    # MACD crossover strategy
│   ├── rsi_strategy.py     # RSI mean-reversion strategy
│   └── cvd_strategy.py     # Cumulative Volume Delta strategy
├── backtesting/
│   ├── engine.py           # Event-driven backtest engine
│   └── data_loader.py      # CSV / synthetic OHLCV data loader
├── bot/
│   ├── trader.py           # Limit order lifecycle manager
│   └── risk_manager.py     # Position sizing & exposure controls
├── deploy/
│   ├── run_bot.py          # Live trading entry point
│   └── run_backtest.py     # Backtest entry point
├── .env.example
├── requirements.txt
└── README.md
```

## Strategies

| Strategy | Indicator | Signal Logic |
|----------|-----------|--------------|
| **MACD** | 12/26/9 EMA crossover | BUY on bullish crossover, SELL on bearish |
| **RSI** | 14-period RSI | BUY below oversold (30), SELL above overbought (70) |
| **CVD** | Cumulative Volume Delta | BUY when fast CVD MA crosses above slow, SELL on opposite cross |

All strategies return **limit orders only** — no market orders are ever sent.

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd polymarket-trading-bot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your Polymarket API credentials and private key
```

Get API credentials from [Polymarket](https://polymarket.com) after connecting your wallet.
**Use a dedicated trading wallet — never commit your private key.**

### 3. Run the backtester

```bash
# All strategies on synthetic data
python deploy/run_backtest.py

# Single strategy on your CSV data, with equity curve plots
python deploy/run_backtest.py --strategy rsi --data data/my_market.csv --plot

# MACD with custom parameters
python deploy/run_backtest.py --strategy macd --fast 8 --slow 21 --signal 5
```

Expected CSV columns: `timestamp, open, high, low, close, volume`

### 4. Run the live bot

```bash
# Find the token ID for your market from the Polymarket UI or API
python deploy/run_bot.py \
  --strategy macd \
  --token-id 0xYOUR_TOKEN_ID \
  --capital 500 \
  --order-size 10 \
  --interval 60
```

## Risk Management

`RiskManager` enforces the following rules before every order:

| Rule | Default |
|------|---------|
| Max position per token | $100 USD |
| Max total exposure | $500 USD |
| Max single order size | $50 USD |
| Min order size | $1 USD |
| Price bounds | 2¢ – 98¢ |
| Max concurrent open orders | 10 |
| Kelly fraction | 25% |

All defaults are configurable via `RiskConfig`.

## Architecture

```
Signal (strategies/) → RiskManager (bot/) → Trader (bot/) → Polymarket CLOB
                             ↑                    ↓
                       PortfolioState    fill / cancel events
```

- **Strategies** are stateless: given a DataFrame window they return a `Signal`.
- **RiskManager** approves and re-sizes the signal.
- **Trader** submits the limit order, tracks open orders, and syncs fills.
- **BacktestEngine** replays historical data, simulating passive limit-order fills.

## Testing

```bash
pytest
```

## Disclaimer

This software is for educational purposes only. Trading prediction markets involves significant financial risk. Never trade with funds you cannot afford to lose. Always backtest thoroughly before deploying real capital.
