# Kronos Trading Bot — paper-trading starter

**This is simulation only.** The code cannot submit exchange orders and does not load exchange API keys. It does not promise or guarantee profits.

## Run locally

Use Python 3.10+ from the repository root (the folder containing `model.py`):

```bash
python -m venv .venv
# activate the virtual environment, then:
pip install -r trading_bot/requirements.txt
cp trading_bot/.env.example trading_bot/.env
python trading_bot/paper_bot.py
```

The first run downloads the open Kronos-small model and tokenizer from Hugging Face, so it needs internet access and enough RAM/disk space. The bot uses public OHLCV/ticker endpoints only; it needs no exchange account, API key, or deposit.

## What it does

- Reads completed OHLCV candles and asks Kronos for a forecast.
- Makes at most one forecast-based entry decision per completed candle.
- Simulates spot-style buy/sell fills with configurable fee and slippage assumptions.
- Checks simulated stop-loss/take-profit exits on each polling cycle.
- Saves a simulated ledger to `paper_trades.csv` and state to `paper_state.csv` in this folder.
- Enforces position-size limits and rejects any `TRADING_MODE` other than `paper`.

## Configuration

Edit `trading_bot/.env`. Defaults are BTC/USDT on 15-minute candles, a 1,000 USDT virtual starting balance, a 10% maximum position, and no leverage. The configured fee/slippage values are assumptions, not live exchange quotes. Delete the two CSV files to reset the simulation.

## Historical walk-forward backtest (simulation only)

From the repository root, after installing `trading_bot/requirements.txt`:

```bash
python -m trading_bot.backtest --exchange binance --symbol BTC/USDT --timeframe 15m --limit 100 --lookback 64 --forecast-candles 4
```

The backtest fetches recent public candles, discards the unfinished candle, generates each forecast only from candles available at that point, and simulates entry on the next candle's open. It uses fee/slippage assumptions, stop-loss/take-profit rules and pessimistically assumes the stop is hit first if both stop and target are touched within the same candle. Metrics and trade/equity CSV files are written under `trading_bot/backtest_results/`. This initial run covers only a short recent sample and is a smoke test, not enough evidence to establish profitability.

A GitHub Actions workflow also runs a short 100-candle backtest on changes to the backtest code and uploads results as an artifact. It does not deploy the bot or place real orders.

## Before considering real trading

1. Verify that the selected venue and product are legally available to you in your jurisdiction.
2. Run a walk-forward historical backtest including realistic fees, slippage, latency and missed fills.
3. Paper-trade long enough to cover different market regimes; track net return, maximum drawdown, turnover and benchmark performance.
4. Review the model outputs and code independently. Forecasts are uncertain; past or simulated performance does not predict future results.

Kronos is a financial time-series forecasting model, not a complete trading strategy or a broker. Its upstream documentation explicitly describes the example backtest as simplified and not production-ready. Do not use this starter with real funds.
