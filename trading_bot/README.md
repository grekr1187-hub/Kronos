# Kronos Trading Bot — starter

This folder is a **paper-trading starter**, not a profitable strategy guarantee. It does not submit real orders and does not need exchange API keys.

## Intended architecture
- Fetch public OHLCV candles from a supported exchange.
- Load the Kronos tokenizer and predictor for forecasts.
- Turn forecasts into candidate signals only after fees/slippage and risk filters.
- Simulate entries/exits and record every decision to a local ledger.
- Require a separate, explicit live-trading implementation and credentials before any real order execution.

## Important
Kronos is a time-series forecasting model, not an exchange, broker, or ready-made trading bot. Forecasts are uncertain and must be evaluated out-of-sample. The upstream README itself warns that its example backtest is simplified and not production-ready. Never treat model output as a promise of profit.

## Next implementation steps
1. Select exchange and market (spot vs futures).
2. Validate data timestamps, candle intervals, fees, slippage, and forecast horizon.
3. Backtest using walk-forward splits; include costs and compare to buy-and-hold.
4. Run paper trading and monitor drawdown, order logic, and outages.
5. Only then consider live trading with restricted API permissions, no withdrawal permission, position limits, a daily loss limit, and a kill switch.

The parent repository is the Kronos forecasting model. Keep exchange execution isolated from model inference so forecasts cannot directly bypass risk controls.
