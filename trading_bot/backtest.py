"""Walk-forward Kronos backtest using historical candles only; never places orders."""
import argparse
from pathlib import Path

import ccxt
import pandas as pd

from trading_bot.backtest_engine import simulate


def fetch_history(exchange_id, symbol, timeframe, limit):
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown CCXT exchange: {exchange_id}")
    exchange = exchange_class({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("No historical candles returned.")
    frame = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["ms"], unit="ms", utc=True)
    # Drop the currently forming candle to avoid using incomplete data.
    return frame.iloc[:-1].reset_index(drop=True)


def kronos_forecasts(candles, timeframe, lookback, forecast_candles):
    # Model dependencies are loaded only when a real model backtest is requested.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Kronos, KronosTokenizer, KronosPredictor

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    units = {"m": "min", "h": "h", "d": "D", "w": "W"}
    if len(timeframe) < 2 or timeframe[-1] not in units or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    freq = pd.tseries.frequencies.to_offset(f"{timeframe[:-1]}{units[timeframe[-1]]}")
    signals = [None] * len(candles)
    for i in range(63, len(candles) - 1):
        hist = candles.iloc[max(0, i - lookback + 1):i + 1].copy().reset_index(drop=True)
        hist_ts = hist["timestamp"].dt.tz_convert(None)
        future_ts = pd.Series(pd.date_range(start=hist_ts.iloc[-1] + freq, periods=forecast_candles, freq=freq))
        features = hist[["open", "high", "low", "close", "volume"]].astype(float)
        forecast = predictor.predict(
            df=features, x_timestamp=hist_ts, y_timestamp=future_ts,
            pred_len=forecast_candles, T=1.0, top_p=0.9, sample_count=3
        )
        if forecast.empty or "close" not in forecast:
            raise RuntimeError(f"Empty Kronos forecast at row {i}")
        current_close = float(hist["close"].iloc[-1])
        predicted_close = float(forecast["close"].iloc[-1])
        if current_close <= 0 or predicted_close <= 0:
            raise RuntimeError(f"Invalid Kronos forecast at row {i}")
        signals[i] = predicted_close / current_close - 1
        if (i - 62) % 25 == 0:
            print(f"Generated {i - 62} / {len(candles) - 64} walk-forward forecasts", flush=True)
    return signals


def main():
    parser = argparse.ArgumentParser(description="Historical Kronos paper backtest; no real orders.")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--limit", type=int, default=200, help="Number of recent candles to fetch (65-1000).")
    parser.add_argument("--lookback", type=int, default=128)
    parser.add_argument("--forecast-candles", type=int, default=4)
    parser.add_argument("--starting-cash", type=float, default=1000.0)
    parser.add_argument("--output-dir", default="trading_bot/backtest_results")
    args = parser.parse_args()
    if not 65 <= args.limit <= 1000:
        parser.error("--limit must be between 65 and 1000")
    if not 64 <= args.lookback <= 512:
        parser.error("--lookback must be between 64 and 512")
    if args.forecast_candles < 1:
        parser.error("--forecast-candles must be at least 1")

    candles = fetch_history(args.exchange, args.symbol, args.timeframe, args.limit)
    if len(candles) < 65:
        raise RuntimeError("Need at least 65 completed historical candles.")
    signals = kronos_forecasts(candles, args.timeframe, args.lookback, args.forecast_candles)
    metrics, trades, equity = simulate(candles, signals, starting_cash=args.starting_cash)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    print("\nWALK-FORWARD BACKTEST RESULTS (historical simulation only)")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
