import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd
from dotenv import load_dotenv

# Allow both `python trading_bot/paper_bot.py` and module execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Kronos, KronosTokenizer, KronosPredictor

load_dotenv(Path(__file__).with_name(".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kronos-paper")

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
LOOKBACK = min(512, max(64, int(os.getenv("LOOKBACK_CANDLES", "256"))))
FORECAST_CANDLES = max(1, int(os.getenv("FORECAST_CANDLES", "4")))
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "1000"))
RISK_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5")) / 100
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "10")) / 100
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "1.0")) / 100
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.5")) / 100
FEE_PCT = float(os.getenv("FEE_RATE_PCT", "0.1")) / 100
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.05")) / 100
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "0.25")) / 100
POLL_SECONDS = max(10, int(os.getenv("POLL_SECONDS", "60")))

if os.getenv("TRADING_MODE", "paper").lower() != "paper":
    raise RuntimeError("This starter only supports TRADING_MODE=paper; live execution is intentionally disabled.")
if STARTING_BALANCE <= 0:
    raise ValueError("STARTING_BALANCE must be greater than zero.")
if not 0 < RISK_PCT <= 0.02:
    raise ValueError("RISK_PER_TRADE_PCT must be greater than 0 and no more than 2.")
if not 0 < MAX_POSITION_PCT <= 0.25:
    raise ValueError("MAX_POSITION_PCT must be greater than 0 and no more than 25.")
if not 0 < STOP_LOSS_PCT < 0.10:
    raise ValueError("STOP_LOSS_PCT must be greater than 0 and less than 10.")
if not 0 < TAKE_PROFIT_PCT < 0.20:
    raise ValueError("TAKE_PROFIT_PCT must be greater than 0 and less than 20.")
if min(FEE_PCT, SLIPPAGE_PCT, MIN_EDGE_PCT) < 0:
    raise ValueError("Fees, slippage, and minimum edge cannot be negative.")

OUT = Path(__file__).with_name("paper_trades.csv")
STATE = Path(__file__).with_name("paper_state.csv")


@dataclass
class Position:
    qty: float
    entry: float
    stop: float
    target: float


def load_state():
    if not STATE.exists():
        return STARTING_BALANCE, None, None

    state = pd.read_csv(STATE)
    if state.empty:
        return STARTING_BALANCE, None, None
    row = state.iloc[-1]
    position = None
    qty = float(row.get("qty", 0.0))
    if qty > 0:
        position = Position(
            qty=qty,
            entry=float(row["entry"]),
            stop=float(row["stop"]),
            target=float(row["target"]),
        )
    last_candle = row.get("last_signal_candle")
    if pd.isna(last_candle) or str(last_candle).strip() == "":
        last_candle = None
    else:
        last_candle = str(last_candle)
    return float(row["cash"]), position, last_candle


def save_state(cash, position, last_signal_candle):
    pd.DataFrame([{
        "cash": cash,
        "qty": position.qty if position else 0.0,
        "entry": position.entry if position else 0.0,
        "stop": position.stop if position else 0.0,
        "target": position.target if position else 0.0,
        "last_signal_candle": last_signal_candle or "",
    }]).to_csv(STATE, index=False)


def log_trade(action, price, qty, cash, reason):
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": SYMBOL,
        "action": action,
        "price": price,
        "qty": qty,
        "cash_after": cash,
        "reason": reason,
    }])
    row.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def get_model():
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model, tokenizer, max_context=512)


def timeframe_frequency(timeframe):
    # CCXT commonly uses values such as 1m, 15m, 1h, 4h, 1d.
    units = {"m": "min", "h": "h", "d": "D", "w": "W"}
    if len(timeframe) < 2 or timeframe[-1] not in units or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe for pandas timestamps: {timeframe!r}")
    return pd.tseries.frequencies.to_offset(f"{timeframe[:-1]}{units[timeframe[-1]]}")


def get_candles(exchange):
    rows = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LOOKBACK + 2)
    if not rows:
        raise RuntimeError("Exchange returned no OHLCV candles.")
    df = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.tz_localize(None)
    # Ignore the current, potentially incomplete candle.
    df = df.iloc[:-1].reset_index(drop=True)
    if len(df) < 64:
        raise RuntimeError("Not enough completed candles to forecast.")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise RuntimeError("Candle data contains invalid close prices.")
    return df


def predict_return(predictor, candles):
    hist = candles.iloc[-min(LOOKBACK, len(candles), 512):].copy().reset_index(drop=True)
    last_ts = hist["timestamps"].iloc[-1]
    freq = timeframe_frequency(TIMEFRAME)
    future_ts = pd.Series(pd.date_range(start=last_ts + freq, periods=FORECAST_CANDLES, freq=freq))
    hist_ts = hist["timestamps"]
    features = hist[["open", "high", "low", "close", "volume"]].astype(float)
    forecast = predictor.predict(
        df=features,
        x_timestamp=hist_ts,
        y_timestamp=future_ts,
        pred_len=FORECAST_CANDLES,
        T=1.0,
        top_p=0.9,
        sample_count=3,
    )
    if forecast.empty or "close" not in forecast:
        raise RuntimeError("Kronos returned an empty forecast or no close column.")
    predicted_close = float(forecast["close"].iloc[-1])
    current_close = float(hist["close"].iloc[-1])
    if predicted_close <= 0 or current_close <= 0:
        raise RuntimeError("Forecast or market close is not a positive price.")
    return predicted_close / current_close - 1.0, current_close


def main():
    exchange_class = getattr(ccxt, EXCHANGE_ID, None)
    if exchange_class is None:
        raise ValueError(f"Unknown CCXT exchange: {EXCHANGE_ID}")
    # Public market data only: no API keys are loaded or required.
    exchange = exchange_class({"enableRateLimit": True})
    predictor = get_model()
    cash, position, last_signal_candle = load_state()
    log.info("PAPER MODE only | %s %s | cash %.2f", SYMBOL, TIMEFRAME, cash)

    while True:
        try:
            candles = get_candles(exchange)
            candle_id = pd.Timestamp(candles["timestamps"].iloc[-1]).isoformat()
            new_candle = candle_id != last_signal_candle
            edge = None
            close = float(candles["close"].iloc[-1])
            if new_candle:
                edge, close = predict_return(predictor, candles)
                last_signal_candle = candle_id

            ticker = exchange.fetch_ticker(SYMBOL)
            price = float(ticker.get("last") or close)
            if price <= 0:
                raise RuntimeError("Exchange returned an invalid ticker price.")

            exited_this_cycle = False
            # Stops and targets are checked on every poll; forecast exits only on new candles.
            if position:
                exit_reason = None
                if price <= position.stop:
                    exit_reason = "stop_loss"
                elif price >= position.target:
                    exit_reason = "take_profit"
                elif new_candle and edge is not None and edge < -MIN_EDGE_PCT:
                    exit_reason = "forecast_turn"
                if exit_reason:
                    fill = price * (1 - SLIPPAGE_PCT)
                    proceeds = position.qty * fill * (1 - FEE_PCT)
                    cash += proceeds
                    log_trade("SELL", fill, position.qty, cash, exit_reason)
                    log.info("PAPER SELL qty=%.6f price=%.2f reason=%s cash=%.2f", position.qty, fill, exit_reason, cash)
                    position = None
                    exited_this_cycle = True

            # Evaluate entry signals once per completed candle; never re-enter immediately after an exit.
            if new_candle and not exited_this_cycle and position is None and edge is not None and edge >= MIN_EDGE_PCT:
                risk_budget = cash * RISK_PCT
                stop_distance = price * STOP_LOSS_PCT
                qty_by_risk = risk_budget / stop_distance if stop_distance > 0 else 0.0
                qty_by_cap = (cash * MAX_POSITION_PCT) / price
                qty_by_cash = cash / (price * (1 + FEE_PCT + SLIPPAGE_PCT))
                qty = max(0.0, min(qty_by_risk, qty_by_cap, qty_by_cash))
                if qty > 0:
                    fill = price * (1 + SLIPPAGE_PCT)
                    cost = qty * fill * (1 + FEE_PCT)
                    if cost <= cash:
                        cash -= cost
                        position = Position(qty, fill, fill * (1 - STOP_LOSS_PCT), fill * (1 + TAKE_PROFIT_PCT))
                        log_trade("BUY", fill, qty, cash, f"forecast_edge={edge:.5f}")
                        log.info("PAPER BUY qty=%.6f price=%.2f forecast_edge=%.3f%% cash=%.2f", qty, fill, edge * 100, cash)

            save_state(cash, position, last_signal_candle)
            equity = cash + (position.qty * price if position else 0.0)
            edge_text = "not recalculated" if edge is None else f"{edge * 100:.3f}%"
            log.info("price=%.2f forecast_edge=%s position=%s equity=%.2f", price, edge_text, "open" if position else "flat", equity)
        except Exception:
            log.exception("Cycle failed; this starter cannot submit real orders.")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
