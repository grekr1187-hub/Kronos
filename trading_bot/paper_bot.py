import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd
from dotenv import load_dotenv

# Run from the repository root so the upstream model.py is importable.
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
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))

if os.getenv("TRADING_MODE", "paper").lower() != "paper":
    raise RuntimeError("This starter only supports TRADING_MODE=paper; live execution is intentionally disabled.")

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
        return STARTING_BALANCE, None
    row = pd.read_csv(STATE).iloc[-1]
    pos = None
    if float(row["qty"]) > 0:
        pos = Position(float(row["qty"]), float(row["entry"]), float(row["stop"]), float(row["target"]))
    return float(row["cash"]), pos


def save_state(cash, pos):
    pd.DataFrame([{
        "cash": cash,
        "qty": pos.qty if pos else 0.0,
        "entry": pos.entry if pos else 0.0,
        "stop": pos.stop if pos else 0.0,
        "target": pos.target if pos else 0.0,
    }]).to_csv(STATE, index=False)


def log_trade(action, price, qty, cash, reason):
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow().isoformat(),
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


def get_candles(exchange):
    rows = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LOOKBACK + 2)
    df = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ms"], unit="ms", utc=True).dt.tz_localize(None)
    # Exclude the current, potentially incomplete candle.
    df = df.iloc[:-1].reset_index(drop=True)
    if len(df) < 64:
        raise RuntimeError("Not enough completed candles to forecast.")
    return df


def predict_return(predictor, candles):
    lookback = min(LOOKBACK, len(candles), 512)
    hist = candles.iloc[-lookback:].copy()
    last_ts = hist["timestamps"].iloc[-1]
    freq = pd.tseries.frequencies.to_offset(TIMEFRAME.replace("m", "min").replace("h", "h").replace("d", "D"))
    future_ts = pd.Series(pd.date_range(start=last_ts + freq, periods=FORECAST_CANDLES, freq=freq))
    hist_ts = hist["timestamps"].reset_index(drop=True)
    features = hist[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True)
    forecast = predictor.predict(
        df=features,
        x_timestamp=hist_ts,
        y_timestamp=future_ts,
        pred_len=FORECAST_CANDLES,
        T=1.0,
        top_p=0.9,
        sample_count=3,
    )
    predicted_close = float(forecast["close"].iloc[-1])
    current_close = float(hist["close"].iloc[-1])
    return predicted_close / current_close - 1.0, current_close


def main():
    exchange_class = getattr(ccxt, EXCHANGE_ID, None)
    if exchange_class is None:
        raise ValueError(f"Unknown CCXT exchange: {EXCHANGE_ID}")
    exchange = exchange_class({"enableRateLimit": True})
    predictor = get_model()
    cash, position = load_state()
    log.info("PAPER MODE only | %s %s | starting cash %.2f", SYMBOL, TIMEFRAME, cash)

    while True:
        try:
            candles = get_candles(exchange)
            edge, close = predict_return(predictor, candles)
            ticker = exchange.fetch_ticker(SYMBOL)
            price = float(ticker.get("last") or close)

            # Simulated exits always take priority over new entries.
            if position:
                exit_reason = None
                if price <= position.stop:
                    exit_reason = "stop_loss"
                elif price >= position.target:
                    exit_reason = "take_profit"
                elif edge < -MIN_EDGE_PCT:
                    exit_reason = "forecast_turn"
                if exit_reason:
                    fill = price * (1 - SLIPPAGE_PCT)
                    proceeds = position.qty * fill * (1 - FEE_PCT)
                    cash += proceeds
                    log_trade("SELL", fill, position.qty, cash, exit_reason)
                    log.info("PAPER SELL qty=%.6f price=%.2f reason=%s cash=%.2f", position.qty, fill, exit_reason, cash)
                    position = None

            if position is None and edge >= MIN_EDGE_PCT:
                risk_budget = cash * RISK_PCT
                stop_distance = price * STOP_LOSS_PCT
                qty_by_risk = risk_budget / stop_distance if stop_distance > 0 else 0
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

            save_state(cash, position)
            equity = cash + (position.qty * price if position else 0)
            log.info("price=%.2f forecast_edge=%.3f%% position=%s equity=%.2f", price, edge * 100, "open" if position else "flat", equity)
        except Exception:
            log.exception("Cycle failed; no real orders are possible in this starter.")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
