"""Deterministic, fee-aware historical simulation. No exchange order endpoints are used."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class SimPosition:
    qty: float
    entry: float
    stop: float
    target: float
    entry_fee: float


def simulate(candles, forecast_returns, starting_cash=1000.0, risk_pct=0.005,
             max_position_pct=0.10, stop_loss_pct=0.01, take_profit_pct=0.015,
             fee_pct=0.001, slippage_pct=0.0005, min_edge_pct=0.0025):
    """Simulate spot-only entries using the prior candle's forecast.

    Required columns: timestamp, open, high, low, close.
    forecast_returns[i] must be generated using data available only through candle i close.
    A signal at i can first trade at candle i+1 open, preventing look-ahead.
    If stop and target are both touched within one candle, stop-loss is assumed first.
    """
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(candles.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    if len(candles) != len(forecast_returns):
        raise ValueError("forecast_returns length must match candles length")
    if starting_cash <= 0 or not 0 < risk_pct <= 0.02 or not 0 < max_position_pct <= 0.25:
        raise ValueError("Invalid starting balance or risk/position limits")
    if not 0 < stop_loss_pct < 0.10 or not 0 < take_profit_pct < 0.20:
        raise ValueError("Invalid stop-loss/take-profit settings")
    if min(fee_pct, slippage_pct, min_edge_pct) < 0:
        raise ValueError("Fees, slippage and signal threshold cannot be negative")

    df = candles.reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any() or (df[col] <= 0).any():
            raise ValueError(f"{col} contains missing or non-positive values")
    cash = float(starting_cash)
    pos = None
    trades = []
    equity_rows = []
    pending_signal = None

    def close_position(i, raw_price, reason):
        nonlocal cash, pos
        fill = raw_price * (1 - slippage_pct)
        gross = pos.qty * fill
        exit_fee = gross * fee_pct
        cash += gross - exit_fee
        pnl = gross - exit_fee - (pos.qty * pos.entry + pos.entry_fee - pos.qty * pos.entry)
        # Entry fee is recorded separately; total trade P&L uses entry notional plus both fees.
        entry_notional = pos.qty * pos.entry
        trade_pnl = gross - exit_fee - entry_notional - pos.entry_fee
        trades.append({
            "entry_timestamp": pos.entry_timestamp,
            "exit_timestamp": df.loc[i, "timestamp"],
            "entry_price": pos.entry,
            "exit_price": fill,
            "qty": pos.qty,
            "pnl": trade_pnl,
            "reason": reason,
        })
        pos = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        signal = forecast_returns[i - 1]
        exited = False

        # Forecast known at previous close: act at this candle's open.
        if pos is not None and signal is not None and pd.notna(signal) and signal < -min_edge_pct:
            close_position(i, float(row["open"]), "forecast_turn")
            exited = True
        elif pos is None and signal is not None and pd.notna(signal) and signal >= min_edge_pct:
            raw_entry = float(row["open"])
            fill = raw_entry * (1 + slippage_pct)
            risk_budget = cash * risk_pct
            qty_risk = risk_budget / (fill * stop_loss_pct)
            qty_cap = cash * max_position_pct / fill
            qty_cash = cash / (fill * (1 + fee_pct))
            qty = max(0.0, min(qty_risk, qty_cap, qty_cash))
            cost = qty * fill
            entry_fee = cost * fee_pct
            if qty > 0 and cost + entry_fee <= cash:
                cash -= cost + entry_fee
                pos = SimPosition(
                    qty=qty, entry=fill, stop=fill * (1 - stop_loss_pct),
                    target=fill * (1 + take_profit_pct), entry_fee=entry_fee
                )
                pos.entry_timestamp = row["timestamp"]

        # Conservative intrabar execution. If both levels were crossed, assume stop first.
        if pos is not None:
            low, high, open_price = float(row["low"]), float(row["high"]), float(row["open"])
            if low <= pos.stop:
                raw_exit = min(open_price, pos.stop)
                close_position(i, raw_exit, "stop_loss")
                exited = True
            elif high >= pos.target:
                close_position(i, pos.target, "take_profit")
                exited = True

        mark = float(row["close"])
        equity = cash + (pos.qty * mark if pos else 0.0)
        equity_rows.append({"timestamp": row["timestamp"], "equity": equity, "cash": cash, "in_position": pos is not None})

    # Liquidate any remaining position at the final close for a fair end-of-test equity.
    if pos is not None and len(df):
        close_position(len(df) - 1, float(df.iloc[-1]["close"]), "end_of_test")
        if equity_rows:
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["in_position"] = False

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades, columns=[
        "entry_timestamp", "exit_timestamp", "entry_price", "exit_price", "qty", "pnl", "reason"
    ])
    if equity_df.empty:
        final_equity = starting_cash
        max_drawdown = 0.0
    else:
        values = pd.concat([pd.Series([float(starting_cash)]), equity_df["equity"].reset_index(drop=True)], ignore_index=True)
        peaks = values.cummax()
        max_drawdown = float((values / peaks - 1).min())
        final_equity = float(equity_df["equity"].iloc[-1])
    buy_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) > 1 else 0.0
    wins = int((trades_df["pnl"] > 0).sum()) if not trades_df.empty else 0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if not trades_df.empty else 0.0
    gross_loss = abs(float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())) if not trades_df.empty else 0.0
    metrics = {
        "starting_cash": float(starting_cash),
        "final_equity": final_equity,
        "net_return_pct": (final_equity / starting_cash - 1) * 100,
        "buy_hold_return_pct": buy_hold_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "closed_trades": int(len(trades_df)),
        "win_rate_pct": (wins / len(trades_df) * 100) if not trades_df.empty else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
    }
    return metrics, trades_df, equity_df
