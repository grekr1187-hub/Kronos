import unittest

import pandas as pd

from trading_bot.backtest_engine import simulate


def candles(opens, highs, lows, closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=len(opens), freq="h", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


class BacktestEngineTests(unittest.TestCase):
    def test_signal_trades_next_bar_not_same_bar(self):
        data = candles([100, 100, 100], [100.5, 100.5, 100.5], [99.5, 99.5, 99.5], [100, 100, 100])
        metrics, trades, equity = simulate(data, [0.01, 0.0, 0.0], fee_pct=0, slippage_pct=0)
        self.assertEqual(metrics["closed_trades"], 1)
        self.assertEqual(trades.iloc[0]["entry_price"], 100)
        self.assertEqual(trades.iloc[0]["entry_timestamp"], data.iloc[1]["timestamp"])
        self.assertEqual(trades.iloc[0]["reason"], "end_of_test")

    def test_stop_wins_when_stop_and_target_hit_same_bar(self):
        data = candles([100, 100, 100], [101, 104, 101], [99, 98, 99], [100, 100, 100])
        metrics, trades, equity = simulate(data, [0.01, 0.0, 0.0], fee_pct=0, slippage_pct=0)
        self.assertEqual(trades.iloc[0]["reason"], "stop_loss")
        self.assertAlmostEqual(trades.iloc[0]["exit_price"], 99.0)

    def test_position_limit_is_respected(self):
        data = candles([100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        metrics, trades, equity = simulate(data, [0.01, 0.0, 0.0], fee_pct=0, slippage_pct=0)
        self.assertLessEqual(trades.iloc[0]["qty"] * trades.iloc[0]["entry_price"], 100.0 + 1e-8)

    def test_rejects_forecast_length_mismatch(self):
        data = candles([100, 100], [101, 101], [99, 99], [100, 100])
        with self.assertRaises(ValueError):
            simulate(data, [0.01])

    def test_no_signal_stays_in_cash(self):
        data = candles([100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100])
        metrics, trades, equity = simulate(data, [0.0, 0.0, 0.0], fee_pct=0, slippage_pct=0)
        self.assertEqual(metrics["closed_trades"], 0)
        self.assertEqual(metrics["final_equity"], 1000.0)


if __name__ == "__main__":
    unittest.main()
