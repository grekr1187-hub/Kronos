import unittest

from trading_bot import paper_bot


class PaperBotTests(unittest.TestCase):
    def test_timeframe_frequency_minutes_and_hours(self):
        self.assertEqual(paper_bot.timeframe_frequency("15m"), __import__("pandas").offsets.Minute(15))
        self.assertEqual(paper_bot.timeframe_frequency("4h"), __import__("pandas").offsets.Hour(4))

    def test_rejects_unsupported_timeframe(self):
        with self.assertRaises(ValueError):
            paper_bot.timeframe_frequency("15x")

    def test_order_quantity_obeys_position_cap_and_cash(self):
        cash, price = 1000.0, 50000.0
        qty = paper_bot.calculate_order_quantity(cash, price)
        total_cost = qty * price * (1 + paper_bot.SLIPPAGE_PCT) * (1 + paper_bot.FEE_PCT)
        self.assertGreaterEqual(qty, 0.0)
        self.assertLessEqual(qty * price, cash * paper_bot.MAX_POSITION_PCT + 1e-9)
        self.assertLessEqual(total_cost, cash + 1e-9)

    def test_order_quantity_is_zero_for_invalid_inputs(self):
        self.assertEqual(paper_bot.calculate_order_quantity(0, 100), 0.0)
        self.assertEqual(paper_bot.calculate_order_quantity(100, 0), 0.0)

    def test_stop_loss_has_priority(self):
        position = paper_bot.Position(qty=0.01, entry=100.0, stop=99.0, target=102.0)
        self.assertEqual(paper_bot.exit_reason_for_position(position, 98.0), "stop_loss")

    def test_take_profit(self):
        position = paper_bot.Position(qty=0.01, entry=100.0, stop=99.0, target=102.0)
        self.assertEqual(paper_bot.exit_reason_for_position(position, 102.0), "take_profit")

    def test_forecast_exit_only_on_new_candle(self):
        position = paper_bot.Position(qty=0.01, entry=100.0, stop=99.0, target=102.0)
        self.assertIsNone(paper_bot.exit_reason_for_position(position, 100.0, edge=-0.02, new_candle=False))
        self.assertEqual(paper_bot.exit_reason_for_position(position, 100.0, edge=-0.02, new_candle=True), "forecast_turn")


if __name__ == "__main__":
    unittest.main()
