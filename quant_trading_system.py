"""Simple quantitative trading system for A-share symbols.

Strategy
--------
- Universe: 000988, 688387
- Poll latest price every 5 seconds.
- Buy trigger: latest price <= (5-day average close + 1)
- Sell trigger: latest price >= (5-day average close * 1.05)

This script is intended for research/paper-trading demonstration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import akshare as ak


POLL_INTERVAL_SECONDS = 5
SYMBOLS = ["000988", "688387"]


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime


class QuantTradingSystem:
    def __init__(self, symbols: list[str], poll_interval: int = POLL_INTERVAL_SECONDS) -> None:
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.positions: Dict[str, Position] = {}

    @staticmethod
    def five_day_avg_close(symbol: str) -> Optional[float]:
        """Get the latest 5 trading-day average close price for a symbol."""
        history = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            adjust="qfq",
        )
        if history.empty or len(history) < 5:
            return None
        return float(history["收盘"].tail(5).mean())

    @staticmethod
    def latest_price(symbol: str) -> Optional[float]:
        """Get latest spot price for one symbol."""
        spot = ak.stock_zh_a_spot_em()
        row = spot[spot["代码"] == symbol]
        if row.empty:
            return None
        return float(row.iloc[0]["最新价"])

    def should_buy(self, symbol: str, price: float, avg5: float) -> bool:
        buy_point = avg5 + 1
        return symbol not in self.positions and price <= buy_point

    def should_sell(self, symbol: str, price: float, avg5: float) -> bool:
        sell_point = avg5 * 1.05
        return symbol in self.positions and price >= sell_point

    def place_buy_order(self, symbol: str, price: float, quantity: int = 100) -> None:
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
        )
        logging.info("BUY  | symbol=%s qty=%s price=%.2f", symbol, quantity, price)

    def place_sell_order(self, symbol: str, price: float) -> None:
        position = self.positions.pop(symbol)
        pnl = (price - position.entry_price) * position.quantity
        logging.info(
            "SELL | symbol=%s qty=%s price=%.2f pnl=%.2f",
            symbol,
            position.quantity,
            price,
            pnl,
        )

    def run(self) -> None:
        logging.info("System started. symbols=%s poll_interval=%ss", self.symbols, self.poll_interval)
        while True:
            for symbol in self.symbols:
                try:
                    avg5 = self.five_day_avg_close(symbol)
                    price = self.latest_price(symbol)
                    if avg5 is None or price is None:
                        logging.warning("No data for %s, skip this cycle", symbol)
                        continue

                    buy_point = avg5 + 1
                    sell_point = avg5 * 1.05
                    logging.info(
                        "TICK | %s price=%.2f avg5=%.2f buy<=%.2f sell>=%.2f",
                        symbol,
                        price,
                        avg5,
                        buy_point,
                        sell_point,
                    )

                    if self.should_buy(symbol, price, avg5):
                        self.place_buy_order(symbol, price)
                    elif self.should_sell(symbol, price, avg5):
                        self.place_sell_order(symbol, price)
                except Exception as exc:  # keep loop alive for long-running service
                    logging.exception("Error while handling %s: %s", symbol, exc)

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    system = QuantTradingSystem(symbols=SYMBOLS)
    system.run()
