"""AI-assisted quantitative trading system for A-share symbols.

Features
--------
- 100,000 CNY initial capital by default.
- Online inference of next-day return using a lightweight linear model.
- Risk controls: max position size, stop-loss, take-profit and cash checks.
- Paper-trading implementation (in-memory portfolio and trade logs).

Disclaimer
----------
This script is for educational/research usage only and is NOT investment advice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import akshare as ak
import numpy as np
import pandas as pd


POLL_INTERVAL_SECONDS = 30
SYMBOLS = ["000988", "688387"]
INITIAL_CAPITAL = 100_000
LOT_SIZE = 100
MODEL_LOOKBACK_DAYS = 120
MIN_TRAIN_SAMPLES = 40
MAX_POSITION_RATIO = 0.4
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
BUY_SIGNAL_THRESHOLD = 0.002


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime


class QuantTradingSystem:
    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = INITIAL_CAPITAL,
        poll_interval: int = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

    @staticmethod
    def get_history(symbol: str, lookback_days: int = MODEL_LOOKBACK_DAYS + 40) -> Optional[pd.DataFrame]:
        """Fetch historical daily bars and normalize column names."""
        history = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        if history.empty or len(history) < 30:
            return None

        hist = history.copy()
        hist = hist.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "开盘": "open"})
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date").tail(lookback_days).reset_index(drop=True)
        return hist

    @staticmethod
    def build_features(hist: pd.DataFrame) -> pd.DataFrame:
        """Create ML features and prediction target (next-day return)."""
        df = hist.copy()
        df["ret_1d"] = df["close"].pct_change(1)
        df["mom_5d"] = df["close"].pct_change(5)
        df["mom_10d"] = df["close"].pct_change(10)
        df["vol_10d"] = df["ret_1d"].rolling(10).std()

        # RSI(14)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        df["target_next_ret"] = df["close"].shift(-1) / df["close"] - 1
        return df

    @staticmethod
    def predict_next_return(feature_df: pd.DataFrame) -> Optional[float]:
        """Train a tiny linear model and predict next-day return for latest row."""
        cols = ["ret_1d", "mom_5d", "mom_10d", "vol_10d", "rsi_14"]
        clean = feature_df.dropna(subset=cols + ["target_next_ret"]).copy()
        if len(clean) < MIN_TRAIN_SAMPLES:
            return None

        train = clean.tail(MODEL_LOOKBACK_DAYS)
        x_train = train[cols].to_numpy(dtype=float)
        y_train = train["target_next_ret"].to_numpy(dtype=float)

        # Feature standardization
        means = x_train.mean(axis=0)
        stds = x_train.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        x_train_scaled = (x_train - means) / stds

        # Linear regression via least squares: y = b0 + b1*x1 + ...
        design = np.column_stack([np.ones(len(x_train_scaled)), x_train_scaled])
        coefs, *_ = np.linalg.lstsq(design, y_train, rcond=None)

        latest = feature_df.iloc[-1]
        if latest[cols].isna().any():
            return None

        x_latest = latest[cols].to_numpy(dtype=float)
        x_latest_scaled = (x_latest - means) / stds
        pred = float(coefs[0] + np.dot(coefs[1:], x_latest_scaled))
        return pred

    @staticmethod
    def fetch_spot_prices(symbols: list[str]) -> Dict[str, float]:
        """Get latest spot prices for all tracked symbols in one request."""
        spot = ak.stock_zh_a_spot_em()
        rows = spot[spot["代码"].isin(symbols)][["代码", "最新价"]]
        return {str(row["代码"]): float(row["最新价"]) for _, row in rows.iterrows()}

    def market_value(self, spot_prices: Dict[str, float]) -> float:
        holdings = 0.0
        for symbol, pos in self.positions.items():
            price = spot_prices.get(symbol, pos.entry_price)
            holdings += pos.quantity * price
        return holdings

    def total_equity(self, spot_prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(spot_prices)

    def compute_order_quantity(self, symbol: str, price: float, spot_prices: Dict[str, float]) -> int:
        equity = self.total_equity(spot_prices)
        target_value = equity * MAX_POSITION_RATIO
        affordable = min(target_value, self.cash)
        lots = int(affordable // (price * LOT_SIZE))
        return max(0, lots * LOT_SIZE)

    def should_force_sell(self, symbol: str, price: float) -> bool:
        position = self.positions[symbol]
        pnl_pct = (price - position.entry_price) / position.entry_price
        return pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT

    def place_buy_order(self, symbol: str, price: float, quantity: int) -> None:
        cost = price * quantity
        if quantity <= 0 or cost > self.cash:
            return

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
        )
        logging.info("BUY  | symbol=%s qty=%s price=%.2f cost=%.2f cash=%.2f", symbol, quantity, price, cost, self.cash)

    def place_sell_order(self, symbol: str, price: float, reason: str) -> None:
        position = self.positions.pop(symbol)
        value = price * position.quantity
        pnl = (price - position.entry_price) * position.quantity
        self.realized_pnl += pnl
        self.cash += value
        logging.info(
            "SELL | symbol=%s qty=%s price=%.2f value=%.2f pnl=%.2f reason=%s cash=%.2f",
            symbol,
            position.quantity,
            price,
            value,
            pnl,
            reason,
            self.cash,
        )

    def run(self) -> None:
        logging.info(
            "System started. symbols=%s poll_interval=%ss initial_capital=%.2f",
            self.symbols,
            self.poll_interval,
            self.initial_capital,
        )

        while True:
            try:
                spot_prices = self.fetch_spot_prices(self.symbols)
                if not spot_prices:
                    logging.warning("No spot prices in this cycle")
                    time.sleep(self.poll_interval)
                    continue

                predictions: Dict[str, float] = {}
                for symbol in self.symbols:
                    price = spot_prices.get(symbol)
                    if price is None:
                        logging.warning("No latest price for %s", symbol)
                        continue

                    hist = self.get_history(symbol)
                    if hist is None:
                        logging.warning("No enough history for %s", symbol)
                        continue

                    feature_df = self.build_features(hist)
                    pred = self.predict_next_return(feature_df)
                    if pred is None:
                        logging.warning("No valid prediction for %s", symbol)
                        continue

                    predictions[symbol] = pred
                    logging.info("TICK | %s price=%.2f pred_next_ret=%.4f", symbol, price, pred)

                # First: risk exits
                for symbol in list(self.positions.keys()):
                    price = spot_prices.get(symbol)
                    if price is None:
                        continue
                    if self.should_force_sell(symbol, price):
                        self.place_sell_order(symbol, price, reason="risk_control")
                    elif predictions.get(symbol, -1.0) < -BUY_SIGNAL_THRESHOLD:
                        self.place_sell_order(symbol, price, reason="model_turn_negative")

                # Second: open best opportunities
                ranked = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
                for symbol, pred in ranked:
                    if pred < BUY_SIGNAL_THRESHOLD:
                        continue
                    if symbol in self.positions:
                        continue

                    price = spot_prices[symbol]
                    quantity = self.compute_order_quantity(symbol, price, spot_prices)
                    if quantity > 0:
                        self.place_buy_order(symbol, price, quantity)

                equity = self.total_equity(spot_prices)
                logging.info(
                    "ACCOUNT | cash=%.2f holdings=%.2f equity=%.2f realized_pnl=%.2f positions=%s",
                    self.cash,
                    self.market_value(spot_prices),
                    equity,
                    self.realized_pnl,
                    list(self.positions.keys()),
                )

            except Exception as exc:  # keep loop alive for long-running service
                logging.exception("Cycle error: %s", exc)

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    system = QuantTradingSystem(symbols=SYMBOLS, initial_capital=INITIAL_CAPITAL)
    system.run()
