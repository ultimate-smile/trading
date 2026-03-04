"""AI增强量化交易系统（A股示例，纸上交易版）。

核心能力：
1) 使用历史行情构造因子并训练轻量线性模型预测下一日收益率；
2) 在 10W 初始资金约束下进行仓位控制与交易决策；
3) 内置止损/止盈/模型转弱退出等基础风险管理。

注意：本脚本仅用于学习研究，不构成任何投资建议。
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


# =============================
# 关键参数（可按需求调优）
# =============================
POLL_INTERVAL_SECONDS = 30  # 轮询间隔（秒）
SYMBOLS = ["000988", "688387"]  # 股票池（A股代码）
INITIAL_CAPITAL = 100_000  # 初始资金（元）
LOT_SIZE = 100  # A股最小交易单位：1手=100股
MODEL_LOOKBACK_DAYS = 120  # 模型训练最多使用最近N天样本
MIN_TRAIN_SAMPLES = 40  # 模型最小训练样本数，低于该值不交易
MAX_POSITION_RATIO = 0.4  # 单标的最大仓位比例（占总权益）
STOP_LOSS_PCT = 0.03  # 止损阈值（-3%）
TAKE_PROFIT_PCT = 0.06  # 止盈阈值（+6%）
BUY_SIGNAL_THRESHOLD = 0.002  # 买入阈值（预测收益率 > 0.2%）


@dataclass
class Position:
    """单个持仓对象（主要变量容器）。

    属性说明：
    - symbol: 股票代码
    - quantity: 持仓股数
    - entry_price: 开仓价格
    - entry_time: 开仓时间
    """

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime


class QuantTradingSystem:
    """量化交易系统主类。"""

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = INITIAL_CAPITAL,
        poll_interval: int = POLL_INTERVAL_SECONDS,
    ) -> None:
        # ===== 主要变量（账户维度） =====
        self.symbols = symbols  # 跟踪股票池
        self.poll_interval = poll_interval  # 主循环轮询间隔
        self.initial_capital = float(initial_capital)  # 初始资金
        self.cash = float(initial_capital)  # 可用现金
        self.positions: Dict[str, Position] = {}  # 当前持仓：symbol -> Position
        self.realized_pnl = 0.0  # 已实现盈亏（平仓后累计）

    @staticmethod
    def get_history(symbol: str, lookback_days: int = MODEL_LOOKBACK_DAYS + 40) -> Optional[pd.DataFrame]:
        """获取并标准化历史K线数据。

        参数：
        - symbol: 股票代码
        - lookback_days: 拉取后保留的历史天数

        返回：
        - 含 date/open/high/low/close 的DataFrame；若数据不足返回None。
        """
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
        """构建模型输入因子与预测目标。

        关键字段：
        - ret_1d: 1日收益率
        - mom_5d/mom_10d: 5/10日动量
        - vol_10d: 10日波动率（1日收益标准差）
        - rsi_14: 14日RSI
        - target_next_ret: 下一交易日收益率（监督学习标签）
        """
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
        """训练轻量线性模型，并预测“最新时点”的下一日收益率。

        方法说明：
        - 先做缺失值过滤与样本量检查；
        - 对特征标准化；
        - 使用最小二乘拟合线性回归；
        - 输出最新一行特征的预测收益率。
        """
        cols = ["ret_1d", "mom_5d", "mom_10d", "vol_10d", "rsi_14"]
        clean = feature_df.dropna(subset=cols + ["target_next_ret"]).copy()
        if len(clean) < MIN_TRAIN_SAMPLES:
            return None

        train = clean.tail(MODEL_LOOKBACK_DAYS)
        x_train = train[cols].to_numpy(dtype=float)
        y_train = train["target_next_ret"].to_numpy(dtype=float)

        # 特征标准化
        means = x_train.mean(axis=0)
        stds = x_train.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        x_train_scaled = (x_train - means) / stds

        # 最小二乘线性回归: y = b0 + b1*x1 + ...
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
        """批量拉取股票池最新价，返回 {symbol: latest_price}。"""
        spot = ak.stock_zh_a_spot_em()
        rows = spot[spot["代码"].isin(symbols)][["代码", "最新价"]]
        return {str(row["代码"]): float(row["最新价"]) for _, row in rows.iterrows()}

    def market_value(self, spot_prices: Dict[str, float]) -> float:
        """计算持仓市值（未实现盈亏体现在这里）。"""
        holdings = 0.0
        for symbol, pos in self.positions.items():
            price = spot_prices.get(symbol, pos.entry_price)
            holdings += pos.quantity * price
        return holdings

    def total_equity(self, spot_prices: Dict[str, float]) -> float:
        """账户总权益 = 现金 + 持仓市值。"""
        return self.cash + self.market_value(spot_prices)

    def compute_order_quantity(self, symbol: str, price: float, spot_prices: Dict[str, float]) -> int:
        """按仓位上限与现金约束计算可买股数（按100股取整）。"""
        equity = self.total_equity(spot_prices)
        target_value = equity * MAX_POSITION_RATIO
        affordable = min(target_value, self.cash)
        lots = int(affordable // (price * LOT_SIZE))
        return max(0, lots * LOT_SIZE)

    def should_force_sell(self, symbol: str, price: float) -> bool:
        """是否触发风控卖出（止损/止盈）。"""
        position = self.positions[symbol]
        pnl_pct = (price - position.entry_price) / position.entry_price
        return pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT

    def place_buy_order(self, symbol: str, price: float, quantity: int) -> None:
        """执行买入（纸上交易）：更新现金与持仓。"""
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
        """执行卖出（纸上交易）：更新现金与已实现盈亏。"""
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
        """主循环：行情获取 -> 特征/预测 -> 风控卖出 -> 信号买入 -> 账户汇总。"""
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

                predictions: Dict[str, float] = {}  # 本轮预测结果：symbol -> pred_next_ret
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

                # 第一步：优先执行风险退出
                for symbol in list(self.positions.keys()):
                    price = spot_prices.get(symbol)
                    if price is None:
                        continue
                    if self.should_force_sell(symbol, price):
                        self.place_sell_order(symbol, price, reason="risk_control")
                    elif predictions.get(symbol, -1.0) < -BUY_SIGNAL_THRESHOLD:
                        self.place_sell_order(symbol, price, reason="model_turn_negative")

                # 第二步：按预测值从高到低开仓
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
