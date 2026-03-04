"""AI量化交易系统（支持多模型、回测与可选实盘网关）。

版本目标：
1) 模型层支持 Linear / XGBoost / LSTM / Transformer（后3者为可选依赖）；
2) 交易层支持纸上交易与“可接入”实盘下单网关；
3) 评估层内置日线回测，并输出 Sharpe / 最大回撤 / 胜率。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Protocol

import akshare as ak
import numpy as np
import pandas as pd


# =============================
# 关键参数（可按需求调优）
# =============================
POLL_INTERVAL_SECONDS = 30
SYMBOLS = ["000988", "688387"]
INITIAL_CAPITAL = 100_000
LOT_SIZE = 100
MODEL_LOOKBACK_DAYS = 150
MIN_TRAIN_SAMPLES = 60
MAX_POSITION_RATIO = 0.4
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
BUY_SIGNAL_THRESHOLD = 0.002
DATA_FETCH_RETRIES = 3  # 行情/历史数据获取重试次数
DATA_FETCH_RETRY_DELAY = 1.5  # 失败重试基础等待秒数（指数退避）

# 模型配置
MODEL_NAME = "xgboost"  # 可选: linear / xgboost / lstm / transformer
SEQUENCE_LENGTH = 20  # LSTM/Transformer序列长度
NN_EPOCHS = 20
NN_LR = 1e-3

# 实盘相关配置
ENABLE_LIVE_TRADING = False  # 生产环境务必配合风控与小额验证后再开启
BROKER_BASE_URL = ""  # 例如: https://your-broker-gateway/api
BROKER_API_KEY = ""


@dataclass
class Position:
    """单个持仓对象。"""

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime


@dataclass
class BacktestResult:
    """回测结果核心指标。"""

    symbol: str
    model_name: str
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int


class BrokerGateway(Protocol):
    """券商网关协议。

    任意实盘网关只要实现 place_order 即可被系统接入。
    """

    def place_order(self, symbol: str, side: str, quantity: int, price: float) -> dict:
        ...


class HttpBrokerGateway:
    """HTTP实盘网关示例。

    通过你自己的中间层服务转发到真实券商。
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def place_order(self, symbol: str, side: str, quantity: int, price: float) -> dict:
        import requests

        if not self.base_url:
            raise ValueError("BROKER_BASE_URL 为空，无法发送实盘订单")
        payload = {"symbol": symbol, "side": side, "quantity": quantity, "price": price}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(f"{self.base_url}/orders", json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()


class QuantTradingSystem:
    """量化交易系统主类。"""

    FEATURE_COLS = ["ret_1d", "mom_5d", "mom_10d", "vol_10d", "rsi_14"]

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = INITIAL_CAPITAL,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        model_name: str = MODEL_NAME,
        enable_live_trading: bool = ENABLE_LIVE_TRADING,
        broker: Optional[BrokerGateway] = None,
    ) -> None:
        # 主要变量（账户状态）
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

        # 主要变量（模型/执行）
        self.model_name = model_name.lower()
        self.enable_live_trading = enable_live_trading
        self.broker = broker

        # 数据兜底缓存（网络抖动时尽量不中断主循环）
        self.last_spot_prices: Dict[str, float] = {}
        self.last_history: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _with_retry(func, *args, **kwargs):
        """对外部数据接口调用做重试（指数退避）。"""
        last_error: Optional[Exception] = None
        for attempt in range(1, DATA_FETCH_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                sleep_s = DATA_FETCH_RETRY_DELAY * (2 ** (attempt - 1))
                logging.warning(
                    "Data fetch failed (attempt=%s/%s): %s; retry in %.1fs",
                    attempt,
                    DATA_FETCH_RETRIES,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Data fetch failed after retries: {last_error}")

    @staticmethod
    def get_history(symbol: str, lookback_days: int = MODEL_LOOKBACK_DAYS + 120) -> Optional[pd.DataFrame]:
        """获取并标准化历史K线数据。"""
        history = QuantTradingSystem._with_retry(ak.stock_zh_a_hist, symbol=symbol, period="daily", adjust="qfq")
        if history.empty or len(history) < 60:
            return None

        hist = history.copy()
        hist = hist.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "开盘": "open"})
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date").tail(lookback_days).reset_index(drop=True)
        return hist

    @staticmethod
    def build_features(hist: pd.DataFrame) -> pd.DataFrame:
        """构建特征与标签。"""
        df = hist.copy()
        df["ret_1d"] = df["close"].pct_change(1)
        df["mom_5d"] = df["close"].pct_change(5)
        df["mom_10d"] = df["close"].pct_change(10)
        df["vol_10d"] = df["ret_1d"].rolling(10).std()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        df["target_next_ret"] = df["close"].shift(-1) / df["close"] - 1
        return df

    @staticmethod
    def _predict_linear(train_df: pd.DataFrame, latest_row: pd.Series, feature_cols: list[str]) -> Optional[float]:
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target_next_ret"].to_numpy(dtype=float)
        means = x_train.mean(axis=0)
        stds = np.where(x_train.std(axis=0) == 0, 1.0, x_train.std(axis=0))
        x_scaled = (x_train - means) / stds
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        coefs, *_ = np.linalg.lstsq(design, y_train, rcond=None)
        x_latest = latest_row[feature_cols].to_numpy(dtype=float)
        x_latest_scaled = (x_latest - means) / stds
        return float(coefs[0] + np.dot(coefs[1:], x_latest_scaled))

    @staticmethod
    def _predict_xgboost(train_df: pd.DataFrame, latest_row: pd.Series, feature_cols: list[str]) -> Optional[float]:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None
        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(train_df[feature_cols], train_df["target_next_ret"])
        pred = model.predict(latest_row[feature_cols].to_frame().T)[0]
        return float(pred)

    @staticmethod
    def _predict_lstm_or_transformer(
        train_df: pd.DataFrame,
        latest_window: np.ndarray,
        feature_cols: list[str],
        model_name: str,
    ) -> Optional[float]:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return None

        x = train_df[feature_cols].to_numpy(dtype=np.float32)
        y = train_df["target_next_ret"].to_numpy(dtype=np.float32)

        if len(x) <= SEQUENCE_LENGTH + 1:
            return None

        # 标准化
        means = x.mean(axis=0)
        stds = x.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        x = (x - means) / stds
        latest_window = (latest_window - means) / stds

        seq_x, seq_y = [], []
        for i in range(SEQUENCE_LENGTH, len(x)):
            seq_x.append(x[i - SEQUENCE_LENGTH : i])
            seq_y.append(y[i])
        X = torch.tensor(np.array(seq_x), dtype=torch.float32)
        Y = torch.tensor(np.array(seq_y), dtype=torch.float32).view(-1, 1)

        class LSTMRegressor(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(z)
                return self.fc(out[:, -1, :])

        class TransformerRegressor(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                self.proj = nn.Linear(input_dim, 32)
                layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.fc = nn.Linear(32, 1)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                h = self.proj(z)
                h = self.encoder(h)
                return self.fc(h[:, -1, :])

        model = LSTMRegressor(len(feature_cols)) if model_name == "lstm" else TransformerRegressor(len(feature_cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=NN_LR)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(NN_EPOCHS):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

        model.eval()
        latest_tensor = torch.tensor(latest_window[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred = model(latest_tensor).item()
        return float(pred)

    def predict_next_return(self, feature_df: pd.DataFrame) -> Optional[float]:
        """根据 model_name 预测下一日收益率。"""
        cols = self.FEATURE_COLS
        clean = feature_df.dropna(subset=cols + ["target_next_ret"]).copy()
        if len(clean) < MIN_TRAIN_SAMPLES:
            return None

        train_df = clean.tail(MODEL_LOOKBACK_DAYS)
        latest = feature_df.iloc[-1]
        if latest[cols].isna().any():
            return None

        if self.model_name == "linear":
            return self._predict_linear(train_df, latest, cols)
        if self.model_name == "xgboost":
            return self._predict_xgboost(train_df, latest, cols) or self._predict_linear(train_df, latest, cols)
        if self.model_name in {"lstm", "transformer"}:
            latest_window_df = feature_df.dropna(subset=cols).tail(SEQUENCE_LENGTH)
            if len(latest_window_df) < SEQUENCE_LENGTH:
                return None
            pred = self._predict_lstm_or_transformer(
                train_df,
                latest_window_df[cols].to_numpy(dtype=np.float32),
                cols,
                self.model_name,
            )
            return pred or self._predict_linear(train_df, latest, cols)

        return self._predict_linear(train_df, latest, cols)

    @staticmethod
    def fetch_spot_prices(symbols: list[str]) -> Dict[str, float]:
        spot = QuantTradingSystem._with_retry(ak.stock_zh_a_spot_em)
        rows = spot[spot["代码"].isin(symbols)][["代码", "最新价"]]
        return {str(row["代码"]): float(row["最新价"]) for _, row in rows.iterrows()}

    def get_history_safe(self, symbol: str) -> Optional[pd.DataFrame]:
        """带缓存兜底的历史数据获取。"""
        try:
            hist = self.get_history(symbol)
            if hist is not None and not hist.empty:
                self.last_history[symbol] = hist
            return hist
        except Exception as exc:
            logging.error("History fetch error for %s: %s", symbol, exc)
            return self.last_history.get(symbol)

    def fetch_spot_prices_safe(self, symbols: list[str]) -> Dict[str, float]:
        """带缓存兜底的实时行情获取。"""
        try:
            prices = self.fetch_spot_prices(symbols)
            if prices:
                self.last_spot_prices.update(prices)
            return prices
        except Exception as exc:
            logging.error("Spot fetch error: %s", exc)
            fallback = {s: p for s, p in self.last_spot_prices.items() if s in symbols}
            if fallback:
                logging.warning("Using cached spot prices for this cycle: %s", list(fallback.keys()))
            return fallback

    def market_value(self, spot_prices: Dict[str, float]) -> float:
        return sum(pos.quantity * spot_prices.get(s, pos.entry_price) for s, pos in self.positions.items())

    def total_equity(self, spot_prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(spot_prices)

    def compute_order_quantity(self, price: float, spot_prices: Dict[str, float]) -> int:
        equity = self.total_equity(spot_prices)
        target_value = equity * MAX_POSITION_RATIO
        affordable = min(target_value, self.cash)
        lots = int(affordable // (price * LOT_SIZE))
        return max(0, lots * LOT_SIZE)

    def should_force_sell(self, symbol: str, price: float) -> bool:
        position = self.positions[symbol]
        pnl_pct = (price - position.entry_price) / position.entry_price
        return pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT

    def _send_live_order(self, symbol: str, side: str, quantity: int, price: float) -> None:
        if not self.enable_live_trading:
            return
        if self.broker is None:
            raise RuntimeError("ENABLE_LIVE_TRADING=True 但未提供 broker 网关")
        result = self.broker.place_order(symbol=symbol, side=side, quantity=quantity, price=price)
        logging.info("LIVE_ORDER | %s", result)

    def place_buy_order(self, symbol: str, price: float, quantity: int) -> None:
        cost = price * quantity
        if quantity <= 0 or cost > self.cash:
            return
        self._send_live_order(symbol, "BUY", quantity, price)
        self.cash -= cost
        self.positions[symbol] = Position(symbol=symbol, quantity=quantity, entry_price=price, entry_time=datetime.now())
        logging.info("BUY  | symbol=%s qty=%s price=%.2f cash=%.2f", symbol, quantity, price, self.cash)

    def place_sell_order(self, symbol: str, price: float, reason: str) -> None:
        position = self.positions.pop(symbol)
        self._send_live_order(symbol, "SELL", position.quantity, price)
        value = price * position.quantity
        pnl = (price - position.entry_price) * position.quantity
        self.realized_pnl += pnl
        self.cash += value
        logging.info("SELL | symbol=%s qty=%s price=%.2f pnl=%.2f reason=%s cash=%.2f", symbol, position.quantity, price, pnl, reason, self.cash)

    def backtest_symbol(self, symbol: str, start_date: str = "2019-01-01", end_date: str = "2024-12-31") -> Optional[BacktestResult]:
        """单标的滚动回测（日线）。"""
        history = self._with_retry(
            ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
        if history.empty or len(history) < MIN_TRAIN_SAMPLES + 30:
            return None
        hist = history.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "开盘": "open"}).copy()
        hist["date"] = pd.to_datetime(hist["date"])
        feat = self.build_features(hist)

        returns = []
        equity_curve = [1.0]
        win, trade_count = 0, 0

        clean = feat.dropna(subset=self.FEATURE_COLS + ["target_next_ret"]).reset_index(drop=True)
        for i in range(MIN_TRAIN_SAMPLES, len(clean) - 1):
            train_slice = clean.iloc[:i].tail(MODEL_LOOKBACK_DAYS).copy()
            latest_row = clean.iloc[i]
            next_ret = float(clean.iloc[i]["target_next_ret"])

            # 为复用预测方法构造临时DataFrame
            pred_df = pd.concat([train_slice, latest_row.to_frame().T], ignore_index=True)
            pred = self.predict_next_return(pred_df)
            if pred is None:
                strategy_ret = 0.0
            elif pred > BUY_SIGNAL_THRESHOLD:
                strategy_ret = next_ret
                trade_count += 1
                if strategy_ret > 0:
                    win += 1
            elif pred < -BUY_SIGNAL_THRESHOLD:
                strategy_ret = -next_ret
                trade_count += 1
                if strategy_ret > 0:
                    win += 1
            else:
                strategy_ret = 0.0

            returns.append(strategy_ret)
            equity_curve.append(equity_curve[-1] * (1 + strategy_ret))

        ret_arr = np.array(returns, dtype=float)
        total_return = equity_curve[-1] - 1
        annualized = (1 + total_return) ** (252 / max(1, len(ret_arr))) - 1
        vol = ret_arr.std() * np.sqrt(252)
        sharpe = 0.0 if vol == 0 else (ret_arr.mean() * 252) / vol

        curve = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(curve)
        drawdown = (curve - peak) / peak
        max_dd = abs(drawdown.min())
        win_rate = 0.0 if trade_count == 0 else win / trade_count

        return BacktestResult(
            symbol=symbol,
            model_name=self.model_name,
            total_return=float(total_return),
            annualized_return=float(annualized),
            sharpe=float(sharpe),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            trades=int(trade_count),
        )

    def run(self) -> None:
        """主循环：数据 -> 预测 -> 风控卖出 -> 选股买入 -> 账户汇总。"""
        logging.info("System started. symbols=%s model=%s live=%s", self.symbols, self.model_name, self.enable_live_trading)
        while True:
            try:
                spot_prices = self.fetch_spot_prices_safe(self.symbols)
                if not spot_prices:
                    logging.warning("No spot prices in this cycle")
                    time.sleep(self.poll_interval)
                    continue

                predictions: Dict[str, float] = {}
                for symbol in self.symbols:
                    price = spot_prices.get(symbol)
                    if price is None:
                        continue
                    hist = self.get_history_safe(symbol)
                    if hist is None:
                        continue
                    feat = self.build_features(hist)
                    pred = self.predict_next_return(feat)
                    if pred is None:
                        continue
                    predictions[symbol] = pred
                    logging.info("TICK | %s price=%.2f pred=%.4f", symbol, price, pred)

                for symbol in list(self.positions.keys()):
                    price = spot_prices.get(symbol)
                    if price is None:
                        continue
                    if self.should_force_sell(symbol, price):
                        self.place_sell_order(symbol, price, "risk_control")
                    elif predictions.get(symbol, -1.0) < -BUY_SIGNAL_THRESHOLD:
                        self.place_sell_order(symbol, price, "model_turn_negative")

                ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                for symbol, pred in ranked:
                    if pred < BUY_SIGNAL_THRESHOLD or symbol in self.positions:
                        continue
                    qty = self.compute_order_quantity(spot_prices[symbol], spot_prices)
                    if qty > 0:
                        self.place_buy_order(symbol, spot_prices[symbol], qty)

                equity = self.total_equity(spot_prices)
                logging.info(
                    "ACCOUNT | cash=%.2f holdings=%.2f equity=%.2f realized=%.2f positions=%s",
                    self.cash,
                    self.market_value(spot_prices),
                    equity,
                    self.realized_pnl,
                    list(self.positions.keys()),
                )
            except Exception as exc:
                logging.exception("Cycle error: %s", exc)

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    broker = None
    if ENABLE_LIVE_TRADING:
        broker = HttpBrokerGateway(base_url=BROKER_BASE_URL, api_key=BROKER_API_KEY)

    system = QuantTradingSystem(
        symbols=SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        poll_interval=POLL_INTERVAL_SECONDS,
        model_name=MODEL_NAME,
        enable_live_trading=ENABLE_LIVE_TRADING,
        broker=broker,
    )

    # 启动前可先做回测：
    # for s in SYMBOLS:
    #     result = system.backtest_symbol(s)
    #     if result:
    #         logging.info("BACKTEST | %s", result)

    system.run()
