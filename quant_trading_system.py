"""AI量化交易系统（多模型 + 回测 + 可选实盘网关 + 可切换数据源）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Protocol

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
DATA_FETCH_RETRIES = 3
DATA_FETCH_RETRY_DELAY = 1.5

# 数据源配置：默认改为 efinance（按你的反馈替换 AkShare）
DATA_PROVIDERS = ["efinance", "akshare"]  # 按顺序故障转移

# 模型配置
MODEL_NAME = "xgboost"  # linear / xgboost / lstm / transformer
SEQUENCE_LENGTH = 20
NN_EPOCHS = 20
NN_LR = 1e-3

# 实盘相关配置
ENABLE_LIVE_TRADING = False
BROKER_BASE_URL = ""
BROKER_API_KEY = ""


@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime


@dataclass
class BacktestResult:
    symbol: str
    model_name: str
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int


class BrokerGateway(Protocol):
    def place_order(self, symbol: str, side: str, quantity: int, price: float) -> dict:
        ...


class HttpBrokerGateway:
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


class MarketDataProvider(Protocol):
    def get_spot_prices(self, symbols: list[str]) -> Dict[str, float]:
        ...

    def get_history(self, symbol: str, lookback_days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        ...


class EFinanceDataProvider:
    """efinance 数据源实现。"""

    @staticmethod
    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"缺少列，候选={candidates}, 实际={list(df.columns)}")

    def get_spot_prices(self, symbols: list[str]) -> Dict[str, float]:
        import efinance as ef

        # 优先按股票池定向拉取，减少全市场请求导致的JSON解析失败概率
        try:
            quote = ef.stock.get_realtime_quotes(stock_codes=symbols)
        except TypeError:
            quote = ef.stock.get_realtime_quotes(symbols)
        except Exception:
            quote = ef.stock.get_realtime_quotes()
        code_col = self._pick_col(quote, ["股票代码", "代码"])
        price_col = self._pick_col(quote, ["最新价", "最新价格"])
        rows = quote[quote[code_col].astype(str).isin(symbols)]
        result: Dict[str, float] = {}
        for _, row in rows.iterrows():
            try:
                result[str(row[code_col])] = float(row[price_col])
            except Exception:
                continue
        return result

    def get_history(self, symbol: str, lookback_days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        import efinance as ef

        beg = (start_date or "20180101").replace("-", "")
        end = (end_date or datetime.now().strftime("%Y%m%d")).replace("-", "")
        hist = ef.stock.get_quote_history(symbol, beg=beg, end=end, klt=101, fqt=1)
        if hist is None or hist.empty:
            return pd.DataFrame()

        date_col = self._pick_col(hist, ["日期"])
        open_col = self._pick_col(hist, ["开盘"])
        high_col = self._pick_col(hist, ["最高"])
        low_col = self._pick_col(hist, ["最低"])
        close_col = self._pick_col(hist, ["收盘"])
        out = hist[[date_col, open_col, high_col, low_col, close_col]].copy()
        out.columns = ["date", "open", "high", "low", "close"]
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").tail(lookback_days).reset_index(drop=True)
        return out


class AkshareDataProvider:
    """AkShare 数据源实现（可选备用）。"""

    def get_spot_prices(self, symbols: list[str]) -> Dict[str, float]:
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        rows = spot[spot["代码"].isin(symbols)][["代码", "最新价"]]
        return {str(row["代码"]): float(row["最新价"]) for _, row in rows.iterrows()}

    def get_history(self, symbol: str, lookback_days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        import akshare as ak

        if start_date or end_date:
            hist = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=(start_date or "20180101").replace("-", ""),
                end_date=(end_date or datetime.now().strftime("%Y%m%d")).replace("-", ""),
                adjust="qfq",
            )
        else:
            hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        if hist.empty:
            return pd.DataFrame()
        out = hist.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"})
        out["date"] = pd.to_datetime(out["date"])
        return out[["date", "open", "high", "low", "close"]].sort_values("date").tail(lookback_days).reset_index(drop=True)


def build_data_provider(name: str) -> MarketDataProvider:
    if name.lower() == "efinance":
        return EFinanceDataProvider()
    if name.lower() == "akshare":
        return AkshareDataProvider()
    raise ValueError(f"不支持的数据源: {name}")


def _normalize_provider_names(names: list[str]) -> list[str]:
    seen = set()
    out = []
    for n in names:
        k = n.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out or ["efinance", "akshare"]


class QuantTradingSystem:
    FEATURE_COLS = ["ret_1d", "mom_5d", "mom_10d", "vol_10d", "rsi_14"]

    def __init__(
        self,
        symbols: list[str],
        initial_capital: float = INITIAL_CAPITAL,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        model_name: str = MODEL_NAME,
        enable_live_trading: bool = ENABLE_LIVE_TRADING,
        broker: Optional[BrokerGateway] = None,
        data_provider_names: Optional[list[str]] = None,
    ) -> None:
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0

        self.model_name = model_name.lower()
        self.enable_live_trading = enable_live_trading
        self.broker = broker
        self.data_provider_names = data_provider_names or list(DATA_PROVIDERS)

        self.last_spot_prices: Dict[str, float] = {}
        self.last_history: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _with_retry(func, *args, **kwargs):
        last_error: Optional[Exception] = None
        for attempt in range(1, DATA_FETCH_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                sleep_s = DATA_FETCH_RETRY_DELAY * (2 ** (attempt - 1))
                logging.warning("Data fetch failed (%s/%s): %s; retry in %.1fs", attempt, DATA_FETCH_RETRIES, exc, sleep_s)
                time.sleep(sleep_s)
        raise RuntimeError(f"Data fetch failed after retries: {last_error}")

    def _iter_providers(self):
        for name in _normalize_provider_names(self.data_provider_names):
            yield name, build_data_provider(name)

    def get_history(self, symbol: str, lookback_days: int = MODEL_LOOKBACK_DAYS + 120) -> Optional[pd.DataFrame]:
        last_error = None
        for provider_name, provider in self._iter_providers():
            try:
                history = self._with_retry(provider.get_history, symbol, lookback_days)
                if history.empty or len(history) < 60:
                    continue
                return history
            except Exception as exc:
                last_error = exc
                logging.warning("history provider failed: %s symbol=%s err=%s", provider_name, symbol, exc)
        if last_error:
            raise RuntimeError(f"all providers failed for history: {last_error}")
        return None

    @staticmethod
    def build_features(hist: pd.DataFrame) -> pd.DataFrame:
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
    def _predict_linear(train_df: pd.DataFrame, latest_row: pd.Series, feature_cols: list[str]) -> float:
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target_next_ret"].to_numpy(dtype=float)
        means = x_train.mean(axis=0)
        stds = np.where(x_train.std(axis=0) == 0, 1.0, x_train.std(axis=0))
        design = np.column_stack([np.ones(len(x_train)), (x_train - means) / stds])
        coefs, *_ = np.linalg.lstsq(design, y_train, rcond=None)
        x_latest = latest_row[feature_cols].to_numpy(dtype=float)
        return float(coefs[0] + np.dot(coefs[1:], (x_latest - means) / stds))

    @staticmethod
    def _predict_xgboost(train_df: pd.DataFrame, latest_row: pd.Series, feature_cols: list[str]) -> Optional[float]:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None
        model = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
        model.fit(train_df[feature_cols], train_df["target_next_ret"])
        return float(model.predict(latest_row[feature_cols].to_frame().T)[0])

    @staticmethod
    def _predict_lstm_or_transformer(train_df: pd.DataFrame, latest_window: np.ndarray, feature_cols: list[str], model_name: str) -> Optional[float]:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return None

        x = train_df[feature_cols].to_numpy(dtype=np.float32)
        y = train_df["target_next_ret"].to_numpy(dtype=np.float32)
        if len(x) <= SEQUENCE_LENGTH + 1:
            return None

        means = x.mean(axis=0)
        stds = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
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
            loss = loss_fn(model(X), Y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            return float(model(torch.tensor(latest_window[np.newaxis, :, :], dtype=torch.float32)).item())

    def predict_next_return(self, feature_df: pd.DataFrame) -> Optional[float]:
        clean = feature_df.dropna(subset=self.FEATURE_COLS + ["target_next_ret"]).copy()
        if len(clean) < MIN_TRAIN_SAMPLES:
            return None
        train_df = clean.tail(MODEL_LOOKBACK_DAYS)
        latest = feature_df.iloc[-1]
        if latest[self.FEATURE_COLS].isna().any():
            return None

        if self.model_name == "linear":
            return self._predict_linear(train_df, latest, self.FEATURE_COLS)
        if self.model_name == "xgboost":
            return self._predict_xgboost(train_df, latest, self.FEATURE_COLS) or self._predict_linear(train_df, latest, self.FEATURE_COLS)
        if self.model_name in {"lstm", "transformer"}:
            latest_window_df = feature_df.dropna(subset=self.FEATURE_COLS).tail(SEQUENCE_LENGTH)
            if len(latest_window_df) < SEQUENCE_LENGTH:
                return None
            pred = self._predict_lstm_or_transformer(train_df, latest_window_df[self.FEATURE_COLS].to_numpy(dtype=np.float32), self.FEATURE_COLS, self.model_name)
            return pred or self._predict_linear(train_df, latest, self.FEATURE_COLS)
        return self._predict_linear(train_df, latest, self.FEATURE_COLS)

    def fetch_spot_prices(self, symbols: list[str]) -> Dict[str, float]:
        last_error = None
        for provider_name, provider in self._iter_providers():
            try:
                prices = self._with_retry(provider.get_spot_prices, symbols)
                if prices:
                    return prices
            except Exception as exc:
                last_error = exc
                logging.warning("spot provider failed: %s err=%s", provider_name, exc)
        if last_error:
            raise RuntimeError(f"all providers failed for spot: {last_error}")
        return {}

    def get_history_safe(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            hist = self.get_history(symbol)
            if hist is not None and not hist.empty:
                self.last_history[symbol] = hist
            return hist
        except Exception as exc:
            logging.error("History fetch error for %s: %s", symbol, exc)
            return self.last_history.get(symbol)

    def fetch_spot_prices_safe(self, symbols: list[str]) -> Dict[str, float]:
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
        affordable = min(self.total_equity(spot_prices) * MAX_POSITION_RATIO, self.cash)
        return max(0, int(affordable // (price * LOT_SIZE)) * LOT_SIZE)

    def should_force_sell(self, symbol: str, price: float) -> bool:
        pos = self.positions[symbol]
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        return pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT

    def _send_live_order(self, symbol: str, side: str, quantity: int, price: float) -> None:
        if not self.enable_live_trading:
            return
        if self.broker is None:
            raise RuntimeError("ENABLE_LIVE_TRADING=True 但未提供 broker 网关")
        logging.info("LIVE_ORDER | %s", self.broker.place_order(symbol=symbol, side=side, quantity=quantity, price=price))

    def place_buy_order(self, symbol: str, price: float, quantity: int) -> None:
        cost = price * quantity
        if quantity <= 0 or cost > self.cash:
            return
        self._send_live_order(symbol, "BUY", quantity, price)
        self.cash -= cost
        self.positions[symbol] = Position(symbol=symbol, quantity=quantity, entry_price=price, entry_time=datetime.now())
        logging.info("BUY  | symbol=%s qty=%s price=%.2f cash=%.2f", symbol, quantity, price, self.cash)

    def place_sell_order(self, symbol: str, price: float, reason: str) -> None:
        pos = self.positions.pop(symbol)
        self._send_live_order(symbol, "SELL", pos.quantity, price)
        pnl = (price - pos.entry_price) * pos.quantity
        self.realized_pnl += pnl
        self.cash += price * pos.quantity
        logging.info("SELL | symbol=%s qty=%s price=%.2f pnl=%.2f reason=%s cash=%.2f", symbol, pos.quantity, price, pnl, reason, self.cash)

    def backtest_symbol(self, symbol: str, start_date: str = "2019-01-01", end_date: str = "2024-12-31") -> Optional[BacktestResult]:
        try:
            hist = None
            last_error = None
            for provider_name, provider in self._iter_providers():
                try:
                    hist = self._with_retry(provider.get_history, symbol, MODEL_LOOKBACK_DAYS + 400, start_date, end_date)
                    if hist is not None and not hist.empty:
                        break
                except Exception as exc:
                    last_error = exc
                    logging.warning("backtest provider failed: %s symbol=%s err=%s", provider_name, symbol, exc)
            if hist is None or hist.empty:
                raise RuntimeError(last_error or "no backtest data")
        except Exception as exc:
            logging.error("Backtest data fetch failed for %s: %s", symbol, exc)
            return None
        if hist.empty or len(hist) < MIN_TRAIN_SAMPLES + 30:
            return None

        feat = self.build_features(hist)
        clean = feat.dropna(subset=self.FEATURE_COLS + ["target_next_ret"]).reset_index(drop=True)
        returns, equity_curve = [], [1.0]
        win, trades = 0, 0

        for i in range(MIN_TRAIN_SAMPLES, len(clean) - 1):
            train_slice = clean.iloc[:i].tail(MODEL_LOOKBACK_DAYS)
            latest_row = clean.iloc[i]
            next_ret = float(clean.iloc[i]["target_next_ret"])
            pred_df = pd.concat([train_slice, latest_row.to_frame().T], ignore_index=True)
            pred = self.predict_next_return(pred_df)

            if pred is None:
                strategy_ret = 0.0
            elif pred > BUY_SIGNAL_THRESHOLD:
                strategy_ret = next_ret
                trades += 1
                win += int(strategy_ret > 0)
            elif pred < -BUY_SIGNAL_THRESHOLD:
                strategy_ret = -next_ret
                trades += 1
                win += int(strategy_ret > 0)
            else:
                strategy_ret = 0.0

            returns.append(strategy_ret)
            equity_curve.append(equity_curve[-1] * (1 + strategy_ret))

        ret_arr = np.array(returns, dtype=float)
        total_return = equity_curve[-1] - 1
        annualized = (1 + total_return) ** (252 / max(1, len(ret_arr))) - 1
        vol = ret_arr.std() * np.sqrt(252)
        sharpe = 0.0 if vol == 0 else (ret_arr.mean() * 252) / vol
        curve = np.array(equity_curve)
        peak = np.maximum.accumulate(curve)
        max_dd = abs(((curve - peak) / peak).min())
        win_rate = 0.0 if trades == 0 else win / trades

        return BacktestResult(symbol, self.model_name, float(total_return), float(annualized), float(sharpe), float(max_dd), float(win_rate), int(trades))

    def run(self) -> None:
        logging.info("System started. symbols=%s model=%s data_providers=%s live=%s", self.symbols, self.model_name, _normalize_provider_names(self.data_provider_names), self.enable_live_trading)
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
                    pred = self.predict_next_return(self.build_features(hist))
                    if pred is not None:
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

                for symbol, pred in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                    if pred < BUY_SIGNAL_THRESHOLD or symbol in self.positions:
                        continue
                    qty = self.compute_order_quantity(spot_prices[symbol], spot_prices)
                    if qty > 0:
                        self.place_buy_order(symbol, spot_prices[symbol], qty)

                equity = self.total_equity(spot_prices)
                logging.info("ACCOUNT | cash=%.2f holdings=%.2f equity=%.2f realized=%.2f positions=%s", self.cash, self.market_value(spot_prices), equity, self.realized_pnl, list(self.positions.keys()))
            except Exception as exc:
                logging.exception("Cycle error: %s", exc)

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    broker = HttpBrokerGateway(BROKER_BASE_URL, BROKER_API_KEY) if ENABLE_LIVE_TRADING else None
    system = QuantTradingSystem(
        symbols=SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        poll_interval=POLL_INTERVAL_SECONDS,
        model_name=MODEL_NAME,
        enable_live_trading=ENABLE_LIVE_TRADING,
        broker=broker,
        data_provider_names=DATA_PROVIDERS,
    )
    system.run()
