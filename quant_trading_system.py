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
DATA_PROVIDERS = ["eastmoney_direct", "efinance", "akshare"]  # 按顺序故障转移

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
        """券商下单抽象接口。

        参数:
        - symbol: 股票代码
        - side: BUY/SELL
        - quantity: 下单数量（股）
        - price: 委托价格

        返回:
        - 券商网关返回的原始响应字典。
        """
        ...


class HttpBrokerGateway:
    def __init__(self, base_url: str, api_key: str) -> None:
        """初始化 HTTP 实盘网关客户端。"""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def place_order(self, symbol: str, side: str, quantity: int, price: float) -> dict:
        """通过 HTTP 中间层发送下单请求。"""
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
        """获取实时行情，返回 {symbol: latest_price}。"""
        ...

    def get_history(self, symbol: str, lookback_days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取历史K线并标准化为 date/open/high/low/close。"""
        ...




class EastmoneyDirectDataProvider:
    """东方财富直连数据源（requests直连，默认禁用系统代理）。"""

    def __init__(self, disable_system_proxy: bool = True) -> None:
        """初始化直连会话。

        - disable_system_proxy=True 时，requests 不读取系统代理环境变量，
          用于规避 `ProxyError: Unable to connect to proxy`。
        """
        import requests

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; QuantTradingSystem/1.0)",
            "Referer": "https://quote.eastmoney.com/",
        })
        if disable_system_proxy:
            self.session.trust_env = False

    @staticmethod
    def _secid(symbol: str) -> str:
        """将A股代码转换为东财 secid。

        上海: 1.xxx（60/68/11）
        深圳: 0.xxx（其余A股常见前缀）
        """
        if symbol.startswith(("60", "68", "11")):
            return f"1.{symbol}"
        return f"0.{symbol}"

    def get_spot_prices(self, symbols: list[str]) -> Dict[str, float]:
        """逐个股票拉取实时价格，避免全市场大请求。"""
        result: Dict[str, float] = {}
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        for symbol in symbols:
            params = {
                "fltt": "2",
                "invt": "2",
                "fields": "f43",
                "secid": self._secid(symbol),
            }
            r = self.session.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            val = (((data or {}).get("data") or {}).get("f43"))
            if val is None:
                continue
            # 东财价格通常放大100倍
            price = float(val) / 100
            if price > 0:
                result[symbol] = price
        return result

    def get_history(self, symbol: str, lookback_days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """通过东财K线接口获取日线历史。"""
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": self._secid(symbol),
            "klt": "101",  # 日线
            "fqt": "1",    # 前复权
            "lmt": str(max(lookback_days, 200)),
            "end": "20500000",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        }
        if start_date:
            params["beg"] = start_date.replace("-", "")
        else:
            params["beg"] = "20100101"
        if end_date:
            params["end"] = end_date.replace("-", "")

        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        klines = (((data or {}).get("data") or {}).get("klines")) or []
        if not klines:
            return pd.DataFrame()

        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            rows.append({
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["date"] = pd.to_datetime(out["date"])
        return out[["date", "open", "high", "low", "close"]].sort_values("date").tail(lookback_days).reset_index(drop=True)


class EFinanceDataProvider:
    """efinance 数据源实现。"""

    @staticmethod
    def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
        """从候选列名中找到 DataFrame 实际存在的一列。"""
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"缺少列，候选={candidates}, 实际={list(df.columns)}")

    def get_spot_prices(self, symbols: list[str]) -> Dict[str, float]:
        """获取股票池实时价格（efinance 实现）。"""
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
        """获取单标的历史日线并映射到统一列名。"""
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
        """获取股票池实时价格（AkShare 实现）。"""
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        rows = spot[spot["代码"].isin(symbols)][["代码", "最新价"]]
        return {str(row["代码"]): float(row["最新价"]) for _, row in rows.iterrows()}

    def get_history(self, symbol: str, lookback_days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取单标的历史日线并映射到统一列名。"""
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
    """按名称构建数据源实现实例。"""
    if name.lower() == "eastmoney_direct":
        return EastmoneyDirectDataProvider(disable_system_proxy=True)
    if name.lower() == "efinance":
        return EFinanceDataProvider()
    if name.lower() == "akshare":
        return AkshareDataProvider()
    raise ValueError(f"不支持的数据源: {name}")


def _normalize_provider_names(names: list[str]) -> list[str]:
    """规范化数据源名称（去空白、去重、转小写）。"""
    seen = set()
    out = []
    for n in names:
        k = n.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out or ["eastmoney_direct", "efinance", "akshare"]


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
        """初始化交易系统。

        说明：
        - 初始化账户状态（cash/positions/realized_pnl）；
        - 注入模型参数、实盘开关、数据源优先级列表；
        - 创建缓存，用于行情/历史数据失败时兜底。
        """
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
        """对外部I/O调用执行重试。

        使用指数退避：第 n 次重试等待 `DATA_FETCH_RETRY_DELAY * 2^(n-1)` 秒。
        """
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
        """按配置顺序返回可用数据源实例，用于故障转移。"""
        for name in _normalize_provider_names(self.data_provider_names):
            yield name, build_data_provider(name)

    def get_history(self, symbol: str, lookback_days: int = MODEL_LOOKBACK_DAYS + 120) -> Optional[pd.DataFrame]:
        """获取单标的历史数据。

        流程：
        1) 按数据源优先级依次尝试；
        2) 每个数据源内部带重试；
        3) 至少满足最小样本长度，否则返回 None。
        """
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
        """构建模型因子与监督学习标签。

        因子包括：1日收益、5/10日动量、10日波动率、RSI14；
        标签为下一交易日收益率 `target_next_ret`。
        """
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
        """用最小二乘线性回归预测下一日收益率。"""
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
        """用 XGBoost 回归预测；若未安装 xgboost 则返回 None。"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None
        model = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
        model.fit(train_df[feature_cols], train_df["target_next_ret"])
        return float(model.predict(latest_row[feature_cols].to_frame().T)[0])

    @staticmethod
    def _predict_lstm_or_transformer(train_df: pd.DataFrame, latest_window: np.ndarray, feature_cols: list[str], model_name: str) -> Optional[float]:
        """用 LSTM/Transformer 进行序列建模预测。

        说明：该方法依赖 PyTorch；当依赖缺失时返回 None 触发上层回退模型。
        """
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
                """构造轻量 LSTM 回归头。"""
                super().__init__()
                self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                """前向传播，输出序列末端时刻预测值。"""
                out, _ = self.lstm(z)
                return self.fc(out[:, -1, :])

        class TransformerRegressor(nn.Module):
            def __init__(self, input_dim: int) -> None:
                """构造轻量 Transformer Encoder 回归头。"""
                super().__init__()
                self.proj = nn.Linear(input_dim, 32)
                layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.fc = nn.Linear(32, 1)

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                """前向传播，输出序列末端时刻预测值。"""
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
        """统一模型预测入口。

        会根据 `self.model_name` 调用对应模型，必要时回退到线性模型，
        返回“最新一行特征”对应的下一日收益预测。
        """
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
        """获取实时行情并在数据源间自动故障转移。"""
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
        """安全获取历史数据：失败时回退到本地缓存。"""
        try:
            hist = self.get_history(symbol)
            if hist is not None and not hist.empty:
                self.last_history[symbol] = hist
            return hist
        except Exception as exc:
            logging.error("History fetch error for %s: %s", symbol, exc)
            return self.last_history.get(symbol)

    def fetch_spot_prices_safe(self, symbols: list[str]) -> Dict[str, float]:
        """安全获取实时行情：失败时回退到缓存价格。"""
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
        """计算当前持仓总市值。"""
        return sum(pos.quantity * spot_prices.get(s, pos.entry_price) for s, pos in self.positions.items())

    def total_equity(self, spot_prices: Dict[str, float]) -> float:
        """计算账户总权益 = 现金 + 持仓市值。"""
        return self.cash + self.market_value(spot_prices)

    def compute_order_quantity(self, price: float, spot_prices: Dict[str, float]) -> int:
        """按仓位限制和可用现金计算下单股数（按整手）。"""
        affordable = min(self.total_equity(spot_prices) * MAX_POSITION_RATIO, self.cash)
        return max(0, int(affordable // (price * LOT_SIZE)) * LOT_SIZE)

    def should_force_sell(self, symbol: str, price: float) -> bool:
        """判断是否命中止损或止盈条件。"""
        pos = self.positions[symbol]
        pnl_pct = (price - pos.entry_price) / pos.entry_price
        return pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT

    def _send_live_order(self, symbol: str, side: str, quantity: int, price: float) -> None:
        """根据实盘开关发送订单到券商网关。"""
        if not self.enable_live_trading:
            return
        if self.broker is None:
            raise RuntimeError("ENABLE_LIVE_TRADING=True 但未提供 broker 网关")
        logging.info("LIVE_ORDER | %s", self.broker.place_order(symbol=symbol, side=side, quantity=quantity, price=price))

    def place_buy_order(self, symbol: str, price: float, quantity: int) -> None:
        """执行买入流程：下单（可选）+ 更新账户状态。"""
        cost = price * quantity
        if quantity <= 0 or cost > self.cash:
            return
        self._send_live_order(symbol, "BUY", quantity, price)
        self.cash -= cost
        self.positions[symbol] = Position(symbol=symbol, quantity=quantity, entry_price=price, entry_time=datetime.now())
        logging.info("BUY  | symbol=%s qty=%s price=%.2f cash=%.2f", symbol, quantity, price, self.cash)

    def place_sell_order(self, symbol: str, price: float, reason: str) -> None:
        """执行卖出流程：下单（可选）+ 更新盈亏与现金。"""
        pos = self.positions.pop(symbol)
        self._send_live_order(symbol, "SELL", pos.quantity, price)
        pnl = (price - pos.entry_price) * pos.quantity
        self.realized_pnl += pnl
        self.cash += price * pos.quantity
        logging.info("SELL | symbol=%s qty=%s price=%.2f pnl=%.2f reason=%s cash=%.2f", symbol, pos.quantity, price, pnl, reason, self.cash)

    def backtest_symbol(self, symbol: str, start_date: str = "2019-01-01", end_date: str = "2024-12-31") -> Optional[BacktestResult]:
        """执行单标的日线回测。

        回测采用滚动训练-滚动预测方式，输出：
        - 总收益、年化收益、夏普比率
        - 最大回撤、胜率、交易次数
        """
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
        """系统主循环。

        每个轮询周期执行：
        1) 获取行情；
        2) 计算信号；
        3) 先风控卖出，再择优买入；
        4) 输出账户状态日志。
        """
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
