# AI增强量化交易系统（Python）

这是一个基于 **Python + AkShare** 的示例量化交易脚本，内置：

- 初始资金：`100000`（10W）
- 交易标的（可改）：`000988`、`688387`
- 轻量 AI 预测：用历史因子训练线性模型，预测下一交易日收益率
- 资金管理与风控：仓位上限、止损、止盈

> ⚠️ 说明：任何策略都无法保证“每日最大化收益”，本项目目标是提供一个可运行、可扩展、带风险控制的 AI 量化框架。

## 策略逻辑（当前版本）

每轮轮询时，系统会：

1. 获取最新行情与历史数据。
2. 计算因子：
   - `ret_1d`（1日收益）
   - `mom_5d`（5日动量）
   - `mom_10d`（10日动量）
   - `vol_10d`（10日波动）
   - `rsi_14`（14日 RSI）
3. 使用最小二乘法拟合线性模型，预测下一日收益率。
4. 执行交易：
   - **买入**：预测收益率高于阈值（默认 `0.2%`）
   - **卖出**：
     - 预测转负（低于 `-0.2%`）
     - 或触发风控：止损 `3%` / 止盈 `6%`
5. 单标的最大仓位：账户权益的 `40%`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python quant_trading_system.py
```

## 可调参数

在 `quant_trading_system.py` 顶部可调整：

- `SYMBOLS`：股票池
- `INITIAL_CAPITAL`：初始资金
- `POLL_INTERVAL_SECONDS`：轮询间隔
- `MAX_POSITION_RATIO`：最大仓位
- `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`：止损/止盈
- `BUY_SIGNAL_THRESHOLD`：买入阈值


## 关键参数 / 关键方法 / 主要变量说明

### 关键参数（`quant_trading_system.py` 顶部）
- `INITIAL_CAPITAL`：初始资金，默认 100000（10W）。
- `MAX_POSITION_RATIO`：单标的最大仓位比例，默认 0.4。
- `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT`：止损/止盈阈值，默认 3% / 6%。
- `BUY_SIGNAL_THRESHOLD`：买入信号阈值，预测收益率高于该值才开仓。
- `MODEL_LOOKBACK_DAYS` / `MIN_TRAIN_SAMPLES`：模型训练窗口与最小样本数。

### 关键方法（`QuantTradingSystem`）
- `get_history`：获取并标准化历史行情。
- `build_features`：构建模型特征与监督标签。
- `predict_next_return`：训练线性模型并给出最新时点预测。
- `compute_order_quantity`：按仓位和现金约束计算下单股数。
- `should_force_sell`：按止损/止盈判断是否强制平仓。
- `run`：主循环，串联数据、预测、交易和风控。

### 主要变量
- `cash`：可用现金。
- `positions`：当前持仓字典（`symbol -> Position`）。
- `realized_pnl`：已实现盈亏（仅平仓后变动）。
- `predictions`：本轮每个标的的预测收益率。

## 后续可接入方向

- 对接券商交易 API（在 `place_buy_order` / `place_sell_order` 中实现实盘下单）
- 接入更多因子（行业、财务、盘口）
- 升级模型（XGBoost、LSTM、Transformer）
- 增加回测模块（按日线历史评估 Sharpe、回撤、胜率）

## 免责声明

本项目仅用于学习与研究，不构成任何投资建议，实盘交易风险自担。
