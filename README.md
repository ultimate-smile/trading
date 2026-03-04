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

## 后续可接入方向

- 对接券商交易 API（在 `place_buy_order` / `place_sell_order` 中实现实盘下单）
- 接入更多因子（行业、财务、盘口）
- 升级模型（XGBoost、LSTM、Transformer）
- 增加回测模块（按日线历史评估 Sharpe、回撤、胜率）

## 免责声明

本项目仅用于学习与研究，不构成任何投资建议，实盘交易风险自担。
