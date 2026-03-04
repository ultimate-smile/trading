# AI量化交易系统（支持多模型 + 回测 + 可选实盘网关）

本项目已从“玩具脚本”升级为可实盘接入架构，包含：

- **模型升级**：`Linear`、`XGBoost`、`LSTM`、`Transformer` 四种预测路径
- **回测模块**：按日线滚动训练/预测，输出 `Sharpe`、`最大回撤`、`胜率`、`年化`、`总收益`
- **交易执行层**：纸上交易 + 可选 HTTP 券商网关（可接真实下单）
- **资金与风控**：10W 初始资金、仓位控制、止损/止盈、信号反转退出

## 快速开始

```bash
pip install -r requirements.txt
python quant_trading_system.py
```

## 关键参数

- `MODEL_NAME`：`linear / xgboost / lstm / transformer`
- `ENABLE_LIVE_TRADING`：是否开启实盘委托
- `BROKER_BASE_URL`、`BROKER_API_KEY`：实盘网关配置
- `INITIAL_CAPITAL`：初始资金（默认 `100000`）
- `MAX_POSITION_RATIO`：单标的最大仓位比例
- `STOP_LOSS_PCT`、`TAKE_PROFIT_PCT`：止损/止盈比例

## 关键方法

- `build_features`：构造因子与监督标签
- `predict_next_return`：统一模型入口，自动按 `MODEL_NAME` 推理
- `backtest_symbol`：单标的日线回测，输出核心绩效指标
- `place_buy_order` / `place_sell_order`：下单执行（可切换为实盘网关）

## 回测示例

在脚本 `__main__` 中取消注释：

```python
for s in SYMBOLS:
    result = system.backtest_symbol(s)
    if result:
        logging.info("BACKTEST | %s", result)
```

## 实盘接入建议

1. 先在 `ENABLE_LIVE_TRADING=False` 下做足够历史回测 + 仿真验证；
2. 对接你自己的券商中间层（HTTP），并实现风控白名单、限价保护、订单状态回报；
3. 小资金灰度上线，再逐步放量；
4. 生产部署需增加：交易日历、重试机制、断线重连、日志审计、告警。

## 说明

- 本系统代码已经具备“可接实盘”的执行接口，但你仍需要对接真实券商API并自行承担交易风险。

## 常见报错与处理

- 报错：`requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected(...))`
  - 原因：AkShare 上游数据源瞬时断连/限流/网络抖动。
  - 当前版本处理：
    1. 对行情与历史数据请求自动重试（指数退避）；
    2. 若实时行情拉取失败，自动回退到最近一次缓存价格继续跑循环；
    3. 若单标的历史数据拉取失败，自动回退该标的最近一次缓存历史。
  - 建议：将轮询间隔适当调大，或部署在网络更稳定的环境中。
