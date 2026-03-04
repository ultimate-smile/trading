# AI量化交易系统（多模型 + 回测 + 可选实盘网关）

已根据你的反馈把默认数据源从 **AkShare 切换为 efinance**，用于降低 `RemoteDisconnected` 这类上游中断影响。

## 关键变化

- 默认数据源：`DATA_PROVIDER="efinance"`
- 数据源可切换：`efinance / akshare`
- 保留重试 + 缓存兜底：网络抖动时尽量不中断交易循环
- 支持模型：`linear / xgboost / lstm / transformer`
- 支持回测指标：`Sharpe / 最大回撤 / 胜率 / 年化 / 总收益`

## 快速开始

```bash
pip install -r requirements.txt
python quant_trading_system.py
```

## 常见配置

- `DATA_PROVIDER`：数据源，默认 `efinance`
- `MODEL_NAME`：模型类型
- `ENABLE_LIVE_TRADING`：是否开启实盘
- `INITIAL_CAPITAL`：初始资金
- `MAX_POSITION_RATIO`：单标的仓位上限

## 你遇到的错误为何能缓解

你日志里的错误来自数据源请求被远端中断：
`requests.exceptions.ConnectionError: RemoteDisconnected(...)`

本版本处理：
1. 改用 `efinance` 作为默认行情/历史来源；
2. 对数据请求统一做重试（指数退避）；
3. 若实时行情失败，回退到最近一次缓存价格；
4. 若历史失败，回退到该标的最近一次缓存历史。

## 说明

- 代码已支持“可接实盘”架构（HTTP broker gateway），但上线前必须先做长周期回测和小资金灰度验证。
