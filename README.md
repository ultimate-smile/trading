# AI量化交易系统（多模型 + 回测 + 可选实盘网关）

你反馈的报错本质是：数据源接口返回了非预期内容（如空响应/反爬页面），导致 JSON 解析失败。  
本次已做两层修复：

1. **新增 Eastmoney 直连数据源**：使用 `requests.Session(trust_env=False)`，默认禁用系统代理，规避 `ProxyError`。  
2. **efinance 调用方式优化**：优先按股票池定向拉取实时行情，避免全市场大请求。  
3. **多数据源故障转移**：按顺序自动切换 `eastmoney_direct -> efinance -> akshare`。

## 关键变化

- 默认数据源链：`DATA_PROVIDERS = ["eastmoney_direct", "efinance", "akshare"]`
- 可配置数据源优先级（按顺序故障转移）
- 保留重试 + 缓存兜底：网络抖动时尽量不中断交易循环
- 支持模型：`linear / xgboost / lstm / transformer`
- 支持回测指标：`Sharpe / 最大回撤 / 胜率 / 年化 / 总收益`

## 快速开始

```bash
pip install -r requirements.txt
python quant_trading_system.py
```

## 常见配置

- `DATA_PROVIDERS`：数据源优先级列表，例如：
  - `['eastmoney_direct', 'efinance', 'akshare']`
  - `['efinance', 'akshare']`
  - `['akshare']`
- `MODEL_NAME`：模型类型
- `ENABLE_LIVE_TRADING`：是否开启实盘
- `INITIAL_CAPITAL`：初始资金
- `MAX_POSITION_RATIO`：单标的仓位上限

## 为什么你之前还会失败

你日志中的 `Expecting value: line 1 column 1 (char 0)` 通常是接口返回了空文本/非JSON内容。单一数据源即使重试也可能连续失败。  
现在系统会自动切换到下一个数据源，显著降低整轮 `No spot prices` 的概率。

## 说明

- 代码已支持“可接实盘”架构（HTTP broker gateway），但上线前必须先做长周期回测和小资金灰度验证。


## 关于你日志里的 ProxyError

你这次日志里已经出现 `Unable to connect to proxy`，这说明运行环境里有代理链路问题。
新版本的 `eastmoney_direct` 会默认忽略系统代理环境变量（`trust_env=False`），可绕开这类代理故障。
