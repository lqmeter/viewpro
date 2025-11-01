# Crypto Quant Lab

一个基于 Streamlit 的加密货币量化交易平台，提供行情看盘、Pine 脚本指标定制以及基础策略回测功能。

## 功能特点

- 支持选择主流加密货币交易对（BTC、ETH、SOL、BNB）。
- 提供 15 分钟至 1 天多种 K 线周期，自动从 Yahoo Finance 抓取历史数据。
- 内置简化版 Pine 执行器，可通过侧边栏编辑脚本，自定义指标与交易信号。
- 可视化展示蜡烛图、叠加指标、独立指标面板以及多空信号。
- 内置长线回测引擎，统计策略收益率、最大回撤与权益曲线。

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

应用启动后，打开浏览器访问终端输出的地址。侧边栏可选择交易对、周期、回看天数并编辑 Pine 脚本。

## Pine 脚本支持

为了便于离线执行，本项目实现了 Pine Script 的简化子集，支持以下能力：

- `indicator`、`plot` 基础调用。
- `sma`、`ema`、`rsi`、`highest`、`lowest` 等常见指标函数。
- `crossover`、`crossunder` 信号辅助函数。
- `input.int`、`input.float` 读取参数。
- 通过定义 `long`、`short` 序列触发开平仓，驱动回测。

> **提示**：脚本中暂未实现完整的 Pine 语法，请保持语句的简洁性，例如单行赋值、单条件表达式等。

## 示例脚本

```pine
// 基于双均线的示例策略
indicator(title="双均线策略", overlay=True)
fastLength = input.int(20, title="快线周期")
slowLength = input.int(60, title="慢线周期")
fastMA = sma(close, fastLength)
slowMA = sma(close, slowLength)
plot(fastMA, title="快线", color=color.orange, linewidth=2)
plot(slowMA, title="慢线", color=color.blue, linewidth=2)
long = crossover(fastMA, slowMA)
short = crossunder(fastMA, slowMA)
```

将脚本粘贴到侧边栏 Pine 编辑器后，即可实时查看指标与回测结果。
