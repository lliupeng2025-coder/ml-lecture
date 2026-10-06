# 机器学习全链路讲解 · 震荡行情 DD 判断法（项目档案）

> 记录在案：2026-10-06 ｜ 状态：已完成真实数据接入与 11 讲课程详解
> 备注：本课程的机器学习建模实验将使用 **微软 Qlib** 平台，需先下载安装（见文末）。

## 项目概述

一个自包含的交互式讲解页面（HTML + Canvas，无外部依赖），内容三部分：

1. **机器学习全链路规划**：数据获取 → 自定义轴 → 数据标签 → 特征工程/降维 → 模型训练/回测/评估 → 模型组合 → 模型推理/实盘的七步闭环。
2. **课程内容详解（11 讲）**：机器学习基础、数据获取与分析、自定义非时间轴（DollarBar / C_1 轴）、数据标签、特征工程（含熵）、数据降维、模型训练（防过拟合）、模型通用性与专用性、模型组合、模型推理与实盘、扩展阅读。
3. **重点：震荡行情判断 —— DD（Data Difference，分布差异）法**（真实数据）：
   - Time Line DD：RB99 主力连续 30m 线 Δclose 的经验分布 vs 正态分布；
   - Turnover Line DD（PD）：30m 换手率（vol/oi）分布差异，交叉验证；
   - 六年滚动 DD 轨迹 + 行情状态判定（震荡/趋势/过渡带）；
   - 交互演示（模拟数据）：拖动滑块在趋势/震荡之间切换，观察分布形态变化。

## 真实数据口径（严格无未来函数）

- 数据源：`D:\data\future_1m\YYYY\YYYYMMDD.parquet`（1m 分合约行情），2019-01-02 – 2025-01-27，1,474 个交易日，聚合后 **17,280 根 30m K 线**（与课程案例同一口径）。
- 主力判定：第 d 日主力 = 第 d-1 日成交量最大合约（并列比持仓量），单日滞后；首日 bootstrap。
- 换月复权：价差回溯调整（back-adjust），共 19 次换月；2025 年 1 月仅有供应商主力连续 `RB8888`，仅用于信号拼接并在接缝处价差对齐。
- 30m 聚合：交易时段自然 30 分钟分桶，收盘 = 桶内最后一分钟收盘价；夜盘按交易日归属排在日前。
- 滚动指标：滚动 DD 窗口 W = 240 根（≈20 交易日），只用 [t−239, t]；状态阈值取扩展窗口 q75/q35 并滞后一期。
- 关键指标：DD = 0.391，峰度 = 18.7（正态 = 3），峰部比值 1.66×；换手率 PD = 0.540；状态分布：震荡 5,354 / 趋势 6,619 / 过渡 4,826 根。

## 目录结构

```
ml-lecture/
├── index.html          # 讲解页面（全部图表内嵌 Canvas 绘制，真实数据由 real_data.js 注入）
├── real_data.js        # 真实数据（RB99 30m 价格、滚动 DD、状态标签、KDE 曲线）
├── build_real_data.py  # 数据构建与 DD 计算脚本（因果口径，可重复执行）
├── package.json        # 预览脚本
├── cache/              # 中间产物（30m bar 与主力判定缓存，git 忽略）
└── README.md           # 本档案
```

## 本地预览

```bash
npm run dev    # http://localhost:7100/
```

## 数据重建

安装依赖（duckdb、pandas、numpy）后执行：

```bash
python build_real_data.py   # 约 1-2 分钟，重新生成 cache/ 与 real_data.js
```

## 备注：下载 Qlib 进行机器学习

本课程进入实操阶段后，统一使用 **微软开源量化平台 Qlib** 完成特征工程、模型训练与回测：

```bash
# 1. 安装
pip install pyqlib

# 2. 准备数据（qlib 格式的中国区数据 dump）
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 3. 验证
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data'); print('qlib OK')"
```

- 官方仓库：https://github.com/microsoft/qlib
- 数据下载较慢时可用社区镜像或按官方文档从原始行情自行转换（本项目的 `D:\data` 1m/1d Parquet 即已具备转换原料）。
- Qlib 工作流（qrun + 配置 yaml）将承接本页面的「数据标签 → 特征工程 → 模型训练 → 回测」环节；DD 特征族（DD 值、分位数、峰度、峰部比值、两轴背离度）将以自定义 Alpha 形式注入 Qlib 的特征配置。
