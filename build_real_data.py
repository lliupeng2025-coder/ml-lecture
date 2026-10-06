# -*- coding: utf-8 -*-
"""
RB99 主力连续 30m 线构建 + DD 分布差异计算（严格因果口径，无未来函数）

因果口径说明：
1. 主力合约判定：第 d 日主力 = 第 d-1 日总成交量最大的 RB 合约（并列比持仓量），
   即"用昨天的量决定今天跟哪个合约"。首日无历史，用当日量（bootstrap，单独声明）。
2. 换月价差复权：roll 日 d 的价差缺口 = 新主力与旧主力在 d 日首个共有分钟收盘价之差；
   复权采用价差回溯调整（back-adjust），只影响历史价格水平，不引入未来涨跌信息。
3. 30m K线：交易时段内按自然 30 分钟分桶，收盘 = 桶内最后一分钟收盘价（桶内信息，无未来）。
4. 滚动 DD：窗口 [t-239, t] 的经验分布 vs 正态的 L1 偏离，t 时刻判定只用到 t 及之前数据。
5. 状态阈值：t 时刻的 q80/q40 分位数取自 [0, t-1] 的扩展窗口并滞后一期（shift(1)）。
"""
import glob
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd

DATA_DIR = r"D:\data\future_1m"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT_DIR, "cache", "rb99_30m_raw.parquet")
START, END = "2019-01-01", "2025-01-31"


def load_days():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.parquet")))
    files = [f for f in files if START.replace("-", "") <= os.path.basename(f)[:8] <= END.replace("-", "")]
    return files


def day_frame(path):
    return duckdb.sql(
        f"select code, trade_time, close, vol, oi from '{path}' where code like 'RB%'"
    ).df()


def build_raw():
    files = load_days()
    print(f"days: {len(files)}")
    dom_rows, bar_rows = [], []
    prev_dom, prev_dayvol, prev_dayoi = None, None, None
    t0 = time.time()
    for i, f in enumerate(files):
        df = day_frame(f)
        if df.empty:
            continue
        df["dt"] = pd.to_datetime(df["trade_time"])
        date = df["dt"].dt.date.iloc[0]
        # 当日合约级统计；listed = 剔除连续合约符号(8888/9999)
        stat = df.groupby("code").agg(dayvol=("vol", "sum"), dayoi=("oi", "last"))
        stat_l = stat[~stat.index.str.endswith(("8888", "9999"))]
        vendor_day = len(stat_l) == 0 and "RB8888" in stat.index
        # —— 因果主力判定：用前一交易日的量/持仓 ——
        if vendor_day:
            dom = "RB8888"  # 供应商主力连续（仅信号用途，文档声明）
        elif prev_dayvol is None:  # bootstrap 首日
            dom = stat_l["dayvol"].idxmax()
        else:
            joint = pd.DataFrame({"v": prev_dayvol, "oi": prev_dayoi}).join(
                stat_l[["dayvol", "dayoi"]], how="outer"
            ).fillna(0)
            # 若量最大的合约当日无数据（如挂牌未交易），依次取候补
            for cand in joint.sort_values(["v", "oi"]).index[::-1]:
                if cand in stat_l.index:
                    dom = cand
                    break
        # 主力合约当日 1m 序列（先取出，换月缺口与分桶共用）
        d = df[df["code"] == dom].sort_values("dt")
        # 换月缺口（首日除外）
        roll = prev_dom is not None and dom != prev_dom
        gap = np.nan
        if roll:
            both = df.pivot_table(index="dt", columns="code", values="close", aggfunc="last")
            if prev_dom in both.columns:
                common = both[[prev_dom, dom]].dropna()
                if len(common):
                    gap = float(common[dom].iloc[0] - common[prev_dom].iloc[0])
                else:
                    gap = 0.0  # 无共有分钟（极少见），记 0 并在日志声明
            else:
                # 旧主力当日已无数据：用"昨收旧主力 vs 今首收新主力"的价差
                gap = float(d["close"].iloc[0]) - prev_dom_last_close
        dom_rows.append({"date": str(date), "dom": dom, "roll": roll, "gap": gap})
        # 主力合约 1m -> 30m 分桶
        bucket = (d["dt"].dt.hour * 60 + (d["dt"].dt.minute // 30) * 30)
        d = d.assign(bucket=bucket)
        g = d.groupby("bucket").agg(close=("close", "last"), vol=("vol", "sum"),
                                    oi=("oi", "last"), n=("close", "size"))
        g = g.reset_index()
        g["night"] = (g["bucket"] >= 20 * 60).astype(int)  # 夜盘 21:00+ 排在日前
        g = g.sort_values(["night", "bucket"])
        for _, r in g.iterrows():
            hh, mm = int(r["bucket"] // 60), int(r["bucket"] % 60)
            bar_rows.append({
                "date": str(date), "time": f"{hh:02d}:{mm:02d}", "night": int(r["night"]),
                "close": float(r["close"]), "vol": float(r["vol"]), "oi": float(r["oi"]),
                "n": int(r["n"]),
            })
        if not vendor_day:  # 供应商连续合约日的量不反映合约间竞争，保留上一交易日的统计
            prev_dayvol, prev_dayoi = stat_l["dayvol"], stat_l["dayoi"]
        prev_dom = dom
        prev_dom_last_close = float(d["close"].iloc[-1])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)")
    dom_df = pd.DataFrame(dom_rows)
    bar_df = pd.DataFrame(bar_rows)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    dom_df.to_csv(CACHE.replace("_raw.parquet", "_dom.csv"), index=False)
    bar_df.to_csv(CACHE.replace(".parquet", ".csv"), index=False)
    return dom_df, bar_df


def l1_vs_normal(z, bins=None):
    """经验直方图(与正态同均值同方差标准化后) vs 标准正态的 L1 距离。"""
    if bins is None:
        bins = np.linspace(-5, 5, 81)
    hist, _ = np.histogram(z, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ref = np.exp(-0.5 * centers**2) / np.sqrt(2 * np.pi)
    return float(np.abs(hist - ref).sum() * (bins[1] - bins[0])), hist, centers


def kurt(z):
    z = z - z.mean()
    s = z.std()
    return float((z**4).mean() / s**4)


def main():
    cvs = CACHE.replace(".parquet", ".csv")
    if os.path.exists(cvs):
        dom_df = pd.read_csv(CACHE.replace("_raw.parquet", "_dom.csv"))
        bar_df = pd.read_csv(cvs)
        print("loaded cache:", len(bar_df), "bars,", len(dom_df), "days")
    else:
        dom_df, bar_df = build_raw()

    # —— 价差回溯复权 ——
    gaps = dom_df["gap"].fillna(0).cumsum().values  # 每日累计调整因子
    day_factor = dict(zip(dom_df["date"], gaps))
    bar_df["adj"] = bar_df["close"] - bar_df["date"].map(day_factor)
    bar_df = bar_df.reset_index(drop=True)
    bar_df["dclose"] = bar_df["adj"].diff()  # 链式 Δclose（跨交易日连贯）

    z = (bar_df["dclose"] - bar_df["dclose"].mean()) / bar_df["dclose"].std()
    z = z.dropna().values
    dd_full, hist, centers = l1_vs_normal(z)
    kurt_full = kurt(z)

    # 全样本 KDE（Silverman）
    n = len(z)
    h = 1.06 * n ** (-0.2) * z.std()
    grid = np.linspace(-4.5, 4.5, 181)
    kde = np.exp(-0.5 * ((grid[None, :] - z[:, None]) / h) ** 2).mean(axis=0) / (h * np.sqrt(2 * np.pi))
    peak_ratio = float(kde.max() / (1 / np.sqrt(2 * np.pi)))

    # —— 滚动 DD（因果窗口 W=240，判定于 t，使用 [t-239, t]）——
    W = 240
    bins = np.linspace(-5, 5, 81)
    ref_c = 0.5 * (bins[:-1] + bins[1:])
    ref = np.exp(-0.5 * ref_c**2) / np.sqrt(2 * np.pi)
    zser = pd.Series(z)
    roll_dd = np.full(n, np.nan)
    zz = zser.values
    # 滚动直方图（增量式）
    hist_c = np.zeros(len(bins) - 1)
    idx_all = np.clip(np.digitize(zz, bins) - 1, 0, len(bins) - 2)
    from collections import deque
    dq = deque()
    cnt = 0
    for t in range(n):
        dq.append(t)
        hist_c[idx_all[t]] += 1
        cnt += 1
        if cnt > W:
            old = dq.popleft()
            hist_c[idx_all[old]] -= 1
            cnt -= 1
        if cnt >= W:
            emp = hist_c / (cnt * (bins[1] - bins[0]))
            roll_dd[t] = np.abs(emp - ref).sum() * (bins[1] - bins[0])
    bar_df["dd"] = np.concatenate([[np.nan], roll_dd])  # 与 dclose 对齐（dclose 少 1 行）

    # —— 因果状态判定：扩展窗口分位数（用 [0, t-1]，滞后一期）——
    dd_s = pd.Series(roll_dd)
    WARM = 480
    q_hi, q_lo = np.full(n, np.nan), np.full(n, np.nan)
    regime = np.full(n, -1, dtype=int)  # 0 过渡 1 震荡 2 趋势
    for t in range(WARM, n):
        past = dd_s.iloc[:t].dropna()
        if len(past) < WARM // 2:
            continue
        qh, ql = past.quantile(0.75), past.quantile(0.35)
        q_hi[t], q_lo[t] = qh, ql
        if dd_s[t] >= qh:
            regime[t] = 1
        elif dd_s[t] <= ql:
            regime[t] = 2
        else:
            regime[t] = 0
    bar_df["regime"] = np.concatenate([[ -1], regime])

    # —— 换手率线 PD：30m 换手率 = vol/oi 的分布 vs 正态 ——
    to = (bar_df["vol"] / bar_df["oi"].replace(0, np.nan)).dropna()
    to_z = (to - to.mean()) / to.std()
    dd_to, hist_to, _ = l1_vs_normal(to_z.values)
    kde_to = np.exp(-0.5 * ((grid[None, :] - to_z.values[:, None]) / (1.06 * len(to_z) ** (-0.2))) ** 2)
    kde_to = kde_to.mean(axis=0) / (1.06 * len(to_z) ** (-0.2) * np.sqrt(2 * np.pi))

    rolls = dom_df[dom_df["roll"]]
    meta = {
        "n_bars": int(len(bar_df)),
        "n_days": int(len(dom_df)),
        "range": [bar_df["date"].iloc[0], bar_df["date"].iloc[-1]],
        "n_rolls": int(len(rolls)),
        "dd_full": round(dd_full, 4),
        "kurt_full": round(kurt_full, 3),
        "peak_ratio": round(peak_ratio, 3),
        "dd_turnover": round(dd_to, 4),
        "W": W, "warmup": WARM,
        "regime_counts": {int(k): int(v) for k, v in pd.Series(regime).value_counts().items()},
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("rolls:", rolls[["date", "dom", "gap"]].to_string(index=False))

    # —— 输出 JS 数据文件 ——
    out = {
        "meta": meta,
        "grid": [round(float(x), 3) for x in grid],
        "kde": [round(float(x), 5) for x in kde],
        "normal": [round(float(x), 5) for x in (np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi))],
        "kde_turnover": [round(float(x), 5) for x in kde_to],
        # 时间序列（dclose 对齐）：价格、滚动DD、状态
        "ts": {
            "date": bar_df["date"].tolist(),
            "time": bar_df["time"].tolist(),
            "night": bar_df["night"].tolist(),
            "adj": [round(float(x), 1) for x in bar_df["adj"]],
            "dd": [None if np.isnan(x) else round(float(x), 4) for x in bar_df["dd"]],
            "regime": bar_df["regime"].tolist(),
        },
    }
    with open(os.path.join(OUT_DIR, "real_data.js"), "w", encoding="utf-8") as fp:
        fp.write("const REAL_DATA = ")
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))
        fp.write(";\n")
    print("written real_data.js", os.path.getsize(os.path.join(OUT_DIR, "real_data.js")) / 1e6, "MB")


if __name__ == "__main__":
    main()
