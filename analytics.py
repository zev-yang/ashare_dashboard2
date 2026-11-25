"""纯计算函数（不依赖网页、不联网），便于单测。金额单位统一"亿元"。"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

MULT = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}


# ───────────── 区间涨跌 ─────────────
def period_mask(idx, ref, freq):
    """idx 中与 ref 同一周(W, 周一~周日)或同一月(M)且 <= ref 的位置。"""
    idx = pd.DatetimeIndex(idx)
    f = "W" if freq == "W" else "M"
    return (idx.to_period(f) == pd.Timestamp(ref).to_period(f)) & (idx <= pd.Timestamp(ref))


def period_returns(close: pd.Series, asof=None) -> dict:
    """当日 / 本周至今(WTD) / 本月至今(MTD)。基准 = 上一周期最后一个交易日收盘。"""
    s = close.dropna().sort_index()
    if asof is not None:
        s = s[s.index <= pd.Timestamp(asof)]
    out = {"day": np.nan, "wtd": np.nan, "mtd": np.nan, "close": np.nan}
    if len(s) < 2:
        return out
    last = s.iloc[-1]
    out["close"] = last
    out["day"] = last / s.iloc[-2] - 1
    for key, f in (("wtd", "W"), ("mtd", "M")):
        m = period_mask(s.index, s.index[-1], f)
        first = s.index[m][0]
        prior = s[s.index < first]
        base = prior.iloc[-1] if len(prior) else s.iloc[0]
        out[key] = last / base - 1
    return out


def returns_table(panel: pd.DataFrame, names: dict, asof=None) -> pd.DataFrame:
    rows = []
    for c in panel.columns:
        r = period_returns(panel[c], asof)
        rows.append({"代码": c, "名称": names.get(c, c), "收盘": r["close"],
                     "当日": r["day"], "本周": r["wtd"], "本月": r["mtd"]})
    return pd.DataFrame(rows)


def basket_returns(panel: pd.DataFrame, codes, asof=None) -> dict:
    """等权篮子：各成分区间涨幅取平均；breadth=当日上涨家数占比。"""
    cols = [c for c in codes if c in panel.columns]
    if not cols:
        return {"day": np.nan, "wtd": np.nan, "mtd": np.nan, "breadth": np.nan, "n": 0}
    t = returns_table(panel[cols], {}, asof)
    return {"day": t["当日"].mean(), "wtd": t["本周"].mean(), "mtd": t["本月"].mean(),
            "breadth": (t["当日"] > 0).mean(), "n": len(cols)}


def basket_curve(panel: pd.DataFrame, codes) -> pd.Series:
    """等权日度收益累乘，得到篮子净值曲线（起点=1）。"""
    cols = [c for c in codes if c in panel.columns]
    if not cols:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    r = panel[cols].pct_change().mean(axis=1).fillna(0)
    return (1 + r).cumprod()


# ───────────── 均线 ─────────────
def ma_frame(close: pd.Series) -> pd.DataFrame:
    s = close.dropna().sort_index()
    return pd.DataFrame({"close": s, "ma5": s.rolling(5).mean(),
                         "ma20": s.rolling(20).mean(), "ma60": s.rolling(60).mean()})


def ma_status(close: pd.Series, asof=None) -> dict:
    s = close.dropna().sort_index()
    if asof is not None:
        s = s[s.index <= pd.Timestamp(asof)]
    f = ma_frame(s)
    if len(f) < 60 or f["ma60"].iloc[-1] != f["ma60"].iloc[-1]:
        return {"state": "数据不足(<60日)", "score": 0, "why": "日线不足60根", **{k: np.nan for k in
                ("close", "ma5", "ma20", "ma60", "dist20", "slope20", "cross", "above60_20d")}}
    c, m5, m20, m60 = (f[k].iloc[-1] for k in ("close", "ma5", "ma20", "ma60"))
    if c > m20 > m60 and m5 > m20:
        state, score = "多头排列", 1
    elif c < m20 < m60 and m5 < m20:
        state, score = "空头排列", -1
    elif c > m60:
        state, score = "震荡偏强", 0
    else:
        state, score = "震荡偏弱", 0
    sign = np.sign(f["ma5"] - f["ma20"]).dropna()
    chg = sign[sign != sign.shift(1)].iloc[1:]
    if len(chg):
        d = chg.index[-1]
        n = len(sign[sign.index > d])
        cross = f"{'金叉' if chg.iloc[-1] > 0 else '死叉'}({n}日前)"
    else:
        cross = "无"
    why = {"多头排列": f"收盘{c:.2f} > MA20 {m20:.2f} > MA60 {m60:.2f}，且 MA5 {m5:.2f} > MA20",
           "空头排列": f"收盘{c:.2f} < MA20 {m20:.2f} < MA60 {m60:.2f}，且 MA5 {m5:.2f} < MA20",
           "震荡偏强": f"不满足多/空头排列；收盘{c:.2f} > MA60 {m60:.2f}",
           "震荡偏弱": f"不满足多/空头排列；收盘{c:.2f} ≤ MA60 {m60:.2f}"}[state]
    return {"state": state, "score": score, "why": why, "close": c, "ma5": m5, "ma20": m20, "ma60": m60,
            "dist20": c / m20 - 1, "slope20": f["ma20"].iloc[-1] / f["ma20"].iloc[-6] - 1,
            "cross": cross, "above60_20d": (f["close"] > f["ma60"]).tail(20).mean()}


# ───────────── ETF 资金 ─────────────
def etf_flow_wide(shares_df: pd.DataFrame, price_wide: pd.DataFrame):
    """净申购(亿元) = 份额变化 x 当日收盘价（用市价近似净值）。返回 (flow, aum)。"""
    sh = shares_df.pivot_table(index="date", columns="code", values="shares", aggfunc="last").sort_index()
    px = price_wide.reindex(columns=sh.columns).reindex(sh.index.union(price_wide.index)).ffill().reindex(sh.index)
    return sh.diff() * px / 1e8, sh * px / 1e8


def _sum_period(s: pd.Series, freq):
    return s[period_mask(s.index, s.index[-1], freq)].sum()


def _aum_before(aum: pd.Series, freq):
    m = period_mask(aum.index, aum.index[-1], freq)
    prior = aum[aum.index < aum.index[m][0]].dropna()
    return prior.iloc[-1] if len(prior) else np.nan


def etf_group_indicators(flow, aum, amount_wide, codes, asof=None):
    """五项指标：①净申购(日/周/月) ②规模归一(周/月/5日/20日) ③累计5/20日 ④成交额倍数 ⑤占四类合计占比在 app 层算。"""
    cols = [c for c in codes if c in flow.columns]
    if not cols:
        return None
    f = flow[cols].sum(axis=1, min_count=1).dropna()
    a = aum[cols].sum(axis=1, min_count=1)
    if asof is not None:
        f, a = f[f.index <= pd.Timestamp(asof)], a[a.index <= pd.Timestamp(asof)]
    if len(f) < 2:
        return None
    r = {"date": f.index[-1], "day": f.iloc[-1], "wtd": _sum_period(f, "W"), "mtd": _sum_period(f, "M"),
         "d5": f.tail(5).sum(), "d20": f.tail(20).sum(), "n_etf": len(cols)}
    for k, fr in (("wtd", "W"), ("mtd", "M")):
        base = _aum_before(a.dropna(), fr)
        r[f"norm_{k}"] = r[k] / base if base == base and base else np.nan
    a2 = a.dropna()
    r["norm_d5"] = r["d5"] / a2.iloc[-6] if len(a2) >= 6 else np.nan
    r["norm_d20"] = r["d20"] / a2.iloc[-21] if len(a2) >= 21 else np.nan
    r["amount_x"] = np.nan
    ac = [c for c in cols if c in amount_wide.columns]
    if ac:
        amt = amount_wide[ac].sum(axis=1, min_count=1).dropna()
        if asof is not None:
            amt = amt[amt.index <= pd.Timestamp(asof)]
        if len(amt) > 21 and amt.iloc[-21:-1].mean() > 0:
            r["amount_x"] = amt.iloc[-1] / amt.iloc[-21:-1].mean()
    return r


def etf_group_flow_series(flow, codes) -> pd.Series:
    cols = [c for c in codes if c in flow.columns]
    return flow[cols].sum(axis=1, min_count=1) if cols else pd.Series(dtype=float, index=pd.DatetimeIndex([]))


# ───────────── 融资融券 ─────────────
def margin_group(margin_df: pd.DataFrame, cons_codes, amount_wide: pd.DataFrame) -> pd.DataFrame:
    """成分股融资买入占比 = Σ融资买入额 / Σ成交额（分子分母只取两边都有数据的个股）。"""
    m = margin_df[margin_df["code"].isin(cons_codes)]
    if m.empty:
        return pd.DataFrame()
    buy = m.pivot_table(index="date", columns="code", values="rz_buy", aggfunc="last").sort_index()
    bal = m.pivot_table(index="date", columns="code", values="rz_balance", aggfunc="last").sort_index()
    common = [c for c in buy.columns if c in amount_wide.columns]
    out = pd.DataFrame(index=buy.index)
    out["buy"] = buy.sum(axis=1, min_count=1) / 1e8
    out["bal"] = bal.sum(axis=1, min_count=1) / 1e8
    out["dbal"] = bal.diff().sum(axis=1, min_count=1) / 1e8
    if common:
        amt = amount_wide[common].reindex(buy.index)
        mask = amt.notna() & buy[common].notna()
        out["ratio"] = buy[common].where(mask).sum(axis=1) / amt.where(mask).sum(axis=1).replace(0, np.nan)
    else:
        out["ratio"] = np.nan
    out["coverage"] = len(common) / max(len(cons_codes), 1)
    return out


def margin_summary(g: pd.DataFrame, asof=None) -> dict:
    if g is None or g.empty:
        return {}
    if asof is not None:
        g = g[g.index <= pd.Timestamp(asof)]
    r = g["ratio"].dropna()
    out = {"date": g.index[-1], "bal": g["bal"].iloc[-1], "buy": g["buy"].iloc[-1],
           "dbal_day": g["dbal"].iloc[-1], "dbal_wtd": _sum_period(g["dbal"].dropna(), "W"),
           "dbal_mtd": _sum_period(g["dbal"].dropna(), "M"), "coverage": g["coverage"].iloc[-1]}
    if len(r):
        out.update(ratio=r.iloc[-1], ratio5=r.tail(5).mean(), ratio20=r.tail(20).mean(),
                   pct=(r.tail(120) <= r.iloc[-1]).mean())
    return out


# ───────────── 期指 ─────────────
def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def contract_expiry(contract: str) -> date:
    return third_friday(2000 + int(contract[2:4]), int(contract[4:6]))


def active_contracts(variety: str, d) -> list:
    """给定日期的在市合约：当月、下月、随后两个季月。"""
    d = pd.Timestamp(d).date()
    months, y, m = [], d.year, d.month
    if third_friday(y, m) < d:
        m += 1
    cur = (y + (m - 1) // 12, (m - 1) % 12 + 1)
    months.append(cur)
    nxt = (cur[0] + cur[1] // 12, cur[1] % 12 + 1)
    months.append(nxt)
    yy, mm = nxt
    while len(months) < 4:
        mm += 1
        if mm > 12:
            yy, mm = yy + 1, 1
        if mm % 3 == 0:
            months.append((yy, mm))
    return [f"{variety}{y % 100:02d}{m:02d}" for y, m in months]


def rank_totals(rank: pd.DataFrame, variety: str) -> pd.DataFrame:
    """前20会员多/空持仓（该品种所有合约合计）及净持仓(多-空，负=净空)。"""
    r = rank[rank["variety"] == variety]
    g = r.groupby("date").agg(long=("long_oi", "sum"), short=("short_oi", "sum")).sort_index()
    g["net"] = g["long"] - g["short"]
    return g


def member_table(rank: pd.DataFrame, d, variety: str) -> pd.DataFrame:
    r = rank[(rank["date"] == pd.Timestamp(d)) & (rank["variety"] == variety)]
    L = r.groupby("long_member").agg(多单=("long_oi", "sum"), 多单增减=("long_chg", "sum"))
    S = r.groupby("short_member").agg(空单=("short_oi", "sum"), 空单增减=("short_chg", "sum"))
    t = L.join(S, how="outer").fillna(0)
    t["净持仓"] = t["多单"] - t["空单"]
    t["净持仓增减"] = t["多单增减"] - t["空单增减"]
    t.index.name = "席位"
    return t.sort_values("净持仓")


def basis_series(daily: pd.DataFrame, index_close: pd.Series, variety: str) -> pd.DataFrame:
    """主力合约(持仓量最大) 的基差与年化基差；临近交割(<3天)不算年化。负=贴水。"""
    d = daily[daily["variety"] == variety]
    rows = []
    for dt, g in d.groupby("date"):
        g = g[g["contract"].map(lambda c: contract_expiry(c) >= dt.date())]
        if g.empty or dt not in index_close.index:
            continue
        top = g.loc[g["oi"].idxmax()]
        S, F = index_close.loc[dt], top["close"]
        days = (contract_expiry(top["contract"]) - dt.date()).days
        bad = abs(F / S - 1) > 0.2                   # 期货与指数点位相差20%以上必有口径问题，不输出年化
        rows.append({"date": dt, "contract": top["contract"], "F": F, "S": S, "oi": top["oi"],
                     "basis": F - S, "scale_bad": bool(bad),
                     "ann": (F / S - 1) * 365 / days if (days >= 3 and not bad) else np.nan})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def oi_price_quadrant(ret: float, doi: float) -> str:
    if np.isnan(ret) or np.isnan(doi):
        return "—"
    if ret >= 0:
        return "价涨增仓(多头主动)" if doi > 0 else "价涨减仓(空头回补)"
    return "价跌增仓(空头主动)" if doi > 0 else "价跌减仓(多头平仓)"


def classify_hedge(short_add: float, spot_add: float, cfg: dict) -> str:
    """用户规则：空单增量/现货增量。≈1:1 → 偏对冲；现货不增而空单大增 → 偏投机。"""
    if short_add != short_add or spot_add != spot_add:
        return "数据不足"
    if short_add < cfg["min_short_add"]:
        return "空单无显著增加"
    if spot_add <= 0:
        return "偏投机(空单增、现货未增)"
    ratio = short_add / spot_add
    if cfg["ratio_lo"] <= ratio <= cfg["ratio_hi"]:
        return "偏对冲(空单与现货同步)"
    if ratio > cfg["ratio_spec"]:
        return "偏投机(空单增幅远超现货)"
    return "不确定"


def hedge_frame(rank_tot: pd.DataFrame, index_close: pd.Series, mult: int, etf_flow: pd.Series,
                margin_dbal: pd.Series, cfg: dict, use_etf=True, use_margin=True) -> pd.DataFrame:
    N = cfg["window"]
    px = index_close.reindex(rank_tot.index).ffill()
    f = pd.DataFrame(index=rank_tot.index)
    f["short_add"] = (rank_tot["short"] - rank_tot["short"].shift(N)) * mult * px / 1e8
    f["long_add"] = (rank_tot["long"] - rank_tot["long"].shift(N)) * mult * px / 1e8
    spot = pd.Series(0.0, index=f.index)
    have = False
    if use_etf and etf_flow is not None and len(etf_flow):
        e = etf_flow.reindex(f.index)
        f["spot_etf"] = e.fillna(0).rolling(N).sum().where(e.rolling(N, min_periods=1).count() > 0)
        spot = spot + f["spot_etf"].fillna(0)
        have = True
    if use_margin and margin_dbal is not None and len(margin_dbal):
        m = margin_dbal.reindex(f.index)
        f["spot_margin"] = m.fillna(0).rolling(N).sum().where(m.rolling(N, min_periods=1).count() > 0)
        spot = spot + f["spot_margin"].fillna(0)
        have = True
    f["spot_add"] = spot if have else np.nan
    f["ratio"] = f["short_add"] / f["spot_add"].where(f["spot_add"] > 0)
    f["label"] = [classify_hedge(a, b, cfg) for a, b in zip(f["short_add"], f["spot_add"])]
    return f


def label_stats(frame: pd.DataFrame, index_close: pd.Series, horizon: int = 5) -> pd.DataFrame:
    """各标签之后 horizon 日指数涨跌统计；并按用户假设(偏对冲→涨, 偏投机→跌)算方向命中率。"""
    px = index_close.reindex(frame.index).ffill()
    fwd = px.shift(-horizon) / px - 1
    d = pd.DataFrame({"label": frame["label"], "fwd": fwd}).dropna()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for lab, g in d.groupby("label"):
        hit = np.nan
        if lab.startswith("偏对冲"):
            hit = (g["fwd"] > 0).mean()
        elif lab.startswith("偏投机"):
            hit = (g["fwd"] < 0).mean()
        rows.append({"标签": lab, "样本数": len(g), f"后{horizon}日均涨跌": g["fwd"].mean(),
                     "上涨占比": (g["fwd"] > 0).mean(), "按假设方向命中率": hit})
    return pd.DataFrame(rows).set_index("标签")
# ───────────── 平均股价（除权修正链式回溯）─────────────
def _chain_from_steps(raw_avg: pd.Series, step: pd.Series) -> pd.Series:
    """以最后一天的真实算术均价为锚，按每日复权后涨跌幅链式回溯，消除除权跳空与停牌/新股噪声。"""
    s = pd.DataFrame({"raw": raw_avg, "step": step}).dropna().sort_index()
    if s.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    st_ = s["step"].where((s["step"] > 0) & np.isfinite(s["step"]), 1.0)
    cum = st_.cumprod()
    return s["raw"].iloc[-1] * cum / cum.iloc[-1]


def market_avg_frame(mkt_df: pd.DataFrame, asof=None) -> pd.DataFrame:
    """由 market_avg 数据集生成：adj_avg(除权修正算术均价)、eq_avg(等权复权均价)、raw_avg(原始均价)、n。"""
    if mkt_df is None or mkt_df.empty:
        return pd.DataFrame()
    d = mkt_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    if asof is not None:
        d = d[d.index <= pd.Timestamp(asof)]
    if d.empty:
        return pd.DataFrame()
    step_pw = d["sum_close"] / d["sum_pre"].replace(0, np.nan)
    step_eq = 1.0 + d["eq_ret"]
    out = pd.DataFrame(index=d.index)
    out["n"] = d["n"]
    out["raw_avg"] = d["raw_avg"]
    out["adj_avg"] = _chain_from_steps(d["raw_avg"], step_pw)
    out["eq_avg"] = _chain_from_steps(d["raw_avg"], step_eq)
    return out


def basket_avg_frame(sector_px: pd.DataFrame, codes, asof=None) -> pd.DataFrame:
    """用 (date, code, close, pre_close) 计算板块篮子的除权修正均价、等权复权均价、原始均价。"""
    if sector_px is None or sector_px.empty:
        return pd.DataFrame()
    m = sector_px[sector_px["code"].isin(list(codes))].copy()
    if m.empty:
        return pd.DataFrame()
    m["date"] = pd.to_datetime(m["date"])
    if asof is not None:
        m = m[m["date"] <= pd.Timestamp(asof)]
    m = m[(m["close"] > 0) & (m["pre_close"] > 0)]
    if m.empty:
        return pd.DataFrame()
    m["ret"] = m["close"] / m["pre_close"] - 1.0
    g = m.groupby("date").agg(
        n=("close", "count"),
        raw_avg=("close", "mean"),
        sum_close=("close", "sum"),
        sum_pre=("pre_close", "sum"),
        eq_ret=("ret", "mean"),
    ).sort_index()
    out = pd.DataFrame(index=g.index)
    out["n"] = g["n"]
    out["raw_avg"] = g["raw_avg"]
    out["adj_avg"] = _chain_from_steps(g["raw_avg"], g["sum_close"] / g["sum_pre"].replace(0, np.nan))
    out["eq_avg"] = _chain_from_steps(g["raw_avg"], 1.0 + g["eq_ret"])
    return out
# ───────────── 维度 1：跨境资金与宏观利率（相对强弱领先/验证模块）─────────────
def _align_proxy_series(proxy_df: pd.DataFrame, symbol: str, ashare_dates: pd.DatetimeIndex, shift_us: bool) -> pd.Series:
    """将外资代理或利率序列对齐到 A 股交易日（不向过去填充，避免未来函数）。
    shift_us=True 用于美股/美债/NY17:00收盘外汇：将日期+1自然日后前向填充，确保 T 日只使用 T 日15:00前已知的隔夜收盘。"""
    if proxy_df is None or proxy_df.empty:
        return pd.Series(dtype=float, index=ashare_dates)
    sub = proxy_df[proxy_df["symbol"] == symbol].copy()
    if sub.empty:
        return pd.Series(dtype=float, index=ashare_dates)
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
    sub = sub.dropna(subset=["date", "close"])
    sub = sub[sub["close"] > 0].sort_values("date").drop_duplicates("date", keep="last")
    if sub.empty:
        return pd.Series(dtype=float, index=ashare_dates)
    idx = pd.DatetimeIndex(sub["date"])
    if shift_us:
        idx = idx + pd.Timedelta(days=1)
    s = pd.Series(sub["close"].to_numpy(float), index=idx)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(ashare_dates)).sort_index().ffill().reindex(ashare_dates)


def cross_border_analysis(proxy_df: pd.DataFrame, bench_close: pd.Series, def_curve: pd.Series, asof=None) -> dict:
    """计算跨境资金与利率 5 指标信号、投票结论、拥挤度分位及「防御/沪深300」相对强弱序列。"""
    b = bench_close.dropna().sort_index()
    if asof is not None:
        b = b[b.index <= pd.Timestamp(asof)]
    if b.empty or proxy_df is None or proxy_df.empty:
        return {}
    ashare_dates = b.index

    # A50 优先取交易时段与A股一致的 A50_IDX(XIN9)，否则取 A50(CHA50CFD)
    s_a50_idx = _align_proxy_series(proxy_df, "A50_IDX", ashare_dates, shift_us=False)
    s_a50_cfd = _align_proxy_series(proxy_df, "A50", ashare_dates, shift_us=False)
    if s_a50_idx.notna().sum() >= 60 and (s_a50_cfd.isna().all() or s_a50_idx.dropna().index.max() >= s_a50_cfd.dropna().index.max()):
        s_a50, a50_src = s_a50_idx, "A50_IDX(BJT)"
    else:
        s_a50, a50_src = s_a50_cfd, "A50(CHA50CFD)"

    s_kweb = _align_proxy_series(proxy_df, "KWEB", ashare_dates, shift_us=True)
    s_hsahp = _align_proxy_series(proxy_df, "HSAHP", ashare_dates, shift_us=False)
    s_us10y = _align_proxy_series(proxy_df, "US10Y", ashare_dates, shift_us=True)
    s_cn10y = _align_proxy_series(proxy_df, "CN10Y", ashare_dates, shift_us=False)

    s_usdcnh = _align_proxy_series(proxy_df, "USDCNH", ashare_dates, shift_us=True)
    if s_usdcnh.notna().sum() >= 20:
        s_fx, fx_src = s_usdcnh, "USDCNH(NY+1日)"
    else:
        s_fx, fx_src = _align_proxy_series(proxy_df, "USDCNY_MID", ashare_dates, shift_us=False), "USDCNY_MID(BJT)"

    s_kweb_a50 = (s_kweb / s_a50).where((s_kweb > 0) & (s_a50 > 0))
    specs = [
        ("kweb_a50", f"离岸成长风险偏好 (KWEB/{a50_src})", s_kweb_a50, -1, False, "美股收盘+1日 vs BJT"),
        ("hsahp", "AH溢价情绪 (HSAHP)", s_hsahp, +1, False, "港股日频(同日)"),
        ("us10y", "10Y美债收益率 (%)", s_us10y, +1, True, "美债收盘+1日"),
        ("fx", f"汇率锚 ({fx_src})", s_fx, +1, False, "离岸NY+1日 / 中间价同日"),
        ("cn10y", "10Y中债收益率 (%)", s_cn10y, -1, True, "中债日频(同日)"),
    ]

    votes = pd.DataFrame(index=ashare_dates)
    impulses = pd.DataFrame(index=ashare_dates)
    series_df = pd.DataFrame(index=ashare_dates)
    meta = []

    for key, label, s, def_sign, is_rate, tz_note in specs:
        series_df[key] = s
        ma5 = s.rolling(5, min_periods=5).mean()
        ma20 = s.rolling(20, min_periods=20).mean()
        if is_rate:
            slope20 = ma20 - ma20.shift(5)
            chg5 = ma5 - ma5.shift(5)
            eps20, eps5 = 0.02, 0.01
        else:
            slope20 = ma20 / ma20.shift(5) - 1.0
            chg5 = ma5 / ma5.shift(5) - 1.0
            eps20, eps5 = 0.002, 0.002

        v = pd.Series(np.nan, index=ashare_dates, dtype=float)
        valid = slope20.notna() & ma5.notna() & ma20.notna()
        v[valid] = 0.0
        v[valid & (def_sign * slope20 > eps20) & (def_sign * (ma5 - ma20) > 0)] = 1.0
        v[valid & (def_sign * slope20 < -eps20) & (def_sign * (ma5 - ma20) < 0)] = -1.0
        votes[key] = v

        imp = pd.Series(np.nan, index=ashare_dates, dtype=float)
        iv = chg5.notna()
        imp[iv] = 0.0
        imp[iv & (def_sign * chg5 > eps5)] = 1.0
        imp[iv & (def_sign * chg5 < -eps5)] = -1.0
        impulses[key] = imp

        meta.append({
            "key": key, "指标": label, "def_sign": def_sign, "is_rate": is_rate, "时区对齐": tz_note,
            "最新值": float(s.iloc[-1]) if s.notna().any() else np.nan,
            "MA5": float(ma5.iloc[-1]) if ma5.notna().any() else np.nan,
            "MA20": float(ma20.iloc[-1]) if ma20.notna().any() else np.nan,
            "MA20近5日变化": float(slope20.iloc[-1]) if slope20.notna().any() else np.nan,
            "短期5日动向": imp.iloc[-1],
            "趋势投票": v.iloc[-1],
        })

    n_avail = votes.notna().sum(axis=1)
    n_cont = (votes == 1.0).sum(axis=1)
    n_conv = (votes == -1.0).sum(axis=1)
    quorum = np.maximum(2, np.ceil(0.6 * n_avail)).astype(int)

    verdict = pd.Series("中性观望", index=ashare_dates)
    verdict[n_avail < 3] = "数据不足"
    verdict[(n_avail >= 3) & (n_cont >= quorum) & (n_cont > n_conv)] = "背离延续"
    verdict[(n_avail >= 3) & (n_conv >= quorum) & (n_conv > n_cont)] = "背离收敛"

    net_score = (n_cont - n_conv) / n_avail.replace(0, np.nan)
    imp_score = impulses.mean(axis=1)

    # 防御综合 / 沪深300 相对强弱及 120 日拥挤度分位
    dc = def_curve.dropna().sort_index() if def_curve is not None else pd.Series(dtype=float)
    if asof is not None and len(dc):
        dc = dc[dc.index <= pd.Timestamp(asof)]
    common = b.index.intersection(dc.index)
    rel_ratio = pd.Series(np.nan, index=ashare_dates, dtype=float)
    rel_pct120 = pd.Series(np.nan, index=ashare_dates, dtype=float)
    if len(common) >= 2:
        raw_rel = (dc.loc[common] / dc.loc[common].iloc[0]) / (b.loc[common] / b.loc[common].iloc[0])
        rel_ratio.loc[common] = raw_rel
        rel_pct120.loc[common] = raw_rel.rolling(120, min_periods=40).rank(pct=True)

    # 连续背离收敛天数
    is_conv = (verdict == "背离收敛").astype(int)
    conv_streak = is_conv.groupby((is_conv != is_conv.shift(1)).cumsum()).cumsum() * is_conv

    panel = pd.DataFrame({
        "n_avail": n_avail, "n_cont": n_cont, "n_conv": n_conv,
        "net_score": net_score, "net_score_ma5": net_score.rolling(5, min_periods=1).mean(),
        "imp_score": imp_score, "verdict": verdict, "conv_streak": conv_streak,
        "rel_ratio": rel_ratio, "rel_pct120": rel_pct120,
        "spread_cn_us": s_cn10y - s_us10y,
    }, index=ashare_dates)

    return {
        "panel": panel, "series": series_df, "votes": votes,
        "meta": pd.DataFrame(meta), "bench": b, "def_curve": dc,
    }


def cross_border_validation(cb_res: dict, horizons=(5, 10)) -> pd.DataFrame:
    """用历史数据检验「背离延续 / 中性观望 / 背离收敛」之后相对强弱(防御综合-沪深300)的真实超额表现，并给出全样本基线对照。"""
    if not cb_res:
        return pd.DataFrame()
    panel, b, dc = cb_res["panel"], cb_res["bench"], cb_res["def_curve"]
    common = panel.index.intersection(b.index).intersection(dc.index)
    if len(common) < 30:
        return pd.DataFrame()
    v = panel.loc[common, "verdict"]
    d_sub = dc.loc[common]
    b_sub = b.loc[common]

    data = pd.DataFrame({"verdict": v}, index=common)
    for h in horizons:
        data[f"ex_{h}"] = (d_sub.shift(-h) / d_sub - 1.0) - (b_sub.shift(-h) / b_sub - 1.0)

    valid_all = data[data["verdict"] != "数据不足"].dropna(subset=[f"ex_{horizons[0]}"])
    if valid_all.empty:
        return pd.DataFrame()

    rows = []
    tot_n = len(valid_all)
    for lab in ("背离延续", "中性观望", "背离收敛", "全样本基准"):
        g = valid_all if lab == "全样本基准" else valid_all[valid_all["verdict"] == lab]
        if g.empty:
            continue
        row = {"信号": lab, "样本数": len(g), "占比": len(g) / tot_n}
        for h in horizons:
            ex = g[f"ex_{h}"].dropna()
            row[f"后{h}日相对超额均值"] = ex.mean() if len(ex) else np.nan
        ex5 = g[f"ex_{horizons[0]}"].dropna()
        row[f"后{horizons[0]}日防御跑赢占比"] = (ex5 > 0).mean() if len(ex5) else np.nan
        if lab == "背离延续":
            row[f"按方向命中率({horizons[0]}日)"] = (ex5 > 0).mean() if len(ex5) else np.nan
        elif lab == "背离收敛":
            row[f"按方向命中率({horizons[0]}日)"] = (ex5 < 0).mean() if len(ex5) else np.nan
        else:
            row[f"按方向命中率({horizons[0]}日)"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
