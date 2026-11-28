# -*- coding: utf-8 -*-
"""升级首页「总览」标签页：整合 ①避险 ②ETF ③期指 ④融资 ⑤均线 ⑥跨境资金 六维信号，
并加入政体总控、去噪死区、上证50成分重合中性化、期指T+1对齐修复与历史回测检验。

前置条件：已运行过 apply_cross_border_patch.py（包含 ⑥ 跨境资金）。
用法（在项目根目录运行）：
    python apply_overview_6factor_patch.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ANALYTICS_OV6 = r'''

# ───────────── 总览：六因子整合信号与回测检验 ─────────────
def etf_group_norm_series(flow: pd.DataFrame, aum: pd.DataFrame, codes, asof=None) -> pd.DataFrame:
    """ETF 组日度序列：d1(当日净申购亿), d5(5日累计净申购亿), norm_d5(5日净申购/5日前规模)。"""
    if flow is None or aum is None or flow.empty or aum.empty:
        return pd.DataFrame()
    cols = [c for c in codes if c in flow.columns]
    if not cols:
        return pd.DataFrame()
    f = flow[cols].sum(axis=1, min_count=1).dropna().sort_index()
    a = aum[cols].sum(axis=1, min_count=1).sort_index()
    if asof is not None:
        ts = pd.Timestamp(asof)
        f, a = f[f.index <= ts], a[a.index <= ts]
    if f.empty:
        return pd.DataFrame()
    a_al = a.reindex(a.index.union(f.index)).sort_index().ffill().reindex(f.index)
    d5 = f.rolling(5, min_periods=5).sum()
    base = a_al.shift(5)
    norm_d5 = d5 / base.where(base > 0)
    return pd.DataFrame({"d1": f, "d5": d5, "norm_d5": norm_d5}, index=f.index)


def etf_factor_series(etf_df: pd.DataFrame) -> pd.DataFrame:
    """② ETF 资金信号：温和流入(+1) / 温和流出(-1) / 死区或极端护盘事件(0)。"""
    if etf_df is None or etf_df.empty:
        return pd.DataFrame()
    n5 = etf_df["norm_d5"]
    sig = pd.Series(np.nan, index=etf_df.index, dtype=float)
    valid = n5.notna()
    sig[valid] = 0.0
    sig[valid & (n5 >= 0.003) & (n5 < 0.03)] = 1.0
    sig[valid & (n5 <= -0.003) & (n5 > -0.03)] = -1.0
    extreme = valid & (n5.abs() >= 0.03)
    return pd.DataFrame({"sig": sig, "norm_d5": n5, "d5": etf_df["d5"], "extreme": extreme}, index=etf_df.index)


def ma_score_series(close: pd.Series, asof=None) -> pd.Series:
    """⑤ 均线信号向量化序列：与 ma_status() 完全等价（多头+1 / 空头-1 / 震荡0）。"""
    s = close.dropna().sort_index()
    if asof is not None:
        s = s[s.index <= pd.Timestamp(asof)]
    if s.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    f = ma_frame(s)
    c, m5, m20, m60 = f["close"], f["ma5"], f["ma20"], f["ma60"]
    sc = pd.Series(np.nan, index=f.index, dtype=float)
    valid = m60.notna()
    sc[valid] = 0.0
    sc[valid & (c > m20) & (m20 > m60) & (m5 > m20)] = 1.0
    sc[valid & (c < m20) & (m20 < m60) & (m5 < m20)] = -1.0
    return sc


def aligned_hedge_frame(rank_tot: pd.DataFrame, index_close: pd.Series, mult: int,
                        etf_flow: pd.Series, margin_dbal: pd.Series, cfg: dict, asof=None):
    """修复收盘后 T+1 错位：若期货持仓已到 T 日而 ETF/融资只到 T-1，将 hedge_frame 截齐至共同最新日，
    避免 5 日空单增量除以 4 日现货增量导致的偏投机偏差。返回 (hf, spot_lag)。"""
    ts = pd.Timestamp(asof) if asof is not None else None
    rt = rank_tot[rank_tot.index <= ts] if (ts is not None and len(rank_tot)) else rank_tot
    ic = index_close[index_close.index <= ts] if (ts is not None and len(index_close)) else index_close
    ef = etf_flow[etf_flow.index <= ts] if (ts is not None and etf_flow is not None and len(etf_flow)) else etf_flow
    md = margin_dbal[margin_dbal.index <= ts] if (ts is not None and margin_dbal is not None and len(margin_dbal)) else margin_dbal

    if rt is None or rt.empty:
        return pd.DataFrame(), False

    spot_maxes = []
    for s in (ef, md):
        if s is not None:
            s2 = s.dropna()
            if len(s2):
                spot_maxes.append(s2.index.max())
    spot_lag = False
    if spot_maxes:
        common_spot_max = min(spot_maxes)
        if rt.index.max() > common_spot_max:
            spot_lag = True
            rt = rt[rt.index <= common_spot_max]
    if rt.empty:
        return pd.DataFrame(), spot_lag
    return hedge_frame(rt, ic, mult, ef, md, cfg), spot_lag


def futures_factor_series(hf: pd.DataFrame, basis: pd.DataFrame, fdaily_var: pd.DataFrame,
                          idx_close: pd.Series, asof=None) -> pd.Series:
    """③ 期指综合日度信号：空单性质(偏对冲+1/偏投机-1) + 5日年化基差变化(±1.5%) + 5日价仓动向，取合力符号。"""
    s_px = idx_close.dropna().sort_index()
    if asof is not None:
        s_px = s_px[s_px.index <= pd.Timestamp(asof)]
    if s_px.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    idx = s_px.index
    subs = []

    if hf is not None and not hf.empty:
        sh = pd.Series(np.nan, index=hf.index, dtype=float)
        labs = hf["label"].astype(str)
        sh[labs != "数据不足"] = 0.0
        sh[labs.str.startswith("偏对冲")] = 1.0
        sh[labs.str.startswith("偏投机")] = -1.0
        subs.append(sh.reindex(idx).ffill(limit=1))

    if basis is not None and not basis.empty and "ann" in basis.columns:
        b = basis.sort_index()
        if asof is not None:
            b = b[b.index <= pd.Timestamp(asof)]
        ann = b["ann"].where(~b["scale_bad"].astype(bool)) if "scale_bad" in b.columns else b["ann"]
        dann = ann - ann.shift(5)
        sb = pd.Series(np.nan, index=b.index, dtype=float)
        vb = dann.notna()
        sb[vb] = 0.0
        sb[vb & (dann > 0.015)] = 1.0
        sb[vb & (dann < -0.015)] = -1.0
        subs.append(sb.reindex(idx))

    if fdaily_var is not None and not fdaily_var.empty:
        fd = fdaily_var.copy()
        fd["date"] = pd.to_datetime(fd["date"])
        if asof is not None:
            fd = fd[fd["date"] <= pd.Timestamp(asof)]
        oi_tot = fd.groupby("date")["oi"].sum().sort_index().reindex(idx)
        doi5 = oi_tot - oi_tot.shift(5)
        ret5 = s_px / s_px.shift(5) - 1.0
        so = pd.Series(np.nan, index=idx, dtype=float)
        vo = doi5.notna() & ret5.notna()
        so[vo] = 0.0
        so[vo & (ret5 > 0.005) & (doi5 > 0)] = 1.0
        so[vo & (ret5 < -0.005) & (doi5 > 0)] = -1.0
        subs.append(so)

    if not subs:
        return pd.Series(dtype=float, index=idx)
    mat = pd.concat(subs, axis=1)
    n_sub = mat.notna().sum(axis=1)
    tot = mat.sum(axis=1, min_count=1)
    out = pd.Series(np.nan, index=idx, dtype=float)
    ok = n_sub > 0
    out[ok] = np.sign(tot[ok])
    return out


def margin_factor_series(mg_df: pd.DataFrame, asof=None) -> pd.DataFrame:
    """④ 融资融券信号：融资买入占比5日 vs 20日(±0.2%死区) + 5日余额变化；>=40日有效分位时85%+触发过热封顶。"""
    if mg_df is None or mg_df.empty:
        return pd.DataFrame()
    g = mg_df.sort_index()
    if asof is not None:
        g = g[g.index <= pd.Timestamp(asof)]
    if g.empty:
        return pd.DataFrame()
    r = g["ratio"]
    r5 = r.rolling(5, min_periods=3).mean()
    r20 = r.rolling(20, min_periods=10).mean()
    db5 = g["dbal"].rolling(5, min_periods=3).sum()
    pct120 = r.rolling(120, min_periods=40).rank(pct=True)

    raw_sig = pd.Series(np.nan, index=g.index, dtype=float)
    valid = r5.notna() & r20.notna() & db5.notna()
    raw_sig[valid] = 0.0
    raw_sig[valid & (r5 > r20 + 0.002) & (db5 > 0)] = 1.0
    raw_sig[valid & (r5 < r20 - 0.002) & (db5 < 0)] = -1.0

    overheated = pct120 >= 0.85
    ice_cold = pct120 <= 0.15
    sig = raw_sig.where(~(overheated & (raw_sig > 0)), 0.0)
    n_valid_r = r.notna().cumsum()
    return pd.DataFrame({
        "sig": sig, "raw_sig": raw_sig, "overheated": overheated,
        "ice_cold": ice_cold, "pct120": pct120, "n_days": n_valid_r,
    }, index=g.index)


def sector_style_series(sec_close: pd.DataFrame, def_codes, bench_close: pd.Series, asof=None) -> pd.DataFrame:
    """① 避险板块风格信号（站在宽基指数偏多视角：防御跑赢=-1，防御跑输=+1）。"""
    b = bench_close.dropna().sort_index()
    if asof is not None:
        b = b[b.index <= pd.Timestamp(asof)]
    if b.empty or sec_close is None or sec_close.empty:
        return pd.DataFrame()
    sc = sec_close[sec_close.index <= b.index[-1]]
    cols = [c for c in def_codes if c in sc.columns]
    if not cols:
        return pd.DataFrame()
    dc = basket_curve(sc[cols].dropna(how="all"), cols)
    common = b.index.intersection(dc.index)
    if len(common) < 25:
        return pd.DataFrame()
    b_c, d_c = b.loc[common], dc.loc[common]
    ex5 = (d_c / d_c.shift(5) - 1.0) - (b_c / b_c.shift(5) - 1.0)
    ex20 = (d_c / d_c.shift(20) - 1.0) - (b_c / b_c.shift(20) - 1.0)
    rel = (d_c / d_c.iloc[0]) / (b_c / b_c.iloc[0])
    pct120 = rel.rolling(120, min_periods=40).rank(pct=True)

    sig = pd.Series(np.nan, index=common, dtype=float)
    valid = ex5.notna() & ex20.notna()
    sig[valid] = 0.0
    sig[valid & (ex5 > 0.008) & (ex20 > 0)] = -1.0
    sig[valid & (ex5 < -0.008) & (ex20 < 0)] = 1.0
    return pd.DataFrame({
        "sig": sig, "ex5": ex5, "ex20": ex20, "rel": rel,
        "pct120": pct120, "crowded": pct120 >= 0.85,
    }, index=common)


def classify_overview_verdict(avg_score: float, n_active: int, f5_ma: float,
                              warn_bull: bool = False, warn_bear: bool = False) -> str:
    if n_active < 3 or avg_score != avg_score:
        return "数据不足"
    if avg_score >= 0.45:
        return "🟠 谨慎偏多" if (warn_bull or (f5_ma == f5_ma and f5_ma < 0)) else "🔴 偏多共振"
    if avg_score >= 0.15:
        return "🟠 偏强观察"
    if avg_score <= -0.45:
        return "🟡 谨慎偏空" if (warn_bear or (f5_ma == f5_ma and f5_ma > 0)) else "🟢 偏空共振"
    if avg_score <= -0.15:
        return "🟡 偏弱观察"
    return "⚪ 震荡分化"


def overview_index_history(idx_close: pd.Series, sec_df: pd.DataFrame, etf_df: pd.DataFrame,
                           fut_sig: pd.Series, mar_df: pd.DataFrame, cb_res: dict,
                           is_sse50: bool = False, asof=None, lag_t1: bool = False) -> pd.DataFrame:
    """构建单个指数的 6 因子历史面板（lag_t1=True 用于严格历史回测，将 T+1 早晨披露的 ETF/融资滞后1日）。"""
    px = idx_close.dropna().sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    if px.empty:
        return pd.DataFrame()
    idx = px.index

    f5 = ma_score_series(px)
    f3 = fut_sig.reindex(idx) if (fut_sig is not None and len(fut_sig)) else pd.Series(np.nan, index=idx)

    if etf_df is not None and not etf_df.empty:
        e_sig = etf_df["sig"].shift(1) if lag_t1 else etf_df["sig"]
        e_ext = etf_df["extreme"].shift(1) if lag_t1 else etf_df["extreme"]
        f2 = e_sig.reindex(idx).ffill(limit=0 if lag_t1 else 1)
        ext2 = e_ext.reindex(idx).ffill(limit=0 if lag_t1 else 1).fillna(False)
    else:
        f2, ext2 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)

    if mar_df is not None and not mar_df.empty:
        m_sig = mar_df["sig"].shift(1) if lag_t1 else mar_df["sig"]
        m_ovh = mar_df["overheated"].shift(1) if lag_t1 else mar_df["overheated"]
        f4 = m_sig.reindex(idx).ffill(limit=0 if lag_t1 else 1)
        ovh4 = m_ovh.reindex(idx).ffill(limit=0 if lag_t1 else 1).fillna(False)
    else:
        f4, ovh4 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)

    if is_sse50:
        f1 = pd.Series(np.nan, index=idx)
        f6 = pd.Series(np.nan, index=idx)
        crd1 = pd.Series(False, index=idx)
        crd6 = pd.Series(False, index=idx)
    else:
        if sec_df is not None and not sec_df.empty:
            f1 = sec_df["sig"].reindex(idx)
            crd1 = sec_df["crowded"].reindex(idx).fillna(False)
        else:
            f1, crd1 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)
        if cb_res and "panel" in cb_res and not cb_res["panel"].empty:
            cp = cb_res["panel"]
            vmap = {"背离收敛": 1.0, "中性观望": 0.0, "背离延续": -1.0}
            f6 = cp["verdict"].map(vmap).reindex(idx)
            crd6 = (cp["rel_pct120"] >= 0.85).reindex(idx).fillna(False)
        else:
            f6, crd6 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)

    mat = pd.DataFrame({
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6,
    }, index=idx)
    core_mat = mat[["f2", "f3", "f4", "f5"]]
    n_active = mat.notna().sum(axis=1)
    score_sum = mat.sum(axis=1, min_count=1)
    avg_score = score_sum / n_active.where(n_active >= 3)
    core_score = core_mat.sum(axis=1, min_count=1) / core_mat.notna().sum(axis=1).where(core_mat.notna().sum(axis=1) >= 2)

    warn_bull = ext2 | ovh4
    warn_bear = crd1 | crd6
    verdicts = [
        classify_overview_verdict(a, int(n), m5, bool(wb), bool(wbr))
        for a, n, m5, wb, wbr in zip(avg_score, n_active, f5, warn_bull, warn_bear)
    ]
    mat["n_active"] = n_active
    mat["core_score"] = core_score
    mat["avg_score"] = avg_score
    mat["verdict"] = verdicts
    mat["warn_bull"] = warn_bull
    mat["warn_bear"] = warn_bear
    return mat


def overview_backtest(idx_close: pd.Series, ov_bt_df: pd.DataFrame, horizons=(5, 10)):
    """分别检验：① 综合评级(+1偏多/-1偏空/0震荡) vs 全样本基线；② 6个单因子各自的历史预测表现。"""
    px = idx_close.dropna().sort_index()
    if ov_bt_df is None or ov_bt_df.empty or len(px) < 30:
        return pd.DataFrame(), pd.DataFrame()
    common = px.index.intersection(ov_bt_df.index)
    if len(common) < 30:
        return pd.DataFrame(), pd.DataFrame()

    p = px.loc[common]
    df = ov_bt_df.loc[common].copy()
    for h in horizons:
        df[f"fwd_{h}"] = p.shift(-h) / p - 1.0

    # 1) 综合评级分组
    v = df["verdict"].astype(str)
    bucket = pd.Series("震荡分化", index=df.index)
    bucket[v.str.contains("偏多|偏强")] = "偏多信号"
    bucket[v.str.contains("偏空|偏弱")] = "偏空信号"
    bucket[v == "数据不足"] = "数据不足"
    df["bucket"] = bucket
    valid_comp = df[df["bucket"] != "数据不足"].dropna(subset=[f"fwd_{horizons[0]}"])

    comp_rows = []
    if len(valid_comp):
        tot_n = len(valid_comp)
        for lab in ("偏多信号", "震荡分化", "偏空信号", "全样本基准"):
            g = valid_comp if lab == "全样本基准" else valid_comp[valid_comp["bucket"] == lab]
            if g.empty:
                continue
            r5 = g[f"fwd_{horizons[0]}"].dropna()
            r10 = g[f"fwd_{horizons[1]}"].dropna()
            hit = np.nan
            if lab == "偏多信号" and len(r5):
                hit = (r5 > 0).mean()
            elif lab == "偏空信号" and len(r5):
                hit = (r5 < 0).mean()
            comp_rows.append({
                "综合分类": lab, "样本日": len(g), "占比": len(g) / tot_n,
                f"后{horizons[0]}日均涨跌": r5.mean() if len(r5) else np.nan,
                f"后{horizons[1]}日均涨跌": r10.mean() if len(r10) else np.nan,
                f"后{horizons[0]}日上涨占比": (r5 > 0).mean() if len(r5) else np.nan,
                f"方向命中率({horizons[0]}日)": hit,
            })

    # 2) 单因子各自表现（每个因子独立取自身全部有效样本日）
    fac_names = [
        ("f1", "① 避险板块"), ("f2", "② ETF资金"), ("f3", "③ 期指持仓"),
        ("f4", "④ 融资融券"), ("f5", "⑤ 均线"), ("f6", "⑥ 跨境资金"),
    ]
    fac_rows = []
    h0 = horizons[0]
    for fk, flab in fac_names:
        sub = df[[fk, f"fwd_{h0}"]].dropna()
        if sub.empty:
            continue
        pos = sub[sub[fk] > 0][f"fwd_{h0}"]
        neg = sub[sub[fk] < 0][f"fwd_{h0}"]
        dir_sub = sub[sub[fk] != 0]
        hit = ((dir_sub[fk] > 0) == (dir_sub[f"fwd_{h0}"] > 0)).mean() if len(dir_sub) else np.nan
        fac_rows.append({
            "因子": flab, "有效样本日": len(sub), "触发率(|1|)": (sub[fk] != 0).mean(),
            f"+1后{h0}日均涨跌": pos.mean() if len(pos) else np.nan,
            f"-1后{h0}日均涨跌": neg.mean() if len(neg) else np.nan,
            "多空差(+1减-1)": (pos.mean() - neg.mean()) if (len(pos) and len(neg)) else np.nan,
            f"方向命中率({h0}日)": hit,
        })
    return pd.DataFrame(comp_rows), pd.DataFrame(fac_rows)
'''

DASH_PRECOMP_CB = r'''
# 跨境资金与避险相对强弱（供总览与 ⑥ 跨境资金 共用）
proxy_df = ds("proxy_daily", S)
def_all_df = sec_close[sec_close.index <= asof][[c for c in def_codes if c in sec_close.columns]].dropna(how="all") if len(sec_close) else pd.DataFrame()
def_curve_all = an.basket_curve(def_all_df, def_codes) if len(def_all_df) else pd.Series(dtype=float)
cb_res = an.cross_border_analysis(proxy_df, bench_close, def_curve_all, asof) if len(proxy_df) else {}
sec_style_df = an.sector_style_series(sec_close, def_codes, bench_close, asof) if len(sec_close) else pd.DataFrame()

# 六因子总览集成（实盘快照 + 严格 T+1 滞后回测）
ov_live, ov_bt, ov_meta = {}, {}, {}
for nm, info in IDX.items():
    cls = idx_close[info["code"]]
    etf_s_df = an.etf_group_norm_series(flow, aum, config.ETF_GROUPS[nm], asof) if flow is not None else pd.DataFrame()
    etf_f_df = an.etf_factor_series(etf_s_df)
    mar_f_df = an.margin_factor_series(mg[nm], asof) if nm in mg else pd.DataFrame()
    f_info = fut.get(nm)
    spot_lag = False
    if f_info:
        hf_al, spot_lag = an.aligned_hedge_frame(f_info["tot"], f_info["cls"], info["mult"], f_info["etf_f"], f_info["dbal"], config.HEDGE, asof)
        fd_var = fdaily[fdaily["variety"] == info["fut"]] if len(fdaily) else pd.DataFrame()
        fut_s = an.futures_factor_series(hf_al, f_info["basis"], fd_var, cls, asof)
    else:
        hf_al, fut_s = pd.DataFrame(), pd.Series(dtype=float)

    is50 = (info["code"] == "000016") or (nm == "上证50")
    ov_live[nm] = an.overview_index_history(cls, sec_style_df, etf_f_df, fut_s, mar_f_df, cb_res, is_sse50=is50, asof=asof, lag_t1=False)
    ov_bt[nm] = an.overview_index_history(cls, sec_style_df, etf_f_df, fut_s, mar_f_df, cb_res, is_sse50=is50, asof=asof, lag_t1=True)
    ov_meta[nm] = {
        "hf": hf_al, "spot_lag": spot_lag, "etf_f": etf_f_df, "mar_f": mar_f_df,
        "basis": f_info["basis"] if f_info else pd.DataFrame(), "is50": is50,
    }
'''

DASH_TAB0_NEW = r'''# ═════════ 总览 ═════════
with tabs[0]:
    # ── Layer A: 市场风格与流动性政体卡片 ──
    c_reg1, c_reg2 = st.columns(2)
    with c_reg1:
        sec_s_last = sec_style_df.iloc[-1] if len(sec_style_df) else None
        cb_last = cb_res["panel"].iloc[-1] if (cb_res and len(cb_res.get("panel", []))) else None
        s1_txt = "偏防御" if (sec_s_last is not None and sec_s_last["sig"] < 0) else ("偏宽基" if (sec_s_last is not None and sec_s_last["sig"] > 0) else "中性")
        s6_txt = str(cb_last["verdict"]) if cb_last is not None else "无数据"
        if s1_txt == "偏防御" and s6_txt == "背离延续":
            reg_title, reg_delta = "避险防御共振", "①防御占优 + ⑥跨境偏防御"
        elif s1_txt == "偏宽基" and s6_txt == "背离收敛":
            reg_title, reg_delta = "宽基风险偏好共振", "①防御跑输 + ⑥跨境偏宽基"
        else:
            reg_title, reg_delta = "风格震荡 / 分化", f"①在岸{s1_txt} ｜ ⑥跨境{s6_txt}"
        crd_p = sec_s_last["pct120"] if sec_s_last is not None else np.nan
        if pd.notna(crd_p) and crd_p >= 0.85:
            reg_delta += f" ｜ ⚠️防御拥挤{crd_p:.0%}"
        c_reg1.metric("市场风格政体（①避险 + ⑥跨境）", reg_title, reg_delta)
    with c_reg2:
        tot_etf5 = sum(v["d5"] for v in etf_res.values() if v and pd.notna(v.get("d5"))) if etf_res else np.nan
        mar_300 = mg_sum.get(config.BENCH, {})
        m_txt = f"沪深300融资分位 {pct(mar_300.get('pct'), 0)}" if "pct" in mar_300 else "融资暂无分位"
        e_txt = f"四大宽基ETF 5日净申购 {num(tot_etf5)} 亿" if tot_etf5 == tot_etf5 else "ETF份额无数据"
        c_reg2.metric("在岸资金与杠杆（②ETF + ④融资）", e_txt, m_txt)

    # ── Layer B: 6 因子信号矩阵（含综合评级）──
    def _badge(v, is50=False, is_style=False, note=""):
        if is50 and is_style:
            return "⚪ 0(重合)"
        if v is None or v != v:
            return "—"
        if v > 0:
            return f"🔴 +1{note}"
        if v < 0:
            return f"🟢 -1{note}"
        return f"⚪ 0{note}"

    sig_rows, detail_rows = [], []
    for nm, info in IDX.items():
        r = an.period_returns(idx_close[info["code"]], asof)
        ov_df = ov_live.get(nm, pd.DataFrame())
        om = ov_meta.get(nm, {})
        is50 = om.get("is50", False)

        if len(ov_df):
            last_ov = ov_df.iloc[-1]
            notes = []
            ef_df = om.get("etf_f", pd.DataFrame())
            mf_df = om.get("mar_f", pd.DataFrame())
            e_note = "护盘" if (len(ef_df) and bool(ef_df["extreme"].iloc[-1])) else ""
            m_note = ""
            if len(mf_df):
                if bool(mf_df["overheated"].iloc[-1]):
                    m_note = "(过热)"
                    notes.append("融资≥85%过热")
                elif bool(mf_df["ice_cold"].iloc[-1]):
                    m_note = "(冰点)"
                    notes.append("融资≤15%冰点")
            if e_note:
                notes.append("ETF极端大额")
            if bool(last_ov["warn_bear"]):
                notes.append("防御≥85%高拥挤")
            if om.get("spot_lag"):
                notes.append("期指对冲用T-1现货")

            sig_rows.append({
                "指数": nm, "当日": r["day"], "本周": r["wtd"], "本月": r["mtd"],
                "①避险": _badge(last_ov["f1"], is50, True),
                "②ETF": _badge(last_ov["f2"], note=f"({e_note})" if e_note else ""),
                "③期指": _badge(last_ov["f3"]),
                "④融资": _badge(last_ov["f4"], note=m_note),
                "⑤均线": _badge(last_ov["f5"]),
                "⑥跨境": _badge(last_ov["f6"], is50, True),
                "自身趋势(②~⑤)": last_ov["core_score"],
                "综合得分": last_ov["avg_score"],
                "有效因子": f"{int(last_ov['n_active'])}/{4 if is50 else 6}",
                "综合信号": str(last_ov["verdict"]),
                "风险/状态提示": "；".join(notes) if notes else "正常",
            })

        e, m = etf_res.get(nm), mg_sum.get(nm, {})
        hf_al = om.get("hf", pd.DataFrame())
        label, d_short, ann = "—", np.nan, np.nan
        if len(hf_al):
            label, d_short = hf_al["label"].iloc[-1], hf_al["short_add"].iloc[-1]
        bs_df = om.get("basis", pd.DataFrame())
        if len(bs_df) and bs_df["ann"].notna().any():
            bs_sub = bs_df[bs_df.index <= asof]["ann"].dropna()
            if len(bs_sub):
                ann = bs_sub.iloc[-1]
        detail_rows.append({
            "指数": nm, "均线状态": ma[nm]["state"],
            "ETF5日(亿)": e["d5"] if e else np.nan,
            "ETF5日/规模": e["norm_d5"] if e else np.nan,
            "融资买入占比": m.get("ratio", np.nan),
            "融资120日分位": m.get("pct", np.nan),
            "空单5日增量(亿)": d_short, "年化基差": ann, "空单性质": label,
        })

    if sig_rows:
        show(
            pd.DataFrame(sig_rows),
            pct_cols=["当日", "本周", "本月"],
            num_cols=["自身趋势(②~⑤)", "综合得分"],
            color_cols=["当日", "本周", "本月", "自身趋势(②~⑤)", "综合得分"],
        )
    st.caption("🔴 +1 = 偏多 ｜ ⚪ 0 = 中性/死区/事件观察 ｜ 🟢 -1 = 偏空。上证50 因重仓银行/石油/电力与防御板块高度重合，①避险与⑥跨境记为 0(重合) 且分母按 4 因子计，其余宽基按 6 因子计。")

    with st.expander("展开各指数底层数值明细（ETF规模比、融资分位、期指基差与空单性质）", expanded=False):
        show(
            pd.DataFrame(detail_rows),
            pct_cols=["ETF5日/规模", "融资买入占比", "融资120日分位", "年化基差"],
            num_cols=["ETF5日(亿)", "空单5日增量(亿)"],
            color_cols=["ETF5日(亿)", "ETF5日/规模"],
        )

    # ── 数据时效 / 缺失与异常诊断（保留原有一致性检查）──
    _asof = pd.Timestamp(asof)
    _stale = []
    for _lab, _df_, _cols in (("融资融券", margin, "④融资"), ("ETF份额", shares, "②ETF"),
                              ("期指持仓排名", rank, "③期指"), ("期指日行情", fdaily, "③年化基差")):
        if len(_df_) and pd.Timestamp(_df_["date"].max()) < _asof:
            _stale.append(f"**{_lab}** 只到 {pd.Timestamp(_df_['date'].max()).date()}（统计截止日 {_asof.date()}）→ 影响列：{_cols}")
    if _stale:
        st.warning("⏳ 以下数据**不是最新**，相关因子使用最近可用日数据（融资/ETF份额通常次日早晨披露）：\n\n- "
                   + "\n- ".join(_stale))

    problems = []
    for nm, info in IDX.items():
        sr = idx_close[info["code"]].dropna()
        if len(sr) > 30:
            chg = sr.pct_change().abs().tail(250)
            if chg.max() > 0.12:
                problems.append(f"**{nm}({info['code']})** 日线在 {chg.idxmax().date()} 单日变动 {chg.max():.0%}，"
                                "疑似混入了同号个股数据或数据错位（000016/000905/000852 也是个股代码）")
        f_ = fut.get(nm)
        if f_ is not None and len(f_["basis"]) and bool(f_["basis"]["scale_bad"].iloc[-1]):
            b_ = f_["basis"].iloc[-1]
            problems.append(f"**{nm}** 期货点位 {b_['F']:.0f} 与指数收盘 {b_['S']:.2f} 相差过大，指数日线口径不对"
                            "（年化基差已不显示；先用「数据」页的健康检查/诊断确认指数文件是否被同号个股覆盖）")
    for p_ in problems:
        st.error(p_)

    st.markdown(f"**避险防御篮子相对沪深300**：当日 {pct(def_rel['day'])} ｜ 本周 {pct(def_rel['wtd'])} ｜ 本月 {pct(def_rel['mtd'])}"
                f"（防御篮子 = 角色为\"防御\"的板块成分股前复权等权；正值表示防御跑赢大盘）")

    # ── Layer C: 6 因子与综合信号历史回测检验 ──
    with st.expander("历史回测检验：综合信号与 6 个单因子在各指数上是否真有预测区分度？", expanded=False):
        bt_idx = st.radio("选择检验指数", list(IDX.keys()), horizontal=True, key="ov_bt_idx")
        bt_df = ov_bt.get(bt_idx, pd.DataFrame())
        comp_bt, fac_bt = an.overview_backtest(idx_close[IDX[bt_idx]["code"]], bt_df, horizons=(5, 10))
        if len(comp_bt):
            n_samp = int(comp_bt.loc[comp_bt["综合分类"] == "全样本基准", "样本日"].iloc[0]) if (comp_bt["综合分类"] == "全样本基准").any() else 0
            if n_samp < 120:
                st.warning(f"当前 6 因子重叠样本仅 **{n_samp}** 个交易日（通常因融资/ETF份额/期指默认只回补了 60 天），统计样本偏少；可到「数据」页加大「首次回补交易日数」后重拉。")
            st.markdown("**① 综合信号分组表现（已对 T+1 早晨披露的 ETF/融资做滞后 1 日防未来对齐）**")
            show(comp_bt, pct_cols=["占比", "后5日均涨跌", "后10日均涨跌", "后5日上涨占比", "方向命中率(5日)"],
                 color_cols=["后5日均涨跌", "后10日均涨跌"])
        else:
            st.info(f"{bt_idx} 当前有效重叠交易日不足 30 天，尚无法运行综合回测。")
        if len(fac_bt):
            st.markdown("**② 6 个单因子各自独立表现（分别取各自全部可用历史）**")
            show(fac_bt, pct_cols=["触发率(|1|)", "+1后5日均涨跌", "-1后5日均涨跌", "多空差(+1减-1)", "方向命中率(5日)"],
                 color_cols=["+1后5日均涨跌", "-1后5日均涨跌", "多空差(+1减-1)"])
            st.caption("判断标准：若「+1后5日均涨跌」显著高于「-1后5日均涨跌」（多空差显著为正），说明该因子在该指数上确实具备正向区分度；反之接近 0% 说明近乎随机噪声。")

'''

DASH_TAB6_CLEAN = r'''# ═════════ ⑥ 跨境资金 ═════════
with tabs[6]:
    st.subheader("维度 1：跨境资金与宏观利率（领先 / 共振风格判断）")

    have_syms = sorted(set(proxy_df["symbol"])) if len(proxy_df) else []
    need_rates = [s for s in ("US10Y", "CN10Y") if s not in have_syms]
    with st.container(border=True):
        u1, u2 = st.columns([3, 1.3])
        u1.caption(
            f"当前缓存序列（{len(have_syms)} 个）：{', '.join(have_syms) if have_syms else '无'}"
            + (f"　｜　⚠️ 缺少利率数据 {', '.join(need_rates)}，请点击右侧按钮更新" if need_rates else "")
        )
        if u2.button("🔄 更新外资代理与利率 (⑤)", width="stretch", key="cb_upd_btn"):
            run_job("⑤ 外资代理与利率", lambda p: fetchers.download_proxies(p))'''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    cnt = src.count(old)
    if cnt != 1:
        raise RuntimeError(f"{label}：期望找到 1 处锚点，实际 {cnt} 处。")
    return src.replace(old, new)


def patch_analytics(src: str):
    if "def overview_index_history(" in src:
        return None
    return src.rstrip() + "\n" + ANALYTICS_OV6.lstrip("\n")


def patch_dashboard(src: str):
    if "ov_live" in src:
        return None
    if "⑥ 跨境资金" not in src:
        raise RuntimeError("views/dashboard.py 尚未包含「⑥ 跨境资金」，请先运行 apply_cross_border_patch.py。")

    # 1. 在 tabs = st.tabs(...) 之前插入预计算（cb_res + sec_style_df + ov_live/ov_bt）
    tab_line = 'tabs = st.tabs(["总览", "① 避险板块", "② ETF资金", "③ 期指持仓", "④ 融资融券", "⑤ 均线", "⑥ 跨境资金", "说明与限制"])'
    src = replace_once(src, tab_line, DASH_PRECOMP_CB.lstrip("\n") + "\n" + tab_line, "views/dashboard.py precompute")

    # 2. 替换 tabs[0] 总览区块
    start_t0 = "# ═════════ 总览 ═════════\nwith tabs[0]:"
    end_t0 = "# ═════════ ① 避险板块 ═════════\nwith tabs[1]:"
    i0, i1 = src.find(start_t0), src.find(end_t0)
    if i0 < 0 or i1 < 0 or i1 <= i0:
        raise RuntimeError("views/dashboard.py 中未找到 总览 / ① 避险板块 锚点。")
    src = src[:i0] + DASH_TAB0_NEW + src[i1:]

    # 3. 清理 tabs[6] 内重复的 cb_res 计算
    old_t6 = '''# ═════════ ⑥ 跨境资金 ═════════
with tabs[6]:
    st.subheader("维度 1：跨境资金与宏观利率（领先 / 共振风格判断）")
    proxy_df = ds("proxy_daily", S)
    def_all_df = sec_close[sec_close.index <= asof][[c for c in def_codes if c in sec_close.columns]].dropna(how="all") if len(sec_close) else pd.DataFrame()
    def_curve_all = an.basket_curve(def_all_df, def_codes) if len(def_all_df) else pd.Series(dtype=float)

    have_syms = sorted(set(proxy_df["symbol"])) if len(proxy_df) else []
    need_rates = [s for s in ("US10Y", "CN10Y") if s not in have_syms]
    with st.container(border=True):
        u1, u2 = st.columns([3, 1.3])
        u1.caption(
            f"当前缓存序列（{len(have_syms)} 个）：{', '.join(have_syms) if have_syms else '无'}"
            + (f"　｜　⚠️ 缺少利率数据 {', '.join(need_rates)}，请点击右侧按钮更新" if need_rates else "")
        )
        if u2.button("🔄 更新外资代理与利率 (⑤)", width="stretch", key="cb_upd_btn"):
            run_job("⑤ 外资代理与利率", lambda p: fetchers.download_proxies(p))

    cb_res = an.cross_border_analysis(proxy_df, bench_close, def_curve_all, asof) if len(proxy_df) else {}'''
    src = replace_once(src, old_t6, DASH_TAB6_CLEAN, "views/dashboard.py tabs[6] dedup")
    return src


def main():
    tasks = [
        (ROOT / "analytics.py", patch_analytics),
        (ROOT / "views" / "dashboard.py", patch_dashboard),
    ]
    planned = []
    for path, fn in tasks:
        if not path.exists():
            sys.exit(f"[中止] 找不到 {path}，请把脚本放在项目根目录运行。")
        old = path.read_text(encoding="utf-8-sig")
        try:
            new = fn(old)
        except Exception as e:
            sys.exit(f"[中止] {path.name} 失败：{e}\n所有文件均未修改。")
        if new is not None:
            compile(new, str(path), "exec")
            planned.append((path, new))
        else:
            print(f"[跳过] {path.relative_to(ROOT)} 已包含六因子总览集成")

    for path, new in planned:
        bak = path.with_suffix(path.suffix + ".bak_before_overview6")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(new, encoding="utf-8")
        print(f"[更新] {path.relative_to(ROOT)}（备份：{bak.name}）")

    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("进入首页「总览」标签页即可看到：")
    print("  1. 市场风格政体 & 资金杠杆总控面板；")
    print("  2. 四大指数 6 因子信号矩阵（①避险 ②ETF ③期指 ④融资 ⑤均线 ⑥跨境）及综合评级；")
    print("  3. 展开底部的「历史回测检验」，可查看综合信号与 6 个单因子各自的历史胜率与多空区分度。")


if __name__ == "__main__":
    main()