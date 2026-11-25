# -*- coding: utf-8 -*-
"""一键更新：
  1. tushare_src.py + fetchers.py：修复 akshare 1.20.65 更名导致的⑤外资代理失败，避开东财限流，
     并接入 US10Y (pro.us_tycr / bond_gb_us_sina) 与 CN10Y (bond_gb_zh_sina)；
  2. analytics.py：新增「维度 1：跨境资金与宏观利率」分析与基线校准回测函数（含美股/NY时区+1日零未来函数对齐）；
  3. views/dashboard.py + views/data.py：在首页新增「⑥ 跨境资金」标签页，与「防御综合 / 沪深300」双轴叠加及信号回测。

用法（在项目根目录运行）：
    python apply_cross_border_patch.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TUSHARE_US10Y = r'''

def us_tycr_hist() -> pd.DataFrame:
    """Tushare 美债收益率曲线（取 10 年期 y10）。"""
    df = _windowed("us_tycr", config.PX_START, dt.date.today(), fields="date,y10")
    if df.empty or "y10" not in df.columns:
        raise ValueError("Tushare us_tycr 为空（无权限或无数据）")
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"].astype(str), errors="coerce"),
        "close": pd.to_numeric(df["y10"], errors="coerce"),
    }).dropna()
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
'''

FETCHERS_PROXIES_NEW = r'''# ───────────── ⑤ 外资代理指标（兼容 akshare 1.20.44+ 更名，优先 Tushare/新浪）─────────────
def _ak_fn(ak, *names):
    for n in names:
        f = getattr(ak, n, None)
        if f is not None:
            return f
    raise AttributeError(f"akshare {ak_version()} 缺少接口：{' / '.join(names)}")


def _boc_mid(ak):
    """备用：央行人民币对美元中间价（新浪-中行外汇牌价）。注意是在岸中间价，保存为 USDCNY_MID。"""
    fn = getattr(ak, "fx_boc_rate_sina", None)
    if fn is not None:
        raw = fn(symbol="美元")
    else:
        raw = ak.currency_boc_sina(symbol="美元", start_date=config.PX_START, end_date=pd.Timestamp.today().strftime("%Y%m%d"))
    col = next((c for c in raw.columns if "中间价" in str(c)), None)
    dcol = next((c for c in raw.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    if col is None or dcol is None:
        raise ParseError(f"中行牌价返回列与预期不符：{list(raw.columns)}")
    close = pd.to_numeric(raw[col], errors="coerce")
    if close.median() > 50:                                    # 牌价按"每100美元"报价
        close = close / 100
    return pd.DataFrame({"date": raw[dcol], "close": close})


def _bond_yield_em(ak, col_kw: str):
    """东财中美国债收益率备用（限流时排在 Sina/Tushare 之后）。"""
    fn = _ak_fn(ak, "bond_zh_us_yield_em", "bond_zh_us_rate")
    raw = fn()
    dcol = next((c for c in raw.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    vcol = next((c for c in raw.columns if col_kw in str(c)), None)
    if dcol is None or vcol is None:
        raise ParseError(f"中美国债接口返回列不符：{list(raw.columns)}")
    return pd.DataFrame({"date": raw[dcol], "close": pd.to_numeric(raw[vcol], errors="coerce")}).dropna()


def proxy_candidates():
    """每个指标按优先级给出多个来源：(指标, 说明, [(来源名, 取数函数, 保存时的代码[None=同指标]), ...])。
    优先 Tushare 与新浪，避免东财 push2 高频限流。"""
    ts = "tushare" in _sources()
    a50 = [(f"新浪 {config.A50_SYMBOL}", lambda ak: _ak_fn(ak, "futures_foreign_hist_sina", "futures_foreign_hist")(symbol=config.A50_SYMBOL), None)]
    usd = [("新浪 中行中间价(替代,存为USDCNY_MID)", _boc_mid, "USDCNY_MID"),
           ("东财 USDCNH", lambda ak: _ak_fn(ak, "fx_hist_em", "forex_hist_em")(symbol="USDCNH"), None)]
    hsi = [("新浪 恒指", lambda ak: _ak_fn(ak, "index_hk_daily_sina", "stock_hk_index_daily_sina")(symbol="HSI"), None),
           ("东财 恒指", lambda ak: _ak_fn(ak, "index_hk_daily_em", "stock_hk_index_daily_em")(symbol="HSI"), None)]
    hsahp = [("新浪 AH溢价", lambda ak: _ak_fn(ak, "index_hk_daily_sina", "stock_hk_index_daily_sina")(symbol="HSAHP"), None),
             ("东财 AH溢价", lambda ak: _ak_fn(ak, "index_hk_daily_em", "stock_hk_index_daily_em")(symbol="HSAHP"), None)]
    us10y = [("新浪 美国10年期国债", lambda ak: ak.bond_gb_us_sina(symbol="美国10年期国债"), None),
             ("东财 美债10Y", lambda ak: _bond_yield_em(ak, "美国国债收益率10年"), None)]
    cn10y = [("新浪 中国10年期国债", lambda ak: ak.bond_gb_zh_sina(symbol="中国10年期国债"), None),
             ("东财 中债10Y", lambda ak: _bond_yield_em(ak, "中国国债收益率10年"), None)]
    if ts:
        a50.append(("Tushare 富时A50现货(A50_IDX)", lambda ak: tsrc.index_global_hist("XIN9"), "A50_IDX"))
        usd.insert(0, ("Tushare fx_daily", lambda ak: tsrc.fx_hist("USDCNH.FXCM"), None))
        hsi.insert(0, ("Tushare index_global", lambda ak: tsrc.index_global_hist("HSI"), None))
        us10y.insert(0, ("Tushare us_tycr(y10)", lambda ak: tsrc.us_tycr_hist(), None))
    else:
        a50.append(("东财 国际期货", lambda ak: ak.futures_global_hist_em(symbol="富时中国A50指数当月连续"), None))
    return [
        ("A50", "富时中国A50期指", a50),
        ("USDCNH", "离岸人民币", usd),
        ("HSAHP", "恒生AH股溢价指数", hsahp),
        ("HSI", "恒生指数", hsi),
        ("FXI", "中国大盘ETF(美股)", [("新浪", lambda ak: _ak_fn(ak, "stock_us_hist_sina", "stock_us_daily")(symbol="FXI"), None)]),
        ("KWEB", "中概互联网ETF(美股)", [("新浪", lambda ak: _ak_fn(ak, "stock_us_hist_sina", "stock_us_daily")(symbol="KWEB"), None)]),
        ("US10Y", "10年期美债收益率(%)", us10y),
        ("CN10Y", "10年期中债收益率(%)", cn10y),
    ]'''

ANALYTICS_CROSS_BORDER = r'''

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
'''

DASHBOARD_TAB6 = r'''# ═════════ ⑥ 跨境资金 ═════════
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

    cb_res = an.cross_border_analysis(proxy_df, bench_close, def_curve_all, asof) if len(proxy_df) else {}
    if not cb_res:
        st.info("还没有外资代理与利率数据，请点击上方「🔄 更新外资代理与利率 (⑤)」。")
    else:
        panel = cb_res["panel"]
        meta = cb_res["meta"].copy()
        last = panel.iloc[-1]

        # 预警提示
        rel_p = last["rel_pct120"]
        if int(last["conv_streak"]) >= 5:
            st.success(f"🔔 **强反转预警**：跨境维度已连续 **{int(last['conv_streak'])}** 个交易日判定「背离收敛」，关注风格向沪深300/宽基切换。")
        elif pd.notna(rel_p) and rel_p >= 0.85 and pd.notna(last["imp_score"]) and last["imp_score"] <= -0.4:
            st.warning(f"🔔 **高拥挤 + 短线转收敛预警**：防御/沪深300 处于近120日 **{rel_p:.0%}** 高分位，且5日边际动向已转向宽基，注意均值回归风险。")
        elif pd.notna(rel_p) and rel_p >= 0.85 and last["verdict"] == "背离延续":
            st.warning(f"⚠️ **高拥挤度提示**：虽然宏观仍在「背离延续」，但防御/沪深300 已处于近120日 **{rel_p:.0%}** 高分位，不宜盲目追高防御。")

        # 与 ② ETF / ④ 融资 的交叉验证提示
        etf_300 = etf_res.get(config.BENCH) if etf_res else None
        mg_300 = mg_sum.get(config.BENCH) if mg_sum else None
        if etf_300 or mg_300:
            etf_txt = f"沪深300 ETF 近5日净申购 {num(etf_300['d5'])} 亿" if etf_300 else ""
            mg_txt = f"沪深300 融资买入占比分位 {pct(mg_300.get('pct'))}" if mg_300 else ""
            st.caption("跨维度在岸验证参考：" + "　｜　".join(x for x in (etf_txt, mg_txt) if x))

        # 顶栏指标卡
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("维度判定", str(last["verdict"]), f"有效指标 {int(last['n_avail'])}/5")
        c2.metric("投票 (延续 / 中性 / 收敛)",
                  f"{int(last['n_cont'])} / {int(last['n_avail'] - last['n_cont'] - last['n_conv'])} / {int(last['n_conv'])}",
                  f"净得分 {last['net_score']:+.2f}" if pd.notna(last["net_score"]) else None)
        c3.metric("防御/沪深300 120日分位", pct(rel_p, 0), f"比值 {num(last['rel_ratio'], 3)}")
        c4.metric("中美 10Y 利差 (中-美)", num(last["spread_cn_us"], 2, "%") if pd.notna(last["spread_cn_us"]) else "—")
        c5.metric("5日短期冲量均值", f"{last['imp_score']:+.2f}" if pd.notna(last["imp_score"]) else "—")

        # 5 指标明细表
        vote_txt = {1.0: "🔴 背离延续(偏防御)", -1.0: "🟢 背离收敛(偏宽基)", 0.0: "⚪ 中性"}
        imp_txt = {1.0: "偏延续(+1)", -1.0: "偏收敛(-1)", 0.0: "中性(0)"}
        disp = pd.DataFrame({
            "指标": meta["指标"],
            "最新值": [num(v, 3 if not r else 2, "%" if r else "") for v, r in zip(meta["最新值"], meta["is_rate"])],
            "MA5": [num(v, 3 if not r else 2) for v, r in zip(meta["MA5"], meta["is_rate"])],
            "MA20": [num(v, 3 if not r else 2) for v, r in zip(meta["MA20"], meta["is_rate"])],
            "MA20近5日斜率": [pct(v, 2) if not r else (num(v * 100, 1, " bps") if pd.notna(v) else "—")
                              for v, r in zip(meta["MA20近5日变化"], meta["is_rate"])],
            "短期5日动向": [imp_txt.get(x, "数据不足") for x in meta["短期5日动向"]],
            "趋势投票": [vote_txt.get(x, "数据不足") for x in meta["趋势投票"]],
            "时区对齐": meta["时区对齐"],
        })
        st.dataframe(disp, hide_index=True, width="stretch")

        # 双轴叠加图
        win_cb = st.radio("叠加走势窗口", ["近120日", "近1年(250日)", "近3年(750日)", "全部历史"],
                          index=1, horizontal=True, key="cb_win")
        wn = {"近120日": 120, "近1年(250日)": 250, "近3年(750日)": 750, "全部历史": 0}[win_cb]
        p_sub = panel.tail(wn) if wn > 0 else panel
        p_sub = p_sub[p_sub["n_avail"] >= 3]

        if len(p_sub) >= 2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if p_sub["rel_ratio"].notna().any():
                fig.add_trace(
                    go.Scatter(x=p_sub.index, y=p_sub["rel_ratio"], name="防御综合 ÷ 沪深300（左轴，上升=防御走强）",
                               line=dict(color="#d62728", width=2.3)),
                    secondary_y=False,
                )
            fig.add_trace(
                go.Scatter(x=p_sub.index, y=p_sub["net_score_ma5"], name="跨境净得分5日均（右轴，>0偏延续/<0偏收敛）",
                           line=dict(color="#1f77b4", width=1.8)),
                secondary_y=True,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#888", secondary_y=True)
            fig.update_yaxes(title_text="防御 ÷ 沪深300 相对比值", secondary_y=False)
            fig.update_yaxes(title_text="跨境综合净得分 [-1, +1]", range=[-1.15, 1.15], secondary_y=True)
            fig.update_layout(height=420, margin=dict(t=30, b=20), hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig, width="stretch")

        with st.expander("各单项指标走势与中美 10Y 利差"):
            s_df = cb_res["series"].loc[p_sub.index] if len(p_sub) else cb_res["series"]
            cA, cB = st.columns(2)
            for idx_i, (col_k, title_t) in enumerate([
                ("kweb_a50", "KWEB ÷ A50（下降=成长跑输防御）"),
                ("hsahp", "恒生AH股溢价 HSAHP（上升=外资/港股情绪折价）"),
                ("usdcnh", "外汇锚 USDCNH/中间价（上升=人民币贬值）"),
                ("spread", "10Y 美债 vs 中债收益率 (%)"),
            ]):
                f_sub = go.Figure()
                if col_k == "spread":
                    if s_df["us10y"].notna().any():
                        f_sub.add_scatter(x=s_df.index, y=s_df["us10y"], name="美债10Y")
                    if s_df["cn10y"].notna().any():
                        f_sub.add_scatter(x=s_df.index, y=s_df["cn10y"], name="中债10Y")
                    if p_sub["spread_cn_us"].notna().any():
                        f_sub.add_scatter(x=p_sub.index, y=p_sub["spread_cn_us"], name="中-美利差", line=dict(dash="dot"))
                else:
                    k_map = {"usdcnh": "fx"}.get(col_k, col_k)
                    sr = s_df[k_map].dropna()
                    if len(sr):
                        f_sub.add_scatter(x=sr.index, y=sr, name=k_map)
                        f_sub.add_scatter(x=sr.index, y=sr.rolling(20).mean(), name="MA20", line=dict(dash="dash"))
                f_sub.update_layout(height=280, margin=dict(t=30, b=10), title=title_t)
                (cA if idx_i % 2 == 0 else cB).plotly_chart(f_sub, width="stretch")

        # 历史回测检验（基线对照）
        st.markdown("#### 历史传导检验（对比「全样本无条件基线」，检查是否真有领先区分度）")
        val = an.cross_border_validation(cb_res, horizons=(5, 10))
        if len(val):
            show(
                val,
                pct_cols=["占比", "后5日相对超额均值", "后10日相对超额均值", "后5日防御跑赢占比", "按方向命中率(5日)"],
                color_cols=["后5日相对超额均值", "后10日相对超额均值"],
            )
            st.caption("阅读方法：若「背离延续」的后5日/10日相对超额显著高于「全样本基准」，且「背离收敛」显著低于基准（甚至转负），"
                       "才说明该维度在统计上具备有效区分度。美股/美债/NY外汇均已做 +1 自然日对齐，回测无未来函数。")
        else:
            st.info("有效历史重叠交易日不足 30 天，尚无法做历史检验（请先更新代理数据并确保防御板块与沪深300日线对齐）。")

'''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    cnt = src.count(old)
    if cnt != 1:
        raise RuntimeError(f"{label}：期望找到 1 处锚点，实际 {cnt} 处。")
    return src.replace(old, new)


def patch_tushare(src: str):
    if "def us_tycr_hist(" in src:
        return None
    anchor = '''def index_global_hist(ts: str) -> pd.DataFrame:
    df = _windowed("index_global", config.PX_START, dt.date.today(), ts_code=ts)
    if df.empty:
        raise ValueError(f"Tushare index_global({ts}) 为空")
    return pd.DataFrame({"date": pd.to_datetime(df["trade_date"]), "open": df["open"], "high": df["high"],
                         "low": df["low"], "close": df["close"], "volume": df.get("vol")})'''
    return replace_once(src, anchor, anchor + TUSHARE_US10Y, "tushare_src.py us_tycr_hist")


def patch_fetchers(src: str):
    if "us_tycr_hist" in src:
        return None
    old_block = '''# ───────────── ⑤ 外资代理指标 ─────────────
def _boc_mid(ak):
    """备用：央行人民币对美元中间价（新浪-中行外汇牌价历史）。注意是在岸中间价，不是离岸即期，保存为 USDCNY_MID。"""
    raw = ak.currency_boc_sina(symbol="美元", start_date=config.PX_START, end_date=pd.Timestamp.today().strftime("%Y%m%d"))
    col = next((c for c in raw.columns if "中间价" in str(c)), None)
    dcol = next((c for c in raw.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    if col is None or dcol is None:
        raise ParseError(f"中行牌价返回列与预期不符：{list(raw.columns)}")
    close = pd.to_numeric(raw[col], errors="coerce")
    if close.median() > 50:                                    # 牌价按"每100美元"报价
        close = close / 100
    return pd.DataFrame({"date": raw[dcol], "close": close})


def proxy_candidates():
    """每个指标按优先级给出多个来源：(指标, 说明, [(来源名, 取数函数, 保存时的代码[None=同指标]), ...])。
    已按实测结果剔除不可用来源（新浪 XIN9、新浪 HSAHP）。东财接口在部分网络下断连，所以排在后面。"""
    ts = "tushare" in _sources()
    a50 = [(f"新浪 {config.A50_SYMBOL}", lambda ak: ak.futures_foreign_hist(symbol=config.A50_SYMBOL), None),
           ("东财 国际期货", lambda ak: ak.futures_global_hist_em(symbol="富时中国A50指数当月连续"), None)]
    usd = [("东财", lambda ak: ak.forex_hist_em(symbol="USDCNH"), None),
           ("中行牌价·央行中间价(替代,存为USDCNY_MID)", _boc_mid, "USDCNY_MID")]
    hsi = [("新浪", lambda ak: ak.stock_hk_index_daily_sina(symbol="HSI"), None),
           ("东财", lambda ak: ak.stock_hk_index_daily_em(symbol="HSI"), None)]
    if ts:
        a50.append(("Tushare 富时A50现货指数(替代,存为A50_IDX)", lambda ak: tsrc.index_global_hist("XIN9"), "A50_IDX"))
        usd.insert(0, ("Tushare fx_daily", lambda ak: tsrc.fx_hist("USDCNH.FXCM"), None))
        hsi.append(("Tushare index_global", lambda ak: tsrc.index_global_hist("HSI"), None))
    return [
        ("A50", "富时中国A50期指", a50),
        ("USDCNH", "离岸人民币", usd),
        ("HSAHP", "恒生AH股溢价指数", [("东财", lambda ak: ak.stock_hk_index_daily_em(symbol="HSAHP"), None)]),
        ("HSI", "恒生指数", hsi),
        ("FXI", "中国大盘ETF(美股)", [("新浪", lambda ak: ak.stock_us_daily(symbol="FXI"), None)]),
        ("KWEB", "中概互联网ETF(美股)", [("新浪", lambda ak: ak.stock_us_daily(symbol="KWEB"), None)]),
    ]'''
    return replace_once(src, old_block, FETCHERS_PROXIES_NEW, "fetchers.py proxy_candidates")


def patch_analytics(src: str):
    if "def cross_border_analysis(" in src:
        return None
    return src.rstrip() + "\n" + ANALYTICS_CROSS_BORDER.lstrip("\n")


def patch_dashboard(src: str):
    if "⑥ 跨境资金" in src:
        return None
    if "make_subplots" not in src:
        src = replace_once(
            src,
            "import plotly.graph_objects as go",
            "import plotly.graph_objects as go\nfrom plotly.subplots import make_subplots",
            "views/dashboard.py make_subplots import",
        )
    if "run_job" not in src:
        src = replace_once(
            src,
            "render_freshness, show, stamp",
            "render_freshness, run_job, show, stamp",
            "views/dashboard.py run_job import",
        )
    src = replace_once(
        src,
        'tabs = st.tabs(["总览", "① 避险板块", "② ETF资金", "③ 期指持仓", "④ 融资融券", "⑤ 均线", "说明与限制"])',
        'tabs = st.tabs(["总览", "① 避险板块", "② ETF资金", "③ 期指持仓", "④ 融资融券", "⑤ 均线", "⑥ 跨境资金", "说明与限制"])',
        "views/dashboard.py st.tabs",
    )
    src = replace_once(
        src,
        "# ═════════ 说明 ═════════\nwith tabs[6]:",
        DASHBOARD_TAB6 + "# ═════════ 说明 ═════════\nwith tabs[7]:",
        "views/dashboard.py tabs[6]",
    )
    return src


def patch_data_view(src: str):
    old = 'if b[1].button("⑤ 外资代理（A50/USDCNH/AH溢价…）", width="stretch"):'
    new = 'if b[1].button("⑤ 外资代理（A50/USDCNH/AH溢价/美债中债10Y）", width="stretch"):'
    if new in src or old not in src:
        return None
    return src.replace(old, new, 1)


def main():
    tasks = [
        (ROOT / "tushare_src.py", patch_tushare),
        (ROOT / "fetchers.py", patch_fetchers),
        (ROOT / "analytics.py", patch_analytics),
        (ROOT / "views" / "dashboard.py", patch_dashboard),
        (ROOT / "views" / "data.py", patch_data_view),
    ]

    planned = []
    for path, fn in tasks:
        if not path.exists():
            sys.exit(f"[中止] 找不到 {path}，请将脚本放在项目根目录运行。")
        old = path.read_text(encoding="utf-8-sig")
        try:
            new = fn(old)
        except Exception as e:
            sys.exit(f"[中止] {path.name} 失败：{e}\n所有文件均未修改。")
        if new is not None:
            compile(new, str(path), "exec")
            planned.append((path, new))
        else:
            print(f"[跳过] {path.relative_to(ROOT)} 已经包含该功能")

    for path, new in planned:
        bak = path.with_suffix(path.suffix + ".bak_before_cross_border")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(new, encoding="utf-8")
        print(f"[更新] {path.relative_to(ROOT)}（备份：{bak.name}）")

    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("建议先执行一次 python diagnose_proxies.py（确认代理全绿），或在网页「⑥ 跨境资金」页直接点「🔄 更新外资代理与利率 (⑤)」。")


if __name__ == "__main__":
    main()