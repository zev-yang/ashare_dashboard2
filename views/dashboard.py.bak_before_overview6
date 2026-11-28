"""页面一：指数分析。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import analytics as an
import config
import fetchers
import loaders
from common import (ETF_ALL, IDX, IDX_CODES, NAME, SEC_CODES, amount_panel, close_panel, ds, num, pct,
                    render_freshness, run_job, show, stamp)

st.title("A股短期趋势监控")
if config.DEMO:
    st.warning("演示模式：全部为随机模拟数据，不代表真实行情。关闭方法：在 PowerShell 运行 "
               "`Remove-Item Env:ASHARE_DEMO`（或新开一个终端窗口），再重新 `streamlit run app.py`。")
render_freshness()

S = stamp()

# ───────────── 指数日线 & as-of ─────────────
idx_close = close_panel(tuple(IDX_CODES), S, tail=None)
BENCH_CODE = IDX[config.BENCH]["code"]
if idx_close.empty or BENCH_CODE not in idx_close.columns:
    st.error(f"没有读到基准指数 {config.BENCH}（{BENCH_CODE}）的日线，程序无法继续。下表说明每个指数文件的情况：")
    st.dataframe(loaders.diagnose(IDX_CODES), hide_index=True, width="stretch")
    st.info(f"当前本地目录：`{config.DAILY_DIR}`。常见原因：①文件名里没有6位指数代码（如写成「沪深300.csv」，或用 399300 而不是 000300）；"
            "②列名没识别出日期/收盘；③环境变量路径没生效（设置后要在同一个 PowerShell 窗口里启动，或重启终端）。"
            "也可以到「数据」页点「更新全部数据」或 ③，让程序自动下载四大指数日线。")
    st.stop()
_missing_idx = [c for c in IDX_CODES if c not in idx_close.columns]
if _missing_idx:
    st.warning(f"以下指数本地没有日线，相关列显示为空：{', '.join(_missing_idx)}。可到「数据」页点 ③ 自动下载。")
    with st.expander("查看原因"):
        st.dataframe(loaders.diagnose(_missing_idx), hide_index=True, width="stretch")
idx_close = idx_close.reindex(columns=IDX_CODES)

dates = idx_close.dropna(how="all").index
asof = st.sidebar.date_input("统计截止日", dates[-1].date(), min_value=dates[0].date(), max_value=dates[-1].date())
asof = dates[dates <= pd.Timestamp(asof)][-1]
st.caption(f"统计截止：{asof.date()}  ｜  红涨绿跌  ｜  本周=周一起至今，本月=月初起至今（以上一周期最后交易日收盘为基准）")

# ───────────── 预计算：各模块 ─────────────
sec_close = close_panel(tuple(SEC_CODES), S, tail=None, adjust="qfq")
etf_close = close_panel(tuple(ETF_ALL), S)
etf_amount = amount_panel(tuple(ETF_ALL), S)

# 避险
sec_rows, sec_curves = [], {}
bench_close = idx_close[IDX[config.BENCH]["code"]]
bench_r = an.period_returns(bench_close, asof)
for nm, info in config.SECTORS.items():
    codes = [c for c, _ in info["stocks"]]
    b = an.basket_returns(sec_close, codes, asof) if len(sec_close) else {"n": 0}
    sec_rows.append({"板块": nm, "角色": {"defensive": "防御", "optional": "可选", "cyclical": "周期对照"}[info["role"]],
                     "成分(有数据/总数)": f"{b.get('n', 0)}/{len(codes)}",
                     "当日": b.get("day"), "本周": b.get("wtd"), "本月": b.get("mtd"),
                     "当日超额": b.get("day", np.nan) - bench_r["day"] if b.get("n") else np.nan,
                     "本周超额": b.get("wtd", np.nan) - bench_r["wtd"] if b.get("n") else np.nan,
                     "本月超额": b.get("mtd", np.nan) - bench_r["mtd"] if b.get("n") else np.nan,
                     "上涨占比": b.get("breadth")})
sec_df = pd.DataFrame(sec_rows)
def_codes = [c for nm, i in config.SECTORS.items() if i["role"] == "defensive" for c, _ in i["stocks"]]
def_b = an.basket_returns(sec_close, def_codes, asof) if len(sec_close) else {"n": 0}
def_rel = {k: (def_b.get(k, np.nan) - bench_r[k]) if def_b.get("n") else np.nan for k in ("day", "wtd", "mtd")}

# ETF
shares = ds("etf_shares", S)
flow = aum = None
etf_res = {}
if len(shares):
    flow, aum = an.etf_flow_wide(shares, etf_close)
    for nm in IDX:
        etf_res[nm] = an.etf_group_indicators(flow, aum, etf_amount, config.ETF_GROUPS[nm], asof)

# 融资
margin = ds("margin", S)
cons = ds("index_cons", S)
mg, mg_sum = {}, {}
if len(margin) and len(cons):
    for nm, info in IDX.items():
        cc = cons.loc[cons["index_code"] == info["code"], "code"].tolist()
        if cc:
            amt = amount_panel(tuple(cc), S)
            mg[nm] = an.margin_group(margin, cc, amt)
            mg_sum[nm] = an.margin_summary(mg[nm], asof)

# 均线
ma = {nm: an.ma_status(idx_close[info["code"]], asof) for nm, info in IDX.items()}

# 期指
rank, fdaily = ds("cffex_rank", S), ds("cffex_daily", S)
fut = {}
for nm, info in IDX.items():
    v = info["fut"]
    if not len(rank):
        continue
    tot = an.rank_totals(rank, v)
    tot = tot[tot.index <= asof]
    if tot.empty:
        continue
    cls = idx_close[info["code"]]
    etf_f = an.etf_group_flow_series(flow, config.ETF_GROUPS[nm]) if flow is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    dbal = mg[nm]["dbal"] if nm in mg else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    fut[nm] = dict(tot=tot, etf_f=etf_f, dbal=dbal, cls=cls,
                   basis=an.basis_series(fdaily, cls, v) if len(fdaily) else pd.DataFrame())

tabs = st.tabs(["总览", "① 避险板块", "② ETF资金", "③ 期指持仓", "④ 融资融券", "⑤ 均线", "⑥ 跨境资金", "说明与限制"])

# ═════════ 总览 ═════════
with tabs[0]:
    rows = []
    for nm, info in IDX.items():
        r = an.period_returns(idx_close[info["code"]], asof)
        e, m, f = etf_res.get(nm), mg_sum.get(nm, {}), fut.get(nm)
        label, ann, d_short = "—", np.nan, np.nan
        if f:
            hf = an.hedge_frame(f["tot"], f["cls"], info["mult"], f["etf_f"], f["dbal"], config.HEDGE)
            label, d_short = hf["label"].iloc[-1], hf["short_add"].iloc[-1]
            if len(f["basis"]):
                ann = f["basis"]["ann"].dropna().iloc[-1] if f["basis"]["ann"].notna().any() else np.nan
        sig = [("均线", ma[nm]["score"]), ("ETF5日", np.sign(e["d5"]) if e else 0),
               ("避险相对强弱(当周)", -np.sign(def_rel["wtd"]) if def_rel["wtd"] == def_rel["wtd"] else 0),
               ("期指空单", 1 if label.startswith("偏对冲") else (-1 if label.startswith("偏投机") else 0))]
        rows.append({"指数": nm, "当日": r["day"], "本周": r["wtd"], "本月": r["mtd"], "均线": ma[nm]["state"],
                     "ETF近5日净申购(亿)": e["d5"] if e else np.nan, "融资买入占比": m.get("ratio", np.nan),
                     "融资占比分位": m.get("pct", np.nan), "前20空单5日增量(亿)": d_short, "年化基差": ann,
                     "空单性质": label, "偏多信号": sum(1 for _, v in sig if v > 0), "偏空信号": sum(1 for _, v in sig if v < 0)})
    show(pd.DataFrame(rows), pct_cols=["当日", "本周", "本月", "融资买入占比", "融资占比分位", "年化基差"],
         num_cols=["ETF近5日净申购(亿)", "前20空单5日增量(亿)"], color_cols=["当日", "本周", "本月", "ETF近5日净申购(亿)"])
    # ── 数据时效 / 缺失原因：哪些列用的是旧数据、为什么是"—" ──
    _asof = pd.Timestamp(asof)
    _stale = []
    for _lab, _df_, _cols in (("融资融券", margin, "融资买入占比、融资占比分位"), ("ETF份额", shares, "ETF近5日净申购"),
                              ("期指持仓排名", rank, "前20空单5日增量、空单性质"), ("期指日行情", fdaily, "年化基差")):
        if len(_df_) and pd.Timestamp(_df_["date"].max()) < _asof:
            _stale.append(f"**{_lab}** 只到 {pd.Timestamp(_df_['date'].max()).date()}（统计截止日 {_asof.date()}）→ 影响列：{_cols}")
    if _stale:
        st.warning("⏳ 以下数据**不是最新**，对应列用的是更早的数据（融资融券 / ETF份额通常次日早晨才披露，之后的更新会补上）：\n\n- "
                   + "\n- ".join(_stale))
    _why = []
    for _nm, _info in IDX.items():
        _m = mg_sum.get(_nm) or {}
        if "ratio" in _m:
            continue
        if not len(margin):
            _why.append("融资融券明细是空的（「数据」页点 ①）")
            break
        if not len(cons):
            _why.append("没有指数成分股名单（「数据」页点「指数成分股名单」）")
            break
        _cc = cons.loc[cons["index_code"] == _info["code"], "code"].tolist()
        if not _cc:
            _why.append(f"{_nm}：没有成分股名单（「数据」页点「指数成分股名单」）")
            continue
        if not margin["code"].isin(_cc).any():
            _why.append(f"{_nm}：融资融券明细里一只成分股都没匹配上（股票代码格式不一致？）")
            continue
        _amt = amount_panel(tuple(_cc), S)
        _na = _amt.shape[1] if len(_amt) else 0
        if _na == 0:
            _why.append(f"{_nm}：成分股 {len(_cc)} 只，本地一只都没有带成交额的日线，算不出占比（「数据」页点 ⑥ 补日线）")
        else:
            _why.append(f"{_nm}：成分股 {len(_cc)} 只里本地有成交额的 {_na} 只，但和融资数据没有重叠的交易日"
                        f"（成交额到 {pd.Timestamp(_amt.index.max()).date()}，融资到 {pd.Timestamp(margin['date'].max()).date()}）")
    if _why:
        st.info("融资买入占比 / 分位显示为「—」的原因：\n\n- " + "\n- ".join(_why))
    st.markdown(f"**避险防御篮子相对沪深300**：当日 {pct(def_rel['day'])} ｜ 本周 {pct(def_rel['wtd'])} ｜ 本月 {pct(def_rel['mtd'])}"
                f"（防御篮子 = 角色为\"防御\"的板块成分股等权；正值表示防御跑赢大盘）")
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
    why = []
    if not len(shares):
        why.append("**ETF份额**数据集为空 → ETF净申购无法计算（到「数据」页点 ④，或看各步骤日志里 ETF份额 那一行的失败原因）")
    elif all(v is None for v in etf_res.values()):
        have = sorted(set(shares["code"]))
        need = sorted({c for v in config.ETF_GROUPS.values() for c in v})
        miss_px = [c for c in need if c not in etf_close.columns]
        why.append(f"ETF份额有 {len(have)} 只，但算不出净申购：本地缺 ETF 日线 {miss_px or '无'}；"
                   f"份额里不含的配置代码 {[c for c in need if c not in have] or '无'}（净申购=份额变化×收盘价，两者都要有）")
    if not len(rank):
        why.append("**期指持仓排名**数据集为空 → 空单增量/空单性质无法计算（到「数据」页点 ②）")
    elif not fut:
        why.append("期指持仓排名有数据，但截止日之前没有记录（检查「统计截止日」或数据日期）")
    if not len(fdaily):
        why.append("**期指日行情**数据集为空 → 年化基差无法计算（到「数据」页点 ②）")
    for w in why:
        st.warning(w)
    st.caption("信号计数规则：均线多头排列=+1/空头排列=−1；ETF近5日净申购>0=+1/<0=−1；防御本周跑赢沪深300=−1，跑输=+1；"
               "空单性质偏对冲=+1/偏投机=−1。**该计数只是规则汇总，尚未经过回测验证，不能当作交易信号。**")

# ═════════ ① 避险板块 ═════════
with tabs[1]:
    st.subheader("板块当日 / 本周 / 本月涨跌（等权篮子，括号内为相对沪深300超额）")
    show(sec_df, pct_cols=["当日", "本周", "本月", "当日超额", "本周超额", "本月超额", "上涨占比"],
         color_cols=["当日", "本周", "本月", "当日超额", "本周超额", "本月超额"])
    ref_rows = []
    for nm, c in config.REF_ETFS.items():
        if loaders.has_local(c):
            p = loaders.load_price(c, adjust="qfq")["close"]
            r = an.period_returns(p, asof)
            ref_rows.append({"参考ETF": nm, "代码": c, "当日": r["day"], "本周": r["wtd"], "本月": r["mtd"]})
    if ref_rows:
        st.markdown("**参考 ETF（前复权）**")
        show(pd.DataFrame(ref_rows), pct_cols=["当日", "本周", "本月"], color_cols=["当日", "本周", "本月"])
    if len(sec_close):
        win_opts = {"近60日": 60, "近120日": 120, "近1年(250日)": 250, "近3年(750日)": 750,
                    "近5年(1250日)": 1250, "近10年(2500日)": 2500, "全部历史": 0}
        win_label = st.radio("走势窗口", list(win_opts.keys()), index=0, horizontal=True, key="sec_win")
        n = win_opts[win_label]

        b_all = bench_close[bench_close.index <= asof].dropna()
        sec_all = sec_close[sec_close.index <= asof]
        if len(b_all):
            start_dt = b_all.index[-n] if (n > 0 and len(b_all) >= n) else b_all.index[0]
        else:
            start_dt = sec_all.index[-n] if (n > 0 and len(sec_all) >= n) else sec_all.index[0]

        fig = go.Figure()
        b = b_all[b_all.index >= start_dt]
        if len(b):
            fig.add_scatter(x=b.index, y=b / b.iloc[0], name="沪深300", line=dict(color="#555", dash="dash"))

        sec_win_df = sec_all[sec_all.index >= start_dt]
        def_sub = sec_win_df[[c for c in def_codes if c in sec_win_df.columns]].dropna(how="all")
        cur = an.basket_curve(def_sub, def_codes)
        if len(cur):
            fig.add_scatter(x=cur.index, y=cur / cur.iloc[0], name="防御综合(前复权)", line=dict(width=3))
        for nm, info in config.SECTORS.items():
            scols = [c for c, _ in info["stocks"] if c in sec_win_df.columns]
            if not scols:
                continue
            sub_s = sec_win_df[scols].dropna(how="all")
            cv = an.basket_curve(sub_s, scols)
            if len(cv) >= 2:
                fig.add_scatter(x=cv.index, y=cv / cv.iloc[0], name=nm, visible="legendonly")
        fig.update_layout(height=420, margin=dict(t=20), yaxis_title="期初=1（前复权等权）", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        with st.expander("长历史回补（4大指数 + 避险板块前复权因子，扩展超过2019年的曲线）"):
            b_span = f"{b_all.index.min().date()} ~ {b_all.index.max().date()}（{len(b_all)} 日）" if len(b_all) else "无"
            s_valid = sec_all.dropna(how="all")
            s_span = f"{s_valid.index.min().date()} ~ {s_valid.index.max().date()}（{len(s_valid)} 日）" if len(s_valid) else "无"
            st.caption(f"沪深300：{b_span}　｜　避险板块：{s_span}")
            hc1, hc2 = st.columns(2)
            if hc1.button("📥 回补至 2005-01-01（对齐沪深300，约1-2分钟）", width="stretch"):
                run_job("回补指数与板块至2005", lambda p: fetchers.backfill_sector_and_index("20050101", p))
            if hc2.button("📥 回补至 2000-01-01（板块更早历史，约2分钟）", width="stretch"):
                run_job("回补指数与板块至2000", lambda p: fetchers.backfill_sector_and_index("20000101", p))
            if "last_log" in st.session_state and str(st.session_state["last_log"][0]).startswith("回补指数与板块"):
                st.text("\n".join(st.session_state["last_log"][1]))
        for nm, info in config.SECTORS.items():
            codes = [c for c, _ in info["stocks"] if c in sec_close.columns]
            if codes:
                with st.expander(f"{nm} 成分股明细"):
                    show(an.returns_table(sec_close[codes], NAME, asof), pct_cols=["当日", "本周", "本月"],
                         color_cols=["当日", "本周", "本月"])
    else:
        st.info("本地没有任何板块成分股日线。")

# ═════════ ② ETF资金 ═════════
with tabs[2]:
    if not etf_res or all(v is None for v in etf_res.values()):
        st.info("还没有 ETF 份额数据。「数据」页点「更新：ETF份额(上交所)」，或手动导入交易所下载的份额文件。")
    else:
        tot_abs = sum(abs(v["day"]) for v in etf_res.values() if v) or np.nan
        rows = []
        for nm, e in etf_res.items():
            if not e:
                continue
            rows.append({"指数": nm, "ETF数": e["n_etf"], "数据日": e["date"].date(),
                         "①净申购·当日(亿)": e["day"], "①本周(亿)": e["wtd"], "①本月(亿)": e["mtd"],
                         "②规模归一·5日": e["norm_d5"], "②20日": e["norm_d20"], "②本月": e["norm_mtd"],
                         "③累计5日(亿)": e["d5"], "③累计20日(亿)": e["d20"],
                         "④成交额/前20日均": e["amount_x"], "⑤当日占四类合计": e["day"] / tot_abs if tot_abs == tot_abs else np.nan})
        show(pd.DataFrame(rows), pct_cols=["②规模归一·5日", "②20日", "②本月", "⑤当日占四类合计"],
             num_cols=["①净申购·当日(亿)", "①本周(亿)", "①本月(亿)", "③累计5日(亿)", "③累计20日(亿)", "④成交额/前20日均"],
             color_cols=["①净申购·当日(亿)", "①本周(亿)", "①本月(亿)", "③累计5日(亿)", "③累计20日(亿)"])
        n = st.slider("走势窗口(日)", 20, 250, 60, key="etf_n")
        fig = go.Figure()
        for nm in IDX:
            s = an.etf_group_flow_series(flow, config.ETF_GROUPS[nm])
            s = s[s.index <= asof].dropna().tail(n)
            if len(s):
                fig.add_scatter(x=s.index, y=s.cumsum(), name=nm)
        fig.update_layout(height=380, margin=dict(t=20), yaxis_title="区间累计净申购(亿元)")
        st.plotly_chart(fig, width="stretch")
        st.caption("净申购 = 份额变化 × 当日收盘价（市价近似净值）。2026年1月以来宽基ETF大额进出与中央汇金操作有关，"
                   "出现极端值时请当作政策事件看待，不要直接当情绪信号。")

# ═════════ ③ 期指持仓 ═════════
with tabs[3]:
    if not fut:
        st.info("还没有期指持仓排名数据。「数据」页点「更新：期指持仓排名(中金所)」，失败则用中金所官网下载后手动导入。")
    else:
        nm = st.radio("品种", list(fut.keys()), horizontal=True, format_func=lambda k: f"{k} ({IDX[k]['fut']})")
        info, f = IDX[nm], fut[nm]
        tot, N = f["tot"], config.HEDGE["window"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("前20 多单", num(tot["long"].iloc[-1], 0), f"{tot['long'].iloc[-1] - tot['long'].iloc[-2]:+,.0f}")
        c2.metric("前20 空单", num(tot["short"].iloc[-1], 0), f"{tot['short'].iloc[-1] - tot['short'].iloc[-2]:+,.0f}")
        c3.metric("前20 净持仓(多−空)", num(tot["net"].iloc[-1], 0), f"{tot['net'].iloc[-1] - tot['net'].iloc[-2]:+,.0f}")
        b = f["basis"]
        if len(b):
            bb = b[b.index <= asof]
            c4.metric(f"主力{bb['contract'].iloc[-1]} 年化基差", pct(bb["ann"].iloc[-1]) if bb["ann"].notna().iloc[-1] else "—",
                      f"{bb['basis'].iloc[-1]:+.1f}点")
        # 周/月：用前20合计的水平差
        wk = tot["net"].iloc[-1] - tot["net"][tot.index < tot.index[an.period_mask(tot.index, tot.index[-1], 'W')][0]].iloc[-1] \
            if (tot.index < tot.index[an.period_mask(tot.index, tot.index[-1], 'W')][0]).any() else np.nan
        mo = tot["net"].iloc[-1] - tot["net"][tot.index < tot.index[an.period_mask(tot.index, tot.index[-1], 'M')][0]].iloc[-1] \
            if (tot.index < tot.index[an.period_mask(tot.index, tot.index[-1], 'M')][0]).any() else np.nan
        st.caption(f"净持仓变化：当日 {tot['net'].diff().iloc[-1]:+,.0f} 手 ｜ 本周 {num(wk, 0)} 手 ｜ 本月 {num(mo, 0)} 手"
                   "（前20榜单成员会变，合计的变化含成员进出的噪声）")

        st.markdown("#### 空单性质判断（你的规则：空单增量 ÷ 现货增量）")
        o1, o2, o3 = st.columns(3)
        use_etf = o1.checkbox("现货含 ETF 净申购", True)
        use_mar = o2.checkbox("现货含融资余额变化", nm in mg)
        cfg = dict(config.HEDGE)
        cfg["window"] = o3.number_input("窗口(日)", 2, 20, cfg["window"], key="hw")
        hf = an.hedge_frame(tot, f["cls"], info["mult"], f["etf_f"], f["dbal"], cfg, use_etf, use_mar)
        last = hf.iloc[-1]
        st.markdown(f"**{asof.date()}：{last['label']}**　｜　近{cfg['window']}日 前20空单增量 {num(last['short_add'])} 亿元，"
                    f"现货增量 {num(last.get('spot_add'))} 亿元（ETF {num(last.get('spot_etf'))} / 融资 {num(last.get('spot_margin'))}），"
                    f"比值 {num(last['ratio'], 2)}")
        st.warning("这个判断是启发式规则，不是确定结论：①期货公司席位的空单混有投机、套保、套利客户，无法拆分；"
                   "②套保可以对冲**已有**的股票持仓，空单增加不要求现货同步增加；③中性策略多持中小盘个股、对冲用IC/IM，"
                   "不会体现在沪深300 ETF份额里；④融资买入主要是杠杆资金，不等于机构加仓；⑤北向每日数据已停，缺少这一条交叉验证。"
                   "下面的历史检验用来看这个标签到底有没有预测力。")
        H = st.select_slider("检验：标签之后看几日涨跌", [3, 5, 10, 20], 5)
        stats = an.label_stats(hf[hf.index <= asof], f["cls"], H)
        if len(stats):
            st.dataframe(stats.style.format({stats.columns[1]: "{:+.2%}", "上涨占比": "{:.0%}", "按假设方向命中率": "{:.0%}"}, na_rep="—"),
                         width="stretch")
            st.caption(f"样本只有本地缓存里的 {len(hf)} 个交易日，且相邻日标签高度重叠，有效样本远小于样本数；命中率接近50%说明规则没有区分度。")

        cA, cB = st.columns(2)
        with cA:
            fig = go.Figure()
            t = tot[tot.index <= asof].tail(120)
            fig.add_scatter(x=t.index, y=t["long"], name="前20多单")
            fig.add_scatter(x=t.index, y=t["short"], name="前20空单")
            fig.add_scatter(x=t.index, y=t["net"], name="净持仓", line=dict(dash="dot"))
            fig.update_layout(height=340, margin=dict(t=30), title="前20会员持仓(手，该品种所有合约合计)")
            st.plotly_chart(fig, width="stretch")
        with cB:
            if len(b):
                bb = b[b.index <= asof].tail(120)
                fig = go.Figure()
                fig.add_scatter(x=bb.index, y=bb["ann"], name="年化基差")
                fig.add_hline(y=0, line_dash="dot")
                fig.update_layout(height=340, margin=dict(t=30), title="主力合约年化基差(未扣分红，负=贴水)", yaxis_tickformat=".1%")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("没有期指日行情，无法算基差。到「数据」页点 ②。")
        if len(b) > 2:
            bb = b[b.index <= asof]
            ret = f["cls"].pct_change().reindex(bb.index)
            oi_all = fdaily[(fdaily["variety"] == info["fut"]) & (fdaily["date"] <= asof)].groupby("date")["oi"].sum().diff()
            doi = oi_all.reindex(bb.index)
            st.markdown(f"**持仓量×价格**：{an.oi_price_quadrant(ret.iloc[-1], doi.iloc[-1])}"
                        f"（该品种全部合约持仓量变化 {num(doi.iloc[-1], 0)} 手，指数当日 {pct(ret.iloc[-1])}）")
        mt = an.member_table(rank, asof, info["fut"])
        if len(mt):
            st.markdown("#### 席位明细（该品种所有合约；只含进入前20榜的部分，未上榜记 0）")
            mt = mt.reset_index()
            mt["重点席位"] = mt["席位"].isin(config.WATCH_MEMBERS)
            show(mt, num_cols=["多单", "空单", "净持仓", "净持仓增减"], color_cols=["净持仓增减"])
            st.caption("期货公司席位里含大量对冲盘（如券商系席位），净空不等于看空；更有用的是其**变化**，并用上面的历史检验验证。")

# ═════════ ④ 融资融券 ═════════
with tabs[4]:
    if not mg_sum:
        st.info("需要同时有：融资融券明细（到「数据」页更新）+ 指数成分股（到「数据」页更新）+ 本地个股日线（用来取成交额）。")
    else:
        rows = []
        for nm, m in mg_sum.items():
            if m:
                rows.append({"指数": nm, "数据日": m["date"].date(), "融资买入占比·当日": m.get("ratio"),
                             "5日均": m.get("ratio5"), "20日均": m.get("ratio20"), "近120日分位": m.get("pct"),
                             "融资余额(亿)": m["bal"], "余额变化·当日(亿)": m["dbal_day"], "本周(亿)": m["dbal_wtd"],
                             "本月(亿)": m["dbal_mtd"], "成交额覆盖率": m["coverage"]})
        df = pd.DataFrame(rows)
        show(df, pct_cols=["融资买入占比·当日", "5日均", "20日均", "近120日分位", "成交额覆盖率"],
             num_cols=["融资余额(亿)", "余额变化·当日(亿)", "本周(亿)", "本月(亿)"],
             color_cols=["余额变化·当日(亿)", "本周(亿)", "本月(亿)"])
        bad = df[(df["融资买入占比·当日"] > 1) | (df["融资买入占比·当日"] < 0.001)]
        if len(bad):
            st.error("融资买入占比数量级异常：很可能是本地日线成交额单位不是“元”。请设置环境变量 ASHARE_AMOUNT_UNIT（万元=10000）。")
        low = df[df["成交额覆盖率"] < 0.9]
        if len(low):
            st.warning("部分指数成分股在本地缺日线，占比只用了有数据的那部分个股；覆盖率低于90%时请补全本地日线。")
        n = st.slider("走势窗口(日)", 20, 250, 90, key="mg_n")
        fig = go.Figure()
        for nm, g in mg.items():
            s = g["ratio"][g.index <= asof].dropna().tail(n)
            fig.add_scatter(x=s.index, y=s, name=nm)
        fig.update_layout(height=380, margin=dict(t=20), yaxis_tickformat=".1%", yaxis_title="融资买入额/成交额")
        st.plotly_chart(fig, width="stretch")
        st.caption("口径：成分股融资买入额合计 ÷ 成分股成交额合计，用的是**当前**成分股（历史成分变动未还原）。"
                   "数据来自沪深交易所融资融券明细，不是东方财富；东财数据中心页面也有个股明细，但页面没有稳定的下载接口。")

# ═════════ ⑤ 均线 ═════════
with tabs[5]:
    rows = [{"指数": nm, "收盘": m["close"], "MA5": m["ma5"], "MA20": m["ma20"], "MA60": m["ma60"],
             "距MA20": m["dist20"], "MA20近5日斜率": m["slope20"], "状态": m["state"], "5/20交叉": m["cross"],
             "近20日在MA60上占比": m["above60_20d"], "判定依据": m["why"]} for nm, m in ma.items()]
    show(pd.DataFrame(rows), pct_cols=["距MA20", "MA20近5日斜率", "近20日在MA60上占比"],
         num_cols=["收盘", "MA5", "MA20", "MA60"], color_cols=["距MA20", "MA20近5日斜率"])
    st.caption("状态：多头排列 = 收盘>MA20>MA60 且 MA5>MA20；空头排列相反；其余按收盘相对 MA60 分偏强/偏弱。")
    n = st.slider("走势窗口(日)", 60, 400, 180, key="ma_n")
    cols = st.columns(2)
    for i, (nm, info) in enumerate(IDX.items()):
        f_ = an.ma_frame(idx_close[info["code"]]); f_ = f_[f_.index <= asof].tail(n)
        fig = go.Figure()
        for c, lab in (("close", "收盘"), ("ma5", "MA5"), ("ma20", "MA20"), ("ma60", "MA60")):
            fig.add_scatter(x=f_.index, y=f_[c], name=lab, line=dict(width=2 if c == "close" else 1.2))
        fig.update_layout(height=320, margin=dict(t=30, b=10), title=nm)
        cols[i % 2].plotly_chart(fig, width="stretch")

# ═════════ ⑥ 跨境资金 ═════════
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

# ═════════ 说明 ═════════
with tabs[7]:
    st.markdown("""
**数据来源与可得性**
- 本地日线：指数、ETF、个股（你自己下载）。ETF 日线需含成交额。
- 期指持仓排名、期指日行情：中金所；融资融券明细：沪/深交易所；ETF份额：上交所（深市 159xxx 的历史份额需手动导入）。
- **北向每日持股/净流入已于 2024-08-19 起停止披露**，本页不使用；"外资代理指标"尚未做进来。

**尚未实测的部分**
- 所有联网下载（中金所直连、akshare 接口）和中金所文件的列解析，开发时无法访问这些网站，**没有实测**。
  首次使用请按顺序点各"更新"按钮，看「数据」页的"数据健康检查"，失败就用官网文件手动导入；解析报错会把识别到的列名打印出来，发给我即可修。
- 板块成分股代码凭记忆整理，请自行核对（config.py 里可改）。

**指标口径**
- 当日/本周/本月：以上一周期最后一个交易日收盘为基准；篮子为成分股等权。
- ETF 净申购 = 份额变化 × 收盘价；规模归一 = 净申购 ÷ 期初规模。
- 期指持仓为"前20会员、该品种所有合约合计"；基差为主力合约（持仓量最大）相对指数，未扣分红。
""")
