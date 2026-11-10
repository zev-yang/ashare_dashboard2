"""页面一：指数分析。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import config
import fetchers
import loaders
from common import (ETF_ALL, IDX, IDX_CODES, NAME, SEC_CODES, amount_panel, close_panel, ds, num, pct,
                    render_freshness, show, stamp)

st.title("A股短期趋势监控")
if config.DEMO:
    st.warning("演示模式：全部为随机模拟数据，不代表真实行情。关闭方法：在 PowerShell 运行 "
               "`Remove-Item Env:ASHARE_DEMO`（或新开一个终端窗口），再重新 `streamlit run app.py`。")
render_freshness()

S = stamp()

# ───────────── 指数日线 & as-of ─────────────
idx_close = close_panel(tuple(IDX_CODES), S)
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
sec_close = close_panel(tuple(SEC_CODES), S)
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

tabs = st.tabs(["总览", "① 避险板块", "② ETF资金", "③ 期指持仓", "④ 融资融券", "⑤ 均线", "说明与限制"])

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
            p = loaders.load_price(c)["close"]
            r = an.period_returns(p, asof)
            ref_rows.append({"参考ETF": nm, "代码": c, "当日": r["day"], "本周": r["wtd"], "本月": r["mtd"]})
    if ref_rows:
        st.markdown("**参考 ETF**")
        show(pd.DataFrame(ref_rows), pct_cols=["当日", "本周", "本月"], color_cols=["当日", "本周", "本月"])
    if len(sec_close):
        n = st.slider("走势窗口(日)", 20, 250, 60, key="sec_n")
        fig = go.Figure()
        b = bench_close[bench_close.index <= asof].tail(n)
        fig.add_scatter(x=b.index, y=b / b.iloc[0], name="沪深300", line=dict(color="#555", dash="dash"))
        cur = an.basket_curve(sec_close[sec_close.index <= asof].tail(n), def_codes)
        if len(cur):
            fig.add_scatter(x=cur.index, y=cur / cur.iloc[0], name="防御综合", line=dict(width=3))
        for nm, info in config.SECTORS.items():
            cv = an.basket_curve(sec_close[sec_close.index <= asof].tail(n), [c for c, _ in info["stocks"]])
            if len(cv):
                fig.add_scatter(x=cv.index, y=cv / cv.iloc[0], name=nm, visible="legendonly")
        fig.update_layout(height=420, margin=dict(t=20), yaxis_title="期初=1")
        st.plotly_chart(fig, width="stretch")
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

# ═════════ 说明 ═════════
with tabs[6]:
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
