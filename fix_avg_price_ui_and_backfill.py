# -*- coding: utf-8 -*-
"""修复平均股价页面的按钮显示、60/120日回补与曲线截取逻辑。

用法（在项目根目录运行）：
    python fix_avg_price_ui_and_backfill.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

AVG_PRICE_VIEW = r'''"""页面：防御板块均价 vs 全A平均股价（除权修正链式回溯）。"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import analytics as an
import config
import fetchers
from common import NAME, ds, num, render_freshness, run_job, show, stamp

st.title("防御板块均价 vs 全A平均股价")
render_freshness()

S = stamp()
mkt_raw = ds("market_avg", S)
sec_raw = ds("sector_px", S)

mkt_days = int(mkt_raw["date"].nunique()) if len(mkt_raw) else 0
sec_days = int(sec_raw["date"].nunique()) if len(sec_raw) else 0

# ───────────── 数据覆盖状态与下载按钮（始终可见，不再折叠隐藏）─────────────
with st.container(border=True):
    m_span = f"{mkt_raw['date'].min().date()} ~ {mkt_raw['date'].max().date()}" if mkt_days else "暂无"
    s_span = f"{sec_raw['date'].min().date()} ~ {sec_raw['date'].max().date()}" if sec_days else "暂无"
    st.markdown(
        f"**数据覆盖**：全A平均股价 **{mkt_days}** 个交易日（{m_span}）　｜　"
        f"板块复权日线 **{sec_days}** 个交易日（{s_span}）"
    )
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("🛡️ 1. 拉防御板块至2019", width="stretch",
                 help="只拉取 config.SECTORS 约45只股票自 2019 起的 close/pre_close（约 1 分钟）"):
        run_job("板块复权日线", lambda p: fetchers.download_sector_px(progress=p))
    if b2.button("⚡ 2a. 回补全A近120日", width="stretch",
                 help="补齐最近 120 个交易日的全A均价数据（约 3 分钟），可立刻查看 60日 / 120日 对比曲线"):
        run_job("⑦ 全A平均股价(近120日)", lambda p: fetchers.download_market_avg(120, full_history=False, progress=p))
    if b3.button("📚 2b. 回补全A至2019", width="stretch",
                 help="由新到旧回补自 2019-01-01 起的全部交易日（每20天存盘，可随时中断并在下次继续）"):
        run_job("⑦ 全A平均股价(2019起)", lambda p: fetchers.download_market_avg(full_history=True, progress=p))
    if b4.button("🔄 增量更新至最新", width="stretch",
                 help="补齐最近 60 日内缺失的最新交易日"):
        run_job("⑦ 增量更新平均股价", lambda p: fetchers.download_market_avg(60, full_history=False, progress=p))

    if "last_log" in st.session_state:
        t_, lg_ = st.session_state["last_log"]
        if any(k in t_ for k in ("平均股价", "板块复权")):
            with st.expander(f"最近一次更新日志：{t_}", expanded=True):
                st.text("\n".join(lg_))

if mkt_raw.empty and sec_raw.empty:
    st.info("暂无数据，请先点上方「🛡️ 1. 拉防御板块至2019」或「⚡ 2a. 回补全A近120日」。")
    st.stop()

# ───────────── 统计截止日与计算参数 ─────────────
def_codes = [c for s in config.SECTORS.values() if s["role"] == "defensive" for c, _ in s["stocks"]]
all_dates = sorted(set(mkt_raw["date"]) | set(sec_raw["date"])) if (len(mkt_raw) and len(sec_raw)) else (
    sorted(mkt_raw["date"]) if len(mkt_raw) else sorted(sec_raw["date"])
)
dates = pd.DatetimeIndex(all_dates)
asof = st.sidebar.date_input("统计截止日", dates[-1].date(), min_value=dates[0].date(), max_value=dates[-1].date(),
                             key="avg_asof")
asof = dates[dates <= pd.Timestamp(asof)][-1]

method_map = {
    "除权修正算术均价（同花顺830000口径）": "adj_avg",
    "等权复权均价（消除高价股权重）": "eq_avg",
    "原始算术均价（未除权，对照用）": "raw_avg",
}
method_label = st.sidebar.radio("均价口径", list(method_map.keys()), index=0)
col = method_map[method_label]

win_opts = {"近60日": 60, "近120日": 120, "近1年(250日)": 250, "近3年(750日)": 750, "全部(2019起)": 0}
win_label = st.sidebar.radio("展示区间", list(win_opts.keys()), index=0)
win_n = win_opts[win_label]

chart_mode = st.sidebar.radio("主图模式", ["双轴实际均价(元)", "归一化走势(起点=1)", "同轴实际均价(元)"], index=0)

mkt_df = an.market_avg_frame(mkt_raw, asof)
def_df = an.basket_avg_frame(sec_raw, def_codes, asof)

# 按统一截止日与窗口起点切分（不让其中一方天数少而截断另一方的历史曲线）
avail_dates = dates[dates <= asof]
start_dt = avail_dates[-win_n] if (win_n > 0 and len(avail_dates) >= win_n) else avail_dates[0]
mkt_view = mkt_df[mkt_df.index >= start_dt] if len(mkt_df) else mkt_df
def_view = def_df[def_df.index >= start_dt] if len(def_df) else def_df

if mkt_days < 10 or sec_days < 10:
    st.warning(
        f"当前缓存交易日较少（全A {mkt_days} 天 / 板块 {sec_days} 天），尚不足以画出完整趋势曲线；"
        "请点击上方「🛡️ 1. 拉防御板块至2019」和「⚡ 2a. 回补全A近120日」（或「📚 2b. 回补全A至2019」）。"
    )
elif len(mkt_view) < len(def_view) * 0.8:
    st.info(
        f"当前展示区间内：防御板块有 {len(def_view)} 个交易日，全A目前只有 {len(mkt_view)} 个交易日。"
        "点上方「⚡ 2a. 回补全A近120日」或「📚 2b. 回补全A至2019」即可补齐全A曲线。"
    )

# ───────────── 汇总指标与区间对比表 ─────────────
rows = []
for name, frame in (("防御篮子平均股价", def_view), ("全A平均股价", mkt_view)):
    if frame is None or frame.empty:
        continue
    s = frame[col].dropna()
    if s.empty:
        continue
    r = an.period_returns(s, asof)
    win_ret = (s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 and s.iloc[0] > 0 else float("nan")
    rows.append({
        "对象": name,
        "最新均价(元)": float(s.iloc[-1]),
        "活跃股数": int(frame["n"].iloc[-1]),
        "样本天数": len(s),
        "当日": r["day"],
        "本周": r["wtd"],
        "本月": r["mtd"],
        f"{win_label}累计": win_ret,
    })

if rows:
    show(
        pd.DataFrame(rows),
        pct_cols=["当日", "本周", "本月", f"{win_label}累计"],
        num_cols=["最新均价(元)"],
        color_cols=["当日", "本周", "本月", f"{win_label}累计"],
    )

# ───────────── 曲线 1：防御篮子均价 vs 全A均价 ─────────────
st.subheader(f"曲线对比：{method_label}")
s_def = def_view[col].dropna() if len(def_view) else pd.Series(dtype=float)
s_mkt = mkt_view[col].dropna() if len(mkt_view) else pd.Series(dtype=float)

if len(s_def) or len(s_mkt):
    mode_def = "lines+markers" if len(s_def) <= 10 else "lines"
    mode_mkt = "lines+markers" if len(s_mkt) <= 10 else "lines"

    if chart_mode == "双轴实际均价(元)" and len(s_def) and len(s_mkt):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=s_def.index, y=s_def, name="防御篮子平均股价(左轴)", mode=mode_def,
                                 line=dict(color="#d62728", width=2.2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=s_mkt.index, y=s_mkt, name="全A平均股价(右轴)", mode=mode_mkt,
                                 line=dict(color="#1f77b4", width=2.0)), secondary_y=True)
        fig.update_yaxes(title_text="防御篮子均价 (元)", secondary_y=False)
        fig.update_yaxes(title_text="全A均价 (元)", secondary_y=True)
    elif chart_mode == "归一化走势(起点=1)" and len(s_def) and len(s_mkt):
        fig = go.Figure()
        common = s_def.index.intersection(s_mkt.index)
        base_d = s_def.loc[common[0]] if len(common) >= 2 else s_def.iloc[0]
        base_m = s_mkt.loc[common[0]] if len(common) >= 2 else s_mkt.iloc[0]
        fig.add_scatter(x=s_def.index, y=s_def / base_d, name="防御篮子(起点=1)", mode=mode_def,
                        line=dict(color="#d62728", width=2.2))
        fig.add_scatter(x=s_mkt.index, y=s_mkt / base_m, name="全A(起点=1)", mode=mode_mkt,
                        line=dict(color="#1f77b4", width=2.0))
        fig.update_yaxes(title_text="区间净值 (重叠起点=1)")
    else:
        fig = go.Figure()
        if len(s_def):
            fig.add_scatter(x=s_def.index, y=s_def, name="防御篮子平均股价(元)", mode=mode_def,
                            line=dict(color="#d62728", width=2.2))
        if len(s_mkt):
            fig.add_scatter(x=s_mkt.index, y=s_mkt, name="全A平均股价(元)", mode=mode_mkt,
                            line=dict(color="#1f77b4", width=2.0))
        fig.update_yaxes(title_text="平均股价 (元)")

    fig.update_layout(height=440, margin=dict(t=25, b=20), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")

    # ───────────── 曲线 2：相对比值（防御均价 ÷ 全A均价）─────────────
    common = s_def.index.intersection(s_mkt.index)
    if len(common) >= 2:
        ratio = (s_def.loc[common] / s_mkt.loc[common]).dropna()
        st.subheader("相对强弱：防御篮子均价 ÷ 全A均价")
        fig2 = go.Figure()
        fig2.add_scatter(x=ratio.index, y=ratio, name="防御均价 / 全A均价",
                         mode="lines+markers" if len(ratio) <= 10 else "lines",
                         line=dict(color="#9467bd", width=2))
        if len(ratio) >= 20:
            fig2.add_scatter(x=ratio.index, y=ratio.rolling(20).mean(), name="20日均线",
                             line=dict(color="#7f7f7f", dash="dash", width=1.4))
        fig2.update_layout(height=300, margin=dict(t=20, b=20), hovermode="x unified",
                           yaxis_title="比值 (上升=防御相对跑赢)")
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            f"当前比值 **{num(ratio.iloc[-1], 3)}**（区间最高 {num(ratio.max(), 3)} / 最低 {num(ratio.min(), 3)}）。"
            "比值上升说明防御板块相对全市场平均水平走强（偏避险），下降说明市场风险偏好回升。"
        )

# ───────────── 分板块对比 & 成分股明细 ─────────────
if len(sec_raw):
    with st.expander("各防御子板块均价对比（同口径）"):
        fig_s = go.Figure()
        sec_rows = []
        for nm, info in config.SECTORS.items():
            if info["role"] != "defensive":
                continue
            codes = [c for c, _ in info["stocks"]]
            sub = an.basket_avg_frame(sec_raw, codes, asof)
            if sub.empty:
                continue
            sub = sub[sub.index >= start_dt]
            s = sub[col].dropna()
            if s.empty:
                continue
            r = an.period_returns(s, asof)
            sec_rows.append({"板块": nm, "最新均价(元)": float(s.iloc[-1]), "当日": r["day"], "本周": r["wtd"], "本月": r["mtd"]})
            if len(s) >= 2 and s.iloc[0] > 0:
                fig_s.add_scatter(x=s.index, y=s / s.iloc[0], name=nm)
        if sec_rows:
            show(pd.DataFrame(sec_rows), pct_cols=["当日", "本周", "本月"], num_cols=["最新均价(元)"],
                 color_cols=["当日", "本周", "本月"])
            fig_s.update_layout(height=340, margin=dict(t=20), yaxis_title="区间净值(起点=1)")
            st.plotly_chart(fig_s, width="stretch")

    with st.expander("防御篮子成分股列表"):
        latest_sec = sec_raw[sec_raw["date"] <= asof].sort_values("date").groupby("code").last()
        items = []
        for nm, info in config.SECTORS.items():
            if info["role"] != "defensive":
                continue
            for c, stock_nm in info["stocks"]:
                row = latest_sec.loc[c] if c in latest_sec.index else None
                items.append({
                    "板块": nm, "代码": c, "名称": NAME.get(c, stock_nm),
                    "最新收盘": float(row["close"]) if row is not None else float("nan"),
                    "最新日": row["date"].date() if row is not None else None,
                })
        show(pd.DataFrame(items), num_cols=["最新收盘"])

st.caption(
    "口径说明：① 每日先剔除停牌股及上市首日新股；② 用 Tushare 每日收盘价 `close` 与除权除息参考价 `pre_close` 计算"
    "全市场日度价格变化 `Σclose / Σpre_close`，再以截止日真实算术均价 `sum(close)/count` 为锚链式回溯，"
    "从而在不引入前/后复权跨期失真的前提下消除除权跳空；③ 防御篮子采用完全相同的公式。"
)
'''

OLD_BLOCK = '''    if start_date is None:
        if len(old):
            cal = trade_dates(old["date"].max() - pd.Timedelta(days=30), latest)
            cal = cal[-11:] if len(cal) > 10 else cal
        else:
            cal = trade_dates(pd.Timestamp(latest) - pd.Timedelta(days=int(backfill_days * 1.7) + 15), latest)
            cal = cal[-backfill_days:] if len(cal) > backfill_days else cal
        todo = [d for d in cal if d.date() not in have]
    else:
        cal = trade_dates(start_date, latest)
        todo = list(reversed([d for d in cal if d.date() not in have]))'''

NEW_BLOCK = '''    if start_date is None:
        cal = trade_dates(pd.Timestamp(latest) - pd.Timedelta(days=int(backfill_days * 1.7) + 20), latest)
        cal = cal[-backfill_days:] if len(cal) > backfill_days else cal
        if len(old):
            recent_cal = trade_dates(old["date"].max() - pd.Timedelta(days=30), latest)
            cal = sorted(set(cal) | set(recent_cal[-11:]))
        todo = list(reversed([d for d in cal if d.date() not in have]))
    else:
        cal = trade_dates(start_date, latest)
        todo = list(reversed([d for d in cal if d.date() not in have]))'''


def main():
    ts_path = ROOT / "tushare_src.py"
    avg_path = ROOT / "views" / "avg_price.py"
    if not ts_path.exists() or not avg_path.exists():
        sys.exit("[中止] 请将脚本放在项目根目录后运行。")

    ts_src = ts_path.read_text(encoding="utf-8-sig")
    if NEW_BLOCK in ts_src:
        print("[跳过] tushare_src.py 已经是最新版")
    else:
        if OLD_BLOCK not in ts_src:
            sys.exit("[中止] tushare_src.py 中未找到预期的 update_market_avg 代码块。")
        ts_new = ts_src.replace(OLD_BLOCK, NEW_BLOCK, 1)
        compile(ts_new, str(ts_path), "exec")
        ts_path.write_text(ts_new, encoding="utf-8")
        print("[更新] tushare_src.py（修复有少量缓存时无法补齐 60/120 日的问题）")

    compile(AVG_PRICE_VIEW, str(avg_path), "exec")
    avg_path.write_text(AVG_PRICE_VIEW, encoding="utf-8")
    print("[更新] views/avg_price.py（按钮直接置顶显示 + 不再截断曲线 + 10天以内自动加点显示）")
    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("打开「⚖️ 平均股价对比」页后，顶部的 4 个按钮会直接显示在卡片里：")
    print("  1) 先点「🛡️ 1. 拉防御板块至2019」（约 1 分钟）")
    print("  2) 再点「⚡ 2a. 回补全A近120日」（约 3 分钟）即可立刻看 60日/120日 两条曲线对比")
    print("  3) 需要看 2019 年起的全A曲线时，点「📚 2b. 回补全A至2019」（可随时中断并在下次续传）")


if __name__ == "__main__":
    main()