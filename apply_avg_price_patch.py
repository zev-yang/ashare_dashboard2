# -*- coding: utf-8 -*-
"""一键添加「防御板块均价 vs 全A平均股价（除权修正链式回溯）」页面与数据管道。

用法（在项目根目录执行）：
    python apply_avg_price_patch.py

安全性：
  1. 先校验所有锚点是否匹配，任意一处不匹配立即中止、不改任何文件；
  2. 修改前备份为 *.bak_before_avg_price；
  3. 写入后逐个文件做语法编译自检；支持重复运行（已打补丁的文件会自动跳过）。
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ANALYTICS_ADD = r'''

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
'''

FETCHERS_ADD_FUNCS = r'''
def download_market_avg(backfill_days: int = 60, full_history: bool = False, progress=None) -> list:
    """⑦ 全A平均股价日度汇总 + 板块成分股 close/pre_close（基于 Tushare daily 按日全市场）。"""
    if not tsrc.available():
        return ["【⑦ 全A平均股价】未配置 Tushare token（免费源不支持按日拉取全市场历史，请在「数据」页配置 token）"]
    if getattr(config, "SOURCE", "auto") == "free":
        return ["【⑦ 全A平均股价】当前数据源设为「仅免费源」，已跳过（需要 Tushare）"]
    try:
        return tsrc.update_market_avg(
            start_date=config.PX_START if full_history else None,
            backfill_days=backfill_days,
            progress=progress,
        )
    except Exception as e:  # noqa
        return [f"【⑦ 全A平均股价】失败：{explain(e)}"]


def download_sector_px(progress=None) -> list:
    """仅补齐 config.SECTORS 成分股自 PX_START 起的 close/pre_close（约45次调用，1分钟内完成）。"""
    if not tsrc.available():
        return ["【板块复权日线】未配置 Tushare token"]
    try:
        return tsrc.update_sector_px_only(progress=progress)
    except Exception as e:  # noqa
        return [f"【板块复权日线】失败：{explain(e)}"]


'''

TUSHARE_ADD_FUNCS = r'''# ───────────── 全A平均股价 & 板块复权基准（除权修正用）─────────────
def _sector_codes() -> set:
    return {c for s in config.SECTORS.values() for c, _ in s["stocks"]}


def _ipo_list_dates() -> dict:
    """缓存全市场上市日期（含已退市），用于剔除新股上市首日（发行价→首日收盘的虚高跳涨）。"""
    p = config.CACHE_DIR / "stock_list_dates.csv"
    if p.exists() and (time.time() - p.stat().st_mtime) < 7 * 86400:
        df = pd.read_csv(p, dtype={"ts_code": str, "list_date": str})
        return dict(zip(df["ts_code"], df["list_date"]))
    frames = []
    for st_ in ("L", "D", "P"):
        try:
            df = call("stock_basic", exchange="", list_status=st_, fields="ts_code,list_date")
            if len(df):
                frames.append(df)
        except Exception:  # noqa
            continue
    if not frames:
        return {}
    allf = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
    allf.to_csv(p, index=False)
    return dict(zip(allf["ts_code"], allf["list_date"]))


def _summarize_day_for_avg(df: pd.DataFrame, d_ymd: str, ipo_map: dict, sec_codes: set):
    """从单日全市场 daily 表提取：全A汇总行 + 板块个股 (date, code, close, pre_close)。"""
    if df is None or df.empty:
        return None, pd.DataFrame()
    d = df.copy()
    d["code"] = d["ts_code"].astype(str).str[:6]
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d["pre_close"] = pd.to_numeric(d["pre_close"], errors="coerce")
    d = d[(d["close"] > 0) & (d["pre_close"] > 0)].copy()
    if ipo_map:
        d = d[d["ts_code"].map(ipo_map).fillna("") != d_ymd]  # 剔除上市首日
    if d.empty:
        return None, pd.DataFrame()
    dt_val = pd.to_datetime(d_ymd)
    ret = d["close"] / d["pre_close"] - 1.0
    mkt_row = {
        "date": dt_val,
        "n": int(len(d)),
        "raw_avg": float(d["close"].mean()),
        "sum_close": float(d["close"].sum()),
        "sum_pre": float(d["pre_close"].sum()),
        "eq_ret": float(ret.mean()),
    }
    sec = d[d["code"].isin(sec_codes)]
    sec_df = pd.DataFrame({
        "date": dt_val,
        "code": sec["code"],
        "close": sec["close"],
        "pre_close": sec["pre_close"],
    }) if len(sec) else pd.DataFrame()
    return mkt_row, sec_df


def record_avg_from_daily(day_frames: list):
    """供 ⑥ 本地个股日线顺便写入全A平均股价缓存（零额外 API 调用）。"""
    import fetchers
    if not day_frames:
        return
    ipo_map, sec_codes = _ipo_list_dates(), _sector_codes()
    mkt_rows, sec_parts = [], []
    for d_ymd, df in day_frames:
        mr, sd = _summarize_day_for_avg(df, d_ymd, ipo_map, sec_codes)
        if mr is not None:
            mkt_rows.append(mr)
        if len(sd):
            sec_parts.append(sd)
    if mkt_rows:
        fetchers.append_ds("market_avg", pd.DataFrame(mkt_rows))
    if sec_parts:
        fetchers.append_ds("sector_px", pd.concat(sec_parts, ignore_index=True))


def update_market_avg(start_date=None, backfill_days: int = 60, progress=None, flush_every: int = 20) -> list:
    """按交易日从 Tushare daily 拉取全市场，汇总保存至 market_avg.csv 与 sector_px.csv。
    start_date=None 时仅做增量（已有则从最后10个交易日起补；为空则回补最近 backfill_days 日）。
    start_date 指定（如 PX_START）时，按「由新到旧」回补缺失历史，每20天落盘一次，可随时中断续传。"""
    import fetchers
    latest = latest_ready_date()
    old = fetchers.read_ds("market_avg")
    have = set(old["date"].dt.date) if len(old) else set()
    if start_date is None:
        if len(old):
            cal = trade_dates(old["date"].max() - pd.Timedelta(days=30), latest)
            cal = cal[-11:] if len(cal) > 10 else cal
        else:
            cal = trade_dates(pd.Timestamp(latest) - pd.Timedelta(days=int(backfill_days * 1.7) + 15), latest)
            cal = cal[-backfill_days:] if len(cal) > backfill_days else cal
        todo = [d for d in cal if d.date() not in have]
    else:
        cal = trade_dates(start_date, latest)
        todo = list(reversed([d for d in cal if d.date() not in have]))
    log = [f"【⑦ 全A平均股价】已缓存 {len(have)} 天；本次待补 {len(todo)} 个交易日（最新入库 {latest}）"]
    if not todo:
        return log
    ipo_map, sec_codes = _ipo_list_dates(), _sector_codes()
    mkt_buf, sec_buf, ok, fail = [], [], 0, 0
    for i, d in enumerate(todo, 1):
        d_ymd = _ymd(d)
        try:
            df = call("daily", trade_date=d_ymd)
            mr, sd = _summarize_day_for_avg(df, d_ymd, ipo_map, sec_codes)
            if mr is not None:
                mkt_buf.append(mr)
                ok += 1
            if len(sd):
                sec_buf.append(sd)
        except Exception as e:  # noqa
            fail += 1
            if fail <= 3:
                log.append(f"  {d.date()}: {e}")
            if "每天" in str(e) or "每小时" in str(e):
                break
        if len(mkt_buf) >= flush_every or i == len(todo):
            if mkt_buf:
                fetchers.append_ds("market_avg", pd.DataFrame(mkt_buf))
                mkt_buf = []
            if sec_buf:
                fetchers.append_ds("sector_px", pd.concat(sec_buf, ignore_index=True))
                sec_buf = []
        if progress:
            progress(i / len(todo), f"全A平均股价 {d.date()} ({i}/{len(todo)})")
    log.append(f"  完成：新增 {ok} 天，失败 {fail} 天")
    return log


def update_sector_px_only(start_date=None, progress=None) -> list:
    """仅补齐配置板块成分股的历史 close/pre_close（按股票逐只拉，约45次调用），方便快速查看防御板块历史曲线。"""
    import fetchers
    start_date = start_date or config.PX_START
    codes = sorted(_sector_codes())
    old = fetchers.read_ds("sector_px")
    latest_map = old.groupby("code")["date"].max().to_dict() if len(old) else {}
    end_d = dt.date.today()
    parts, ok = [], 0
    for i, c in enumerate(codes, 1):
        st_ = start_date
        if c in latest_map:
            st_ = (latest_map[c] - pd.Timedelta(days=7)).strftime("%Y%m%d")
        try:
            df = _windowed("daily", st_, end_d, ts_code=ts_code(c, "stock"))
            if len(df):
                sub = pd.DataFrame({
                    "date": pd.to_datetime(df["trade_date"]),
                    "code": c,
                    "close": pd.to_numeric(df["close"], errors="coerce"),
                    "pre_close": pd.to_numeric(df["pre_close"], errors="coerce"),
                }).dropna()
                parts.append(sub[sub["date"] >= pd.Timestamp(start_date)])
                ok += 1
        except Exception:  # noqa
            pass
        if progress:
            progress(i / len(codes), f"板块复权日线 {c} ({i}/{len(codes)})")
    n = fetchers.append_ds("sector_px", pd.concat(parts, ignore_index=True)) if parts else 0
    return [f"【板块复权日线】更新 {ok}/{len(codes)} 只成分股，写入 {n} 行"]


'''

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

# ───────────── 数据状态与更新入口 ─────────────
with st.expander("数据覆盖与历史回补（自 PX_START = 2019-01-01 起）", expanded=mkt_raw.empty or sec_raw.empty):
    c1, c2, c3 = st.columns(3)
    if len(mkt_raw):
        st.caption(
            f"全A平均股价：已缓存 **{mkt_raw['date'].nunique()}** 个交易日"
            f"（{mkt_raw['date'].min().date()} ~ {mkt_raw['date'].max().date()}）；"
            f"板块复权日线：**{sec_raw['date'].nunique() if len(sec_raw) else 0}** 个交易日"
        )
    else:
        st.info("还没有「全A平均股价」缓存数据。首次使用可先点右侧「仅拉防御/板块复权日线(约1分钟)」，再点「回补全A历史至 2019」"
                "（全市场按交易日拉取，每20天自动存盘，可随时中断并在下次继续）。")
    if c1.button("🔄 增量更新至最新", width="stretch"):
        run_job("⑦ 增量更新平均股价", lambda p: fetchers.download_market_avg(60, full_history=False, progress=p))
    if c2.button("⚡ 回补全A历史至 2019", width="stretch",
                 help="从最新日期由新到旧回补缺失交易日（2019起约1900次调用，每20天存盘，可断点续传）"):
        run_job("⑦ 全A平均股价历史回补", lambda p: fetchers.download_market_avg(full_history=True, progress=p))
    if c3.button("🛡️ 仅拉防御/板块复权日线", width="stretch",
                 help="只拉取 config.SECTORS 约45只股票的历史 close/pre_close（约1分钟）"):
        run_job("板块复权日线", lambda p: fetchers.download_sector_px(progress=p))

if mkt_raw.empty and sec_raw.empty:
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
win_label = st.sidebar.radio("展示区间", list(win_opts.keys()), index=4)
win_n = win_opts[win_label]

chart_mode = st.sidebar.radio("主图模式", ["双轴实际均价(元)", "归一化走势(起点=1)", "同轴实际均价(元)"], index=0)

mkt_df = an.market_avg_frame(mkt_raw, asof)
def_df = an.basket_avg_frame(sec_raw, def_codes, asof)

# 对齐展示区间
if len(mkt_df) and len(def_df):
    common_idx = mkt_df.index.intersection(def_df.index)
    if len(common_idx):
        mkt_view = mkt_df.reindex(common_idx)
        def_view = def_df.reindex(common_idx)
    else:
        st.warning("全A与防御板块暂无重叠交易日，请先同步更新两边数据。")
        mkt_view, def_view = mkt_df, def_df
else:
    mkt_view, def_view = mkt_df, def_df

if win_n > 0:
    mkt_view = mkt_view.tail(win_n)
    def_view = def_view.tail(win_n)

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
if len(def_view) and len(mkt_view) and len(def_view.index.intersection(mkt_view.index)):
    s_def = def_view[col].dropna()
    s_mkt = mkt_view[col].dropna()

    if chart_mode == "双轴实际均价(元)":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=s_def.index, y=s_def, name="防御篮子平均股价(左轴)",
                                 line=dict(color="#d62728", width=2.2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=s_mkt.index, y=s_mkt, name="全A平均股价(右轴)",
                                 line=dict(color="#1f77b4", width=2.0)), secondary_y=True)
        fig.update_yaxes(title_text="防御篮子均价 (元)", secondary_y=False)
        fig.update_yaxes(title_text="全A均价 (元)", secondary_y=True)
    elif chart_mode == "归一化走势(起点=1)":
        fig = go.Figure()
        fig.add_scatter(x=s_def.index, y=s_def / s_def.iloc[0], name="防御篮子(起点=1)",
                        line=dict(color="#d62728", width=2.2))
        fig.add_scatter(x=s_mkt.index, y=s_mkt / s_mkt.iloc[0], name="全A(起点=1)",
                        line=dict(color="#1f77b4", width=2.0))
        fig.update_yaxes(title_text="区间净值 (起点=1)")
    else:
        fig = go.Figure()
        fig.add_scatter(x=s_def.index, y=s_def, name="防御篮子平均股价(元)",
                        line=dict(color="#d62728", width=2.2))
        fig.add_scatter(x=s_mkt.index, y=s_mkt, name="全A平均股价(元)",
                        line=dict(color="#1f77b4", width=2.0))
        fig.update_yaxes(title_text="平均股价 (元)")

    fig.update_layout(height=440, margin=dict(t=25, b=20), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")

    # ───────────── 曲线 2：相对比值（防御均价 ÷ 全A均价）─────────────
    ratio = (s_def / s_mkt).dropna()
    if len(ratio):
        st.subheader("相对强弱：防御篮子均价 ÷ 全A均价")
        fig2 = go.Figure()
        fig2.add_scatter(x=ratio.index, y=ratio, name="防御均价 / 全A均价", line=dict(color="#9467bd", width=2))
        fig2.add_scatter(x=ratio.index, y=ratio.rolling(20).mean(), name="20日均线",
                         line=dict(color="#7f7f7f", dash="dash", width=1.4))
        fig2.update_layout(height=300, margin=dict(t=20, b=20), hovermode="x unified",
                           yaxis_title="比值 (上升=防御相对跑赢)")
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            f"当前比值 **{num(ratio.iloc[-1], 3)}**（区间最高 {num(ratio.max(), 3)} / 最低 {num(ratio.min(), 3)}）。"
            "比值上升说明防御板块相对全市场平均水平走强（偏避险），下降说明市场风险偏好回升。"
        )
elif len(def_view):
    st.warning("当前只有防御篮子数据，缺少全A数据，先在上方点「增量更新」或「回补全A历史至2019」。")
    fig = go.Figure()
    fig.add_scatter(x=def_view.index, y=def_view[col], name="防御篮子平均股价(元)")
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
elif len(mkt_view):
    st.warning("当前只有全A数据，缺少板块复权日线，点上方「仅拉防御/板块复权日线」。")
    fig = go.Figure()
    fig.add_scatter(x=mkt_view.index, y=mkt_view[col], name="全A平均股价(元)")
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

# ───────────── 分板块对比 & 口径说明 ─────────────
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
            if win_n > 0:
                sub = sub.tail(win_n)
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def replace_once(src: str, old: str, new: str, label: str) -> str:
    cnt = src.count(old)
    if cnt != 1:
        raise RuntimeError(f"{label}：期望找到 1 处锚点，实际 {cnt} 处。")
    return src.replace(old, new)


def patch_analytics(src: str):
    if "def market_avg_frame(" in src:
        return None
    return src.rstrip() + "\n" + ANALYTICS_ADD.lstrip("\n")


def patch_fetchers(src: str):
    if '"market_avg"' in src:
        return None
    src = replace_once(
        src,
        '    "proxy_daily": ["date", "symbol", "open", "high", "low", "close", "volume"],\n}',
        '    "proxy_daily": ["date", "symbol", "open", "high", "low", "close", "volume"],\n'
        '    "market_avg": ["date", "n", "raw_avg", "sum_close", "sum_pre", "eq_ret"],\n'
        '    "sector_px": ["date", "code", "close", "pre_close"],\n}',
        "fetchers.py SCHEMAS",
    )
    src = replace_once(
        src,
        '        "proxy_daily": ["date", "symbol"]}',
        '        "proxy_daily": ["date", "symbol"], "market_avg": ["date"], "sector_px": ["date", "code"]}',
        "fetchers.py KEYS",
    )
    src = replace_once(
        src,
        "# ───────────── 一键更新全部 ─────────────",
        FETCHERS_ADD_FUNCS + "# ───────────── 一键更新全部 ─────────────",
        "fetchers.py download_market_avg 插入点",
    )
    src = replace_once(
        src,
        '        ("⑤ 外资代理", lambda p: download_proxies(p)),\n    ]',
        '        ("⑤ 外资代理", lambda p: download_proxies(p)),\n'
        '        ("⑦ 全A平均股价", lambda p: download_market_avg(backfill_days, full_history=False, progress=p)),\n    ]',
        "fetchers.py update_all",
    )
    return src


def patch_tushare(src: str):
    if "def update_market_avg(" in src:
        return None
    src = replace_once(
        src,
        '            frames = []\n'
        '            for d in days[i:i + chunk_days]:\n'
        '                df = call("daily", trade_date=_ymd(d))\n'
        '                if len(df):\n'
        '                    frames.append(df)\n'
        '                if progress:\n'
        '                    progress(min((days.index(d) + 1) / len(days), 1.0), f"本地日线 {d.date()}")\n'
        '            if not frames:',
        '            frames, day_frames = [], []\n'
        '            for d in days[i:i + chunk_days]:\n'
        '                df = call("daily", trade_date=_ymd(d))\n'
        '                if len(df):\n'
        '                    frames.append(df)\n'
        '                    day_frames.append((_ymd(d), df))\n'
        '                if progress:\n'
        '                    progress(min((days.index(d) + 1) / len(days), 1.0), f"本地日线 {d.date()}")\n'
        '            if day_frames:\n'
        '                record_avg_from_daily(day_frames)\n'
        '            if not frames:',
        "tushare_src.py update_local_stock_daily",
    )
    src = replace_once(
        src,
        "# ───────────── 接口自检 ─────────────",
        TUSHARE_ADD_FUNCS + "# ───────────── 接口自检 ─────────────",
        "tushare_src.py update_market_avg 插入点",
    )
    return src


def patch_freshness(src: str):
    if '"全A平均股价"' in src:
        return None
    return replace_once(
        src,
        '    pr = fetchers.read_ds("proxy_daily")',
        '    mkt = fetchers.read_ds("market_avg")\n'
        '    last_mkt = mkt["date"].max() if len(mkt) else None\n'
        '    rows.append(_row("全A平均股价", last_mkt, expected,\n'
        '                     f"{mkt[\'date\'].nunique()} 个交易日" if len(mkt) else "需点回补/更新", core=False))\n'
        '    pr = fetchers.read_ds("proxy_daily")',
        "freshness.py collect",
    )


def patch_app(src: str):
    if "views/avg_price.py" in src:
        return None
    return replace_once(
        src,
        '    st.Page("views/dashboard.py", title="指数分析", icon="📈", default=True),\n'
        '    st.Page("views/value_screen.py", title="价值质量选股", icon="💎"),',
        '    st.Page("views/dashboard.py", title="指数分析", icon="📈", default=True),\n'
        '    st.Page("views/avg_price.py", title="平均股价对比", icon="⚖️"),\n'
        '    st.Page("views/value_screen.py", title="价值质量选股", icon="💎"),',
        "app.py st.navigation",
    )


def patch_data_view(src: str):
    if "⑦ 全A平均股价" in src:
        return None
    return replace_once(
        src,
        'if st.button("指数成分股名单（融资买入占比需要）"):\n'
        '    run_job("指数成分股", lambda p: fetchers.update_index_cons())',
        'if st.button("指数成分股名单（融资买入占比需要）"):\n'
        '    run_job("指数成分股", lambda p: fetchers.update_index_cons())\n'
        'd_cols = st.columns(2)\n'
        'if d_cols[0].button("⑦ 全A平均股价（增量补齐最新）", width="stretch"):\n'
        '    run_job("⑦ 全A平均股价", lambda p: fetchers.download_market_avg(int(backfill), full_history=False, progress=p))\n'
        'if d_cols[1].button("⑦ 全A平均股价（历史回补至 2019）", width="stretch"):\n'
        '    run_job("⑦ 全A平均股价历史回补", lambda p: fetchers.download_market_avg(full_history=True, progress=p))',
        "views/data.py 单独下载按钮",
    )


def main():
    tasks = [
        (ROOT / "analytics.py", patch_analytics),
        (ROOT / "fetchers.py", patch_fetchers),
        (ROOT / "tushare_src.py", patch_tushare),
        (ROOT / "freshness.py", patch_freshness),
        (ROOT / "app.py", patch_app),
        (ROOT / "views" / "data.py", patch_data_view),
    ]

    # 1. 先检查并计算所有补丁（全有或全无，不留半截修改）
    planned = []
    for path, fn in tasks:
        if not path.exists():
            sys.exit(f"[中止] 找不到文件：{path}。请确认脚本放在项目根目录。")
        old = read_text(path)
        try:
            new = fn(old)
        except Exception as e:
            sys.exit(f"[中止] {path.name} 失败：{e}\n所有文件均未修改。")
        if new is not None:
            compile(new, str(path), "exec")
            planned.append((path, new))
        else:
            print(f"[跳过] {path.relative_to(ROOT)} 已包含该功能")

    avg_view_path = ROOT / "views" / "avg_price.py"
    compile(AVG_PRICE_VIEW, str(avg_view_path), "exec")

    # 2. 备份并写入
    for path, new in planned:
        bak = path.with_suffix(path.suffix + ".bak_before_avg_price")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(new, encoding="utf-8")
        print(f"[更新] {path.relative_to(ROOT)}（备份：{bak.name}）")

    avg_view_path.parent.mkdir(parents=True, exist_ok=True)
    avg_view_path.write_text(AVG_PRICE_VIEW, encoding="utf-8")
    print(f"[新建] {avg_view_path.relative_to(ROOT)}")

    print("\n全部完成！请重启 Streamlit：streamlit run app.py")
    print("进入新页面「⚖️ 平均股价对比」后：")
    print("  1. 先点「🛡️ 仅拉防御/板块复权日线」（约 1 分钟）即可先看到 2019 起的防御板块曲线；")
    print("  2. 再点「⚡ 回补全A历史至 2019」（按日由新到旧拉取并每 20 天存盘，可随时中断续传）。")


if __name__ == "__main__":
    main()