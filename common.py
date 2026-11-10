"""两个页面共用：常量、格式化、带缓存的读数、更新任务、新鲜度展示、自动更新触发。"""
import os

import numpy as np
import pandas as pd
import streamlit as st

import analytics as an  # noqa: F401
import autoupdate
import config
import fetchers
import freshness
import loaders
import tushare_src as tsrc  # noqa: F401

IDX = config.INDEXES
IDX_CODES = [v["code"] for v in IDX.values()]
ETF_ALL = sorted({c for v in config.ETF_GROUPS.values() for c in v})
SEC_CODES = sorted({c for s in config.SECTORS.values() for c, _ in s["stocks"]})
NAME = {c: n for s in config.SECTORS.values() for c, n in s["stocks"]}


# ───────────── 格式化 ─────────────
def pct(x, d=2):
    return "—" if x is None or x != x else f"{x * 100:+.{d}f}%"


def num(x, d=1, unit=""):
    return "—" if x is None or x != x else f"{x:,.{d}f}{unit}"


def color_updown(v):          # A股习惯：红涨绿跌
    if isinstance(v, str) or v is None or v != v:
        return ""
    return "color:#d62728" if v > 0 else ("color:#2ca02c" if v < 0 else "")


def show(df, pct_cols=(), num_cols=(), color_cols=(), height=None):
    """数值列先转成带单位的字符串（空值显示"—"，避免前端显示 None），颜色按原始数值计算。"""
    disp = df.copy()
    for c in pct_cols:
        if c in disp.columns:
            disp[c] = [pct(v) for v in df[c]]
    for c in num_cols:
        if c in disp.columns:
            disp[c] = [num(v) for v in df[c]]
    sty = disp.style
    for c in color_cols:
        if c in df.columns:
            sty = sty.apply(lambda col, c=c: [color_updown(v) for v in df[c]], subset=[c])
    kw = {"height": height} if height else {}
    st.dataframe(sty, width="stretch", hide_index=True, **kw)


# ───────────── 数据读取（带缓存）─────────────
def stamp():
    return (fetchers.cache_stamp(), st.session_state.get("rev", 0))


@st.cache_data(show_spinner=False)
def close_panel(codes, stamp_key):
    return loaders.load_panel(list(codes), "close", tail=500)


@st.cache_data(show_spinner=False)
def amount_panel(codes, stamp_key):
    return loaders.load_panel(list(codes), "amount", tail=500)


@st.cache_data(show_spinner=False)
def ds(name, stamp_key):
    return fetchers.read_ds(name)


@st.cache_data(show_spinner=False)
def fresh_table(stamp_key, day_key):
    return freshness.collect()


def bump():
    """数据变了：重扫文件、清缓存、让所有缓存键失效。"""
    loaders.rescan()
    st.cache_data.clear()
    st.session_state["rev"] = st.session_state.get("rev", 0) + 1


def run_job(title, fn, rerun=True):
    bar = st.progress(0.0, text=title)
    log = fn(lambda p, t: bar.progress(min(max(float(p), 0.0), 1.0), text=t))
    bar.empty()
    bump()
    st.session_state["last_log"] = (title, log)
    st.session_state.setdefault("log_hist", {})[title] = log
    autoupdate.append_log(title, log)
    if rerun:
        st.rerun()      # 按钮在日志展示之后执行，重跑一次才能立刻显示本次日志


def render_freshness(expanded=False, link=True):
    """数据新鲜度：一行结论 + 明细表。"""
    df = fresh_table(stamp(), str(pd.Timestamp.now().date()))
    exp = df["应有日期"].iloc[0] if len(df) else None
    idx_row = df[df["数据"].str.startswith("指数日线")]
    last = idx_row["最新日期"].iloc[0] if len(idx_row) else None
    bad = df[df["核心"] & df["状态"].str.startswith(("⚠️", "❌"))]
    if last is None or last != last:
        head = "数据状态：❌ 没有指数日线，请到「数据」页更新"
        box = st.error
    elif len(bad):
        head = (f"数据状态：⚠️ 日线最新到 **{pd.Timestamp(last).date()}**，应有 **{pd.Timestamp(exp).date() if exp is not None else '—'}**；"
                f"落后/缺失：{'、'.join(bad['数据'].str.split('(').str[0])}")
        box = st.warning
    else:
        head = f"数据状态：✅ 日线最新到 **{pd.Timestamp(last).date()}**（应有 {pd.Timestamp(exp).date() if exp is not None else '—'}），各数据均已更新"
        box = st.success
    box(head)
    if link:
        st.page_link("views/data.py", label="去「数据」页更新 / 查看详情 →")
    with st.expander("各数据最新日期明细", expanded=expanded):
        t = df.drop(columns=["核心"]).copy()
        for c in ("最新日期", "应有日期"):
            t[c] = pd.to_datetime(t[c]).dt.strftime("%Y-%m-%d").fillna("—")
        st.dataframe(t, hide_index=True, width="stretch")


@st.fragment(run_every="5m")
def auto_update_watcher():
    """每 5 分钟（页面开着的话）检查一次：交易日收盘后核心数据落后 → 自动更新，完成后刷新整页。
    首次打开页面也会立刻检查一次。同一会话 4 分钟内的重复运行（如点按钮触发的整页重跑）会被跳过。"""
    now = pd.Timestamp.now()
    nxt = st.session_state.get("auto_next_check")
    if nxt is not None and now < nxt:
        return
    st.session_state["auto_next_check"] = now + pd.Timedelta(minutes=4)
    try:
        go, why = autoupdate.should_run(now)
    except Exception as e:  # noqa
        st.session_state["auto_status"] = f"自动更新检查出错：{e}"
        return
    st.session_state["auto_status"] = f"{now:%H:%M} " + ("触发自动更新：" if go else "检查：未触发，") + why
    if go:
        with st.status("收盘后自动更新最新数据…", expanded=True) as status:
            run_job("自动更新", lambda p: autoupdate.run(p, now), rerun=False)
            status.update(label="自动更新完成", state="complete", expanded=False)
        st.rerun()          # 整页刷新，让页面用上新数据
