"""美股价值 + 质量选股页：宏观背景(VIX/JOLTS) → 后台断点续传下载 → 选股结果与决策卡。"""
import time

import pandas as pd
import streamlit as st

import tushare_src as ts_src
import us_value_screen_core as uvc
import us_value_screen_data as ud

st.title("🇺🇸 美股价值 + 质量选股")
st.caption("便宜（总市值÷净利润TTM / 总市值÷净资产的 5 年自身分位，GICS 行业与市值中性化）∩ 财务健康（TTM F-score ≥ 5）。"
           "同时保留总股东回报（分红+净回购）与前复权历史最低价对照。")

# ───────────────────────── 0. 宏观背景：VIX & JOLTS（纯 CSV 读取）─────────────────────────
m1, m2 = st.columns(2)
if ud.VIX_FILE.exists():
    try:
        vix = pd.read_csv(ud.VIX_FILE).sort_values("trade_date")
        v_last = float(vix["close"].iloc[-1])
        v_chg = float(vix["close"].iloc[-1] - vix["close"].iloc[-6]) if len(vix) >= 6 else 0.0
        v_pct = float((vix["close"].tail(1250) <= v_last).mean())
        zone = "平静 (<20)" if v_last < 20 else ("偏高 (20~30)" if v_last < 30 else "恐慌 (>30，常见折价机会)")
        m1.metric(f"CBOE VIX（{pd.Timestamp(vix['trade_date'].iloc[-1]).date()} · {zone}）",
                  f"{v_last:.2f}", f"近5日 {v_chg:+.2f} ｜ 5年分位 {v_pct:.0%}")
    except Exception:  # noqa
        m1.info("VIX 数据读取失败，可点下方更新。")
else:
    m1.info("暂无 VIX 数据（更新宏观或一键更新后显示）。")

if ud.JOLTS_FILE.exists():
    try:
        jo = pd.read_csv(ud.JOLTS_FILE).sort_values("ts_date")
        if "category" in jo.columns and jo["category"].nunique() > 1:
            tot_rows = jo[jo["category"].astype(str).str.contains("Total|非农|总", case=False, na=False)]
            if len(tot_rows):
                jo = tot_rows
        jo = jo.drop_duplicates("ts_date", keep="last").sort_values("ts_date")
        j_last = float(jo["value"].iloc[-1])
        j_mom = float(jo["value"].iloc[-1] - jo["value"].iloc[-2]) if len(jo) >= 2 else 0.0
        j_unit = str(jo["unit"].iloc[-1]) if "unit" in jo.columns else ""
        m2.metric(f"美国 JOLTS 职位空缺（{pd.Timestamp(jo['ts_date'].iloc[-1]).date()}）",
                  f"{j_last:,.0f} {j_unit}", f"环比 {j_mom:+,.0f}")
    except Exception:  # noqa
        m2.info("JOLTS 数据读取失败。")
else:
    m2.info("暂无 JOLTS 职位空缺数据（更新宏观或一键更新后显示）。")

# ───────────────────────── ① 数据下载与状态 ─────────────────────────
st.subheader("① 数据")
if not ts_src.available():
    st.warning("未检测到 Tushare token：仍可拉取 Nasdaq / Sina / 东财数据，但无法核验 us_basic (EQ/ADR) 与更新 VIX / JOLTS。")

st.dataframe(ud.data_status(), hide_index=True, width="stretch")
st.caption("后台独立子进程下载、不阻塞网页：Stage 1 仅对 Top N 市值股票拉取日线与财务/资产负债表；Stage 2 仅对最便宜 ~10% 候选拉取现金流量表，支持随时停止与断点续传。")

with st.container(border=True):
    s1, s2, s3 = st.columns([1.2, 1.2, 1.2])
    top_n_opt = s1.selectbox("Stage 1 覆盖规模 (按市值 Top N)", [500, 800, 1000, 1500, 2200], index=2, key="us_topn")
    include_adr = s2.checkbox("包含中概股与外资 ADR", value=True, key="us_adr")
    allow_neg_pb = s3.checkbox("负 PB 回购股仅按 PE 排名", value=True, key="us_negpb",
                               help="如 AAPL / MCD / HD 因大额回购导致账面净资产为负，勾选后保留并仅按 PE 5年分位计算便宜度")

c_excl = st.session_state.get("us_excl", 0.30)
c_cheap = st.session_state.get("us_cheap", 0.10)
c_amt = st.session_state.get("us_amt", 5.0)

prog_now = ud.read_progress()
bg_running = (prog_now.get("state") == "running") and (
    ud.is_pid_alive(prog_now.get("pid")) or (time.time() - float(prog_now.get("updated_at", 0)) < 15)
)


def _start_bg(label, step_key):
    proc = ud.start_background_update(
        label=label, step=step_key,
        top_n=int(top_n_opt), include_adr=bool(include_adr),
        excl_small_pct=float(c_excl), cheap_pct=float(c_cheap),
        amt_min_musd=float(c_amt), allow_neg_pb=bool(allow_neg_pb),
    )
    st.session_state["us_proc"] = proc
    st.session_state["us_bg_was_running"] = True
    st.rerun()


b0, b1, b2, b3, b4 = st.columns([1.4, 1, 1, 1.2, 1.2])
if b0.button("🔄 一键增量更新全部", type="primary", disabled=bg_running, width="stretch", key="us_upd_all"):
    _start_bg("美股全部数据", "all")
if b1.button("更新宏观(VIX/JOLTS)", disabled=bg_running, width="stretch", key="us_upd_macro"):
    _start_bg("美股宏观", "macro")
if b2.button("更新股票池(秒级)", disabled=bg_running, width="stretch", key="us_upd_univ"):
    _start_bg("美股股票池", "universe")
if b3.button("更新 Stage 1 (行情+财务)", disabled=bg_running, width="stretch", key="us_upd_s1"):
    _start_bg("Stage 1", "stage1")
if b4.button("更新 Stage 2 (便宜现金流)", disabled=bg_running, width="stretch", key="us_upd_s2"):
    _start_bg("Stage 2", "stage2")


@st.fragment(run_every="2s")
def _monitor_bg():
    pg = ud.read_progress()
    if not pg:
        return
    st_state = pg.get("state")
    pid = pg.get("pid")
    alive = ud.is_pid_alive(pid)
    if st_state == "running" and not alive and (time.time() - float(pg.get("updated_at", 0)) > 10):
        st_state = "error"
        pg["text"] = "后台子进程意外终止（请看下方日志；已保存分片保留，可直接重跑续传）"

    if st_state == "running":
        st.session_state["us_bg_was_running"] = True
        with st.container(border=True):
            c_bar, c_stop = st.columns([4.5, 1])
            c_bar.progress(min(max(float(pg.get("frac", 0.0)), 0.0), 1.0), text=str(pg.get("text", "更新中…")))
            if c_stop.button("⏹ 停止任务", width="stretch", key="us_bg_stop"):
                ud.stop_background_update(st.session_state.pop("us_proc", None))
                st.session_state["us_bg_was_running"] = False
                st.rerun(scope="app")
            tail = ud.read_log_tail(12)
            if tail:
                st.code(tail, language=None)
    elif st.session_state.get("us_bg_was_running"):
        st.session_state["us_bg_was_running"] = False
        st.session_state.pop("us_proc", None)
        st.rerun(scope="app")


_monitor_bg()

pg_done = ud.read_progress()
if pg_done and pg_done.get("state") in ("done", "error", "stopped"):
    label_s = {"done": "完成", "error": "出错", "stopped": "已停止"}[pg_done["state"]]
    with st.expander(f"最近一次后台更新日志（{pg_done.get('label', '')} · {label_s}）", expanded=(pg_done["state"] != "done")):
        logs_list = pg_done.get("logs") or []
        st.code("\n".join(logs_list) if logs_list else (ud.read_log_tail(30) or str(pg_done.get("text", ""))), language=None)

# ───────────────────────── ② 运行选股 ─────────────────────────
st.subheader("② 运行选股")
with st.expander("筛选参数"):
    p1, p2, p3, p4 = st.columns(4)
    excl = p1.slider("剔除最小市值比例", 0.0, 0.6, 0.30, 0.05, key="us_excl")
    cheap = p2.slider("取最便宜比例 (decile)", 0.05, 0.30, 0.10, 0.05, key="us_cheap")
    fmin = p3.slider("F-score 下限 (≥)", 1, 7, 5, 1, key="us_fmin")
    amt_min = p4.number_input("20日均成交额门槛(百万美元)", 0.0, 100.0, 5.0, 1.0, key="us_amt")

params = uvc.Params(
    top_n=int(top_n_opt),
    excl_small_pct=float(excl),
    cheap_pct=float(cheap),
    fscore_min=int(fmin),
    amt_min_musd=float(amt_min),
    include_adr=bool(include_adr),
    allow_neg_pb=bool(allow_neg_pb),
)

missing = ud.missing_required()
if missing:
    st.info("还缺必需数据：" + "、".join(missing) + "。请先点上方「🔄 一键增量更新全部」或按阶段更新。")

if st.button("▶ 运行美股选股", type="primary", disabled=bool(missing) or bg_running, key="us_run"):
    with st.spinner("计算中…（在独立进程里运行）"):
        st.session_state.pop("us_error", None)
        try:
            st.session_state["us_result"] = uvc.run_in_subprocess(params)
        except uvc.DataMissing as e:
            st.error(f"数据不足：{e}")
        except uvc.ScreenFailed as e:
            st.session_state["us_error"] = e

err = st.session_state.get("us_error")
if err is not None:
    st.error(f"选股进程异常退出（{err}）")
    if err.stderr:
        st.code(err.stderr, language=None)

res = st.session_state.get("us_result")
if res is not None:
    asof = pd.Timestamp(res.asof)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据截止", f"{asof:%Y-%m-%d}")
    c2.metric("域内美股", res.counts.get("域", 0))
    c3.metric("最便宜 decile", res.counts.get("最便宜", 0))
    c4.metric("最终候选", res.counts.get("候选", 0))
    if res.candidates.empty:
        st.info("没有满足条件的候选股。可适当放宽 F-score 或成交额门槛，并确认 Stage 2 现金流量表已更新。")
    else:
        st.caption("7项明细顺序：ROA>0 / CFO>0 / ΔROA>0 / CFO>NI / Δ负债<0 / Δ毛利>0 / Δ周转>0（✓=满足）。"
                   "「股东回报率%」= (最近年报现金分红 + 净股份回购) ÷ 总市值。")
        st.dataframe(res.candidates, hide_index=True, width="stretch")
        st.download_button(
            "⬇ 下载 CSV",
            data=res.candidates.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"us_value_quality_screen_{asof:%Y%m%d}.csv",
            mime="text/csv",
            key="us_dl",
        )
        if res.csv_path:
            st.caption(f"已同时保存到：{res.csv_path}")
    with st.expander("运行日志"):
        st.code("\n".join(res.log), language=None)
