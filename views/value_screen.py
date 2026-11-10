"""价值 + 质量选股页：数据下载 → 一键运行 → 候选短名单（决策信息卡）。

逻辑移植自 G_Claude 的 build_value_quality_screen.py；数据下载复用本项目的 Tushare 设置（token 在「数据」页配置）。
计算在 value_screen_core.py，下载在 value_screen_data.py。"""
import pandas as pd
import streamlit as st

import tushare_src as ts_src
import value_screen_core as vc
import value_screen_data as vd

st.title("💎 价值 + 质量选股")
st.caption("便宜（PE/PB 的 5 年自身分位，剔除行业与市值倾斜）∩ 财务健康（F-score ≥ 5）。"
           "这是**候选池 + 决策信息卡**，不是持仓单：最后挑 10–15 只你看得懂的，行业分散由你人工把握。")

# ───────────────────────── ① 数据 ─────────────────────────
st.subheader("① 数据")
has_token = ts_src.available()
if not has_token:
    st.warning("没有检测到 Tushare token：请先到「数据」页保存 token（或设置环境变量 TUSHARE_TOKEN）。")

st.dataframe(vd.data_status(), hide_index=True)
st.caption("必需：daily_basic、财务指标、行业分类。现金流 / 分红 / 近期成交额只用来生成三列信息，缺了也能跑。"
           "全部增量更新、断点续传——中途关掉或点了别的按钮，再点一次会接着下。")


def _run_update(label, fn):
    bar = st.progress(0.0, text=f"{label}：准备中…")

    def cb(frac, text):
        bar.progress(min(max(float(frac), 0.0), 1.0), text=text)

    try:
        logs = fn(cb)
    except Exception as e:  # noqa
        logs = [f"【{label}】失败：{e!r}"]
    bar.empty()
    st.session_state["vs_update_log"] = logs
    st.rerun()


if st.button("🔄 更新全部数据（增量）", type="primary", disabled=not has_token, key="vs_update_all"):
    _run_update("全部数据", vd.update_all)

with st.expander("单独更新某类数据"):
    cols = st.columns(3)
    for i, (label, fn) in enumerate(vd.STEPS):
        if cols[i % 3].button(label, key=f"vs_upd_{i}", disabled=not has_token):
            _run_update(label, fn)
    st.caption("首次下载较久（daily_basic 约 1500 次调用、分红约 1200 次调用）；之后每天只补新的交易日。"
               "调用太慢/太快可设环境变量 VS_CALLS_PER_MIN（默认 100，遇到频控会自动降速）。")

if st.session_state.get("vs_update_log"):
    with st.expander("上次更新日志", expanded=True):
        st.code("\n".join(st.session_state["vs_update_log"]), language=None)

# ───────────────────────── ② 运行 ─────────────────────────
st.subheader("② 运行选股")
with st.expander("筛选参数（默认 = 原脚本的预注册口径；改动即偏离预注册，谨慎）"):
    c1, c2, c3 = st.columns(3)
    excl = c1.slider("剔除流通市值最小的比例", 0.0, 0.6, 0.30, 0.05)
    cheap = c2.slider("取最便宜的比例（decile）", 0.05, 0.30, 0.10, 0.05)
    fmin = c3.slider("F-score 健康线（≥）", 1, 7, 5, 1)
    c4, c5 = st.columns(2)
    amt_min = c4.number_input("流动性提示线：20 日均成交额（亿元）", 0.0, 10.0, 0.5, 0.1)
    ex_bj = c5.checkbox("daily_basic 里剔除北交所", value=False,
                        help="原脚本不剔除（北交所股票会参与中性化和市值分位，但因没有财务数据最终进不了候选）。"
                             "勾选后与财务/行业数据的沪深口径完全一致，但结果会和原脚本略有差异。")
params = vc.Params(excl_small_pct=excl, cheap_pct=cheap, fscore_min=int(fmin),
                   amt_min_yi=float(amt_min), exclude_bj=bool(ex_bj))

missing = vd.missing_required()
if missing:
    names = {"daily_basic": "daily_basic", "fundamentals": "财务指标", "industry": "行业分类"}
    st.info("还缺必需数据：" + "、".join(names[m] for m in missing) + "。请先在上面点「更新全部数据」。")

if st.button("▶ 运行选股", type="primary", disabled=bool(missing), key="vs_run"):
    with st.spinner("计算中…（在独立进程里运行，通常 10–30 秒）"):
        st.session_state.pop("vs_error", None)
        try:
            st.session_state["vs_result"] = vc.run_in_subprocess(params)
        except vc.DataMissing as e:
            st.error(f"数据不足：{e}。请先更新数据。")
        except vc.ScreenFailed as e:
            st.session_state["vs_error"] = e

err = st.session_state.get("vs_error")
if err is not None:
    st.error(f"选股进程异常退出（{err}）：{err.hint()}")
    if err.trace:
        st.caption("运行轨迹（最后一行 = 出事的阶段；若有 Fatal / access violation 则下面是崩溃时的调用栈）：")
        st.code(err.trace, language=None)
    if err.stderr:
        st.caption("错误输出：")
        st.code(err.stderr, language=None)
    st.caption(f"轨迹文件：{vc.trace_path()}")

res = st.session_state.get("vs_result")
if res is not None:
    asof = pd.Timestamp(res.asof)
    age = (pd.Timestamp.today().normalize() - asof).days
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("数据截止", f"{asof:%Y-%m-%d}")
    m2.metric("域内股票", res.counts.get("域", 0))
    m3.metric("最便宜 decile", res.counts.get("最便宜", 0))
    m4.metric("候选（便宜 ∩ 健康）", res.counts.get("候选", 0))
    if age > 5:
        st.warning(f"daily_basic 只到 {asof:%Y-%m-%d}（距今 {age} 天），建议先更新数据再看结果。")
    if res.candidates.empty:
        st.info("没有满足条件的候选。可以放宽参数，或检查财务指标数据是否已更新。")
    else:
        st.caption("按「value便宜排名」从高到低。7 项明细顺序：ROA>0 / CFO>0 / ΔROA>0 / CFO>NI / Δ负债<0 / Δ毛利>0 / Δ周转>0（✓=满足）。")
        st.dataframe(res.candidates, hide_index=True)
        st.download_button("⬇ 下载 CSV", data=res.candidates.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"value_quality_screen_{asof:%Y%m%d}.csv", mime="text/csv", key="vs_dl")
        if res.csv_path:
            st.caption(f"已同时保存到：{res.csv_path}")
    with st.expander("运行日志"):
        st.code("\n".join(res.log), language=None)

with st.expander("怎么用这张表"):
    st.markdown(
        "- **便宜**：PE分位/PB分位低 + value便宜排名高。**健康**：F分高（5–7）+ 7 项明细。\n"
        "- **流动性**：「成交额亿20d」；标 ⚠ 的票资金大了可能买卖困难，优先避开。\n"
        "- **质量避雷**：现金流正年数、分红年数（近 3 个财年，越接近 3 越好 = 持续造血、真金白银派现，利润更可信）。\n"
        "- 三个信息列**不是硬门槛**；行业扎堆自己看着分散（人工判断，没写进规则）。\n"
        "- 「财报期」是该股 F-score 所用的最新已公告财报期末日。")
