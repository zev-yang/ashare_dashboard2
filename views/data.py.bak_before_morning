"""页面二：数据。所有下载 / 上传 / 自动更新 / 检查功能都在这里。"""
import os

import pandas as pd
import streamlit as st

import autoupdate
import config
import fetchers
import freshness
import loaders
import tushare_src as tsrc
from common import IDX_CODES, ETF_ALL, SEC_CODES, bump, render_freshness, run_job

st.title("数据中心")
st.caption(f"本地日线目录：`{config.DAILY_DIR}`　｜　自动下载缓存：`{config.CACHE_DIR}`")
if config.DEMO:
    st.warning("演示模式：读写的是 data_demo/ 里的模拟数据。")

# ───────────── 1. 新鲜度 ─────────────
st.subheader("1. 数据是不是最新？")
render_freshness(expanded=True, link=False)
with st.expander("扫描本地日线目录（个股日线整体最新到哪天）"):
    if st.button("开始扫描（读取每个文件的最后一行，几秒钟）"):
        sc = loaders.scan_local_dir()
        exp = freshness.expected_latest()
        if sc.empty:
            st.info("目录里没有找到日线文件。")
        else:
            dist = sc.groupby(sc["最后日期"].dt.strftime("%Y-%m-%d").fillna("无法读取")).size().rename("文件数")
            st.write(f"共 {len(sc)} 个文件；应有最新日期 **{exp.date() if exp is not None else '—'}**；"
                     f"最新日期分布（只列最近 15 个）：")
            st.dataframe(dist.sort_index(ascending=False).head(15), width="stretch")
            stale = sc[sc["最后日期"] < exp] if exp is not None else sc.iloc[0:0]
            if len(stale):
                st.warning(f"{len(stale)} 个文件落后于 {exp.date()}（可能是停牌/退市，或需要点 ⑥ 更新）")
                st.dataframe(stale.sort_values("最后日期").head(50), hide_index=True, width="stretch")

# ───────────── 2. 一键更新 ─────────────
st.subheader("2. 一键更新")
set_ = autoupdate.load_settings()
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1.3, 1.5])
    backfill = c2.number_input("首次回补交易日数", 5, 500, int(set_["backfill"]), key="backfill")
    inc_sector = c3.checkbox("含板块个股日线", set_["include_sector"], key="inc_sector")
    inc_cons = c4.checkbox("含成分股日线(免费源慢)", set_["include_cons"], key="inc_cons",
                           help="免费源逐只下载约1800只很慢。配置了 Tushare token 时由「⑥ 本地个股日线」按日批量覆盖，无需勾选")
    inc_local = c5.checkbox("含⑥本地个股日线(追加)", set_["include_local"], key="inc_local",
                            help="仅在配置了 Tushare token 时生效：把你日线目录里的个股日线补到最新")
    if c1.button("🔄 更新全部数据", type="primary", width="stretch"):
        autoupdate.save_settings({"backfill": int(backfill), "include_sector": inc_sector,
                                  "include_cons": inc_cons, "include_local": inc_local})
        run_job("更新全部数据", lambda p: fetchers.update_all(int(backfill), inc_sector, inc_cons, inc_local, p))

if "last_log" in st.session_state:
    t, lg = st.session_state["last_log"]
    with st.expander(f"最近一次更新日志：{t}", expanded=True):
        st.text("\n".join(lg))
    hist = st.session_state.get("log_hist", {})
    if len(hist) > 1:
        with st.expander("各步骤最近一次日志"):
            for k, v in hist.items():
                st.markdown(f"**{k}**")
                st.text("\n".join(v))
with st.expander("历史更新日志（含自动更新 / 计划任务的运行记录）"):
    st.text(autoupdate.read_log_tail(80) or "还没有记录")

# ───────────── 3. 单独下载 ─────────────
st.subheader("3. 单独下载（增量，均按交易日）")
b = st.columns(3)
if b[0].button("① 融资融券（个股明细）", width="stretch"):
    run_job("① 融资融券", lambda p: fetchers.download_margin(int(backfill), p))
if b[1].button("② 期指（持仓排名 + 合约日行情）", width="stretch"):
    run_job("② 期指", lambda p: fetchers.download_futures(int(backfill), p))
if b[2].button("③ 四大指数日线 IF/IH/IC/IM", width="stretch"):
    run_job("③ 四大指数日线", lambda p: fetchers.download_prices("③", progress=p))
b = st.columns(3)
if b[0].button("④ ETF + 板块个股日线 + ETF份额", width="stretch"):
    run_job("④ ETF/个股", lambda p: fetchers.download_prices("④", inc_sector, inc_cons, progress=p)
            + fetchers.incremental_update("etf_shares", int(backfill)))
if b[1].button("⑤ 外资代理（A50/USDCNH/AH溢价…）", width="stretch"):
    run_job("⑤ 外资代理", lambda p: fetchers.download_proxies(p))
if b[2].button("⑥ 本地个股日线批量更新（Tushare）", width="stretch",
               help="按交易日一次取全市场，把你日线目录里每只股票补到最新（追加）"):
    run_job("⑥ 本地个股日线", lambda p: fetchers.download_local_stocks(p))
if st.button("指数成分股名单（融资买入占比需要）"):
    run_job("指数成分股", lambda p: fetchers.update_index_cons())

# ───────────── 4. 自动更新 ─────────────
st.subheader("4. 自动更新（每个交易日）")
with st.container(border=True):
    a1, a2, a3 = st.columns([1.2, 1, 1])
    auto_on = a1.toggle("收盘后首次打开网页时自动更新", set_["auto_update"], key="auto_on",
                        help="仅在：今天是交易日 + 已过截止时间 + 核心数据落后 时触发；失败后隔一段时间才会重试")
    h, m = freshness.parse_cutoff(set_["cutoff"])
    cut = a2.time_input("截止时间（之后才算当日数据已齐）", pd.Timestamp(2000, 1, 1, h, m).time(), key="auto_cut")
    retry = a3.number_input("失败后重试间隔(分钟)", 5, 240, int(set_["retry_minutes"]), key="auto_retry")
    new_set = {"auto_update": bool(auto_on), "cutoff": cut.strftime("%H:%M"), "retry_minutes": int(retry)}
    if any(set_[k] != v for k, v in new_set.items()):
        autoupdate.save_settings(new_set)
    stt = autoupdate.load_state()
    st.caption(f"上次尝试：{stt.get('last_attempt', '无')}　｜　上次成功：{stt.get('last_success', '无')}　｜　"
               f"{st.session_state.get('auto_status', '本次会话尚未检查')}")
    if st.button("现在按自动更新规则检查一次"):
        go, why = autoupdate.should_run()
        st.write(("会更新：" if go else "不会更新：") + why)
    st.markdown("**网页关着时也要每天自动更新** → 用 Windows 计划任务运行 `auto_update.py`"
                "（每个工作日 18:00，非交易日/已最新会自动跳过）。在 PowerShell 里执行下面这一行创建任务：")
    st.code(autoupdate.schtasks_command(), language="powershell")
    st.caption("计划任务读不到你终端里的环境变量，Tushare token 请在下面「数据源」里点「保存到本地文件」。"
               "手动试跑：`python auto_update.py --force`。")

# ───────────── 5. 数据源 / token ─────────────
st.subheader("5. 数据源 / Tushare")
with st.container(border=True):
    st.caption("✅ 已检测到 Tushare token" if tsrc.available() else "未配置 token：只使用免费源（东财/新浪，网络不稳时会失败）。")
    k1, k2 = st.columns([2, 1])
    tok = k1.text_input("Tushare token", type="password", placeholder="粘贴后点右侧按钮")
    if k2.button("仅本次使用", disabled=not tok):
        os.environ["TUSHARE_TOKEN"] = tok.strip()
        tsrc.reset()
        bump()
        st.rerun()
    if k2.button("保存到本地文件", disabled=not tok, help="保存为项目目录 tushare_token.txt，已在 .gitignore，请勿提交到 GitHub"):
        tsrc.save_token(tok)
        bump()
        st.rerun()
    labels = {"自动：有 token 优先 Tushare，失败再用免费源": "auto", "仅免费源": "free", "仅 Tushare": "tushare"}
    pick = st.radio("数据源", list(labels), index=list(labels.values()).index(config.SOURCE), key="src_pick", horizontal=True)
    config.SOURCE = labels[pick]
    config.TUSHARE_CALLS_PER_MIN = int(st.number_input("Tushare 每分钟调用上限", 5, 5000,
                                                      min(max(int(config.TUSHARE_CALLS_PER_MIN), 5), 5000),
                                                      help="你的 daily 接口上限 50 次/分，默认 42 留余量"))
    if tsrc.available() and st.button("自检 Tushare 接口（权限/字段）"):
        with st.spinner("逐个接口测试中…"):
            st.dataframe(tsrc.self_check(), hide_index=True, width="stretch")

# ───────────── 6. 检查 / 导入 / 导出 ─────────────
st.subheader("6. 检查、导入、导出")
with st.expander("数据对齐检查（颗粒度）"):
    if st.button("运行对齐检查"):
        st.dataframe(fetchers.alignment_report(), hide_index=True, width="stretch")
with st.expander("数据健康检查"):
    st.dataframe(fetchers.health(), hide_index=True, width="stretch")
    miss = [c for c in IDX_CODES + ETF_ALL + SEC_CODES if not loaders.has_local(c)]
    st.write(f"本地缺日线的代码({len(miss)})：", ", ".join(miss) if miss else "无")
    if st.button("重新扫描本地日线目录"):
        bump()
        st.rerun()
with st.expander("手动导入（官网下载的文件）"):
    IMPORT_KINDS = {"期指持仓排名(中金所)": "cffex_rank", "期指日行情(基差用)": "cffex_daily",
                    "融资融券明细(交易所)": "margin", "ETF份额(上交所)": "etf_shares"}
    kind = st.selectbox("文件类型", list(IMPORT_KINDS.keys()) + ["指数成分股(中证官网)"])
    d_in = st.date_input("数据日期", pd.Timestamp.today()) if kind != "指数成分股(中证官网)" else None
    icode = st.selectbox("指数", IDX_CODES) if kind == "指数成分股(中证官网)" else None
    files = st.file_uploader("选择文件(csv/xlsx，可多选)", accept_multiple_files=True)
    if files and st.button("导入"):
        key = IMPORT_KINDS.get(kind, "index_cons")
        for f in files:
            try:
                st.write(fetchers.import_file(key, f.name, f.getvalue(), icode if key == "index_cons" else d_in))
            except Exception as e:  # noqa
                st.error(f"{f.name}: {e}")
        bump()
with st.expander("导出标准化 CSV（程序读取的格式）"):
    name = st.selectbox("数据集", list(fetchers.SCHEMAS.keys()))
    df_x = fetchers.read_ds(name)
    st.caption(f"{len(df_x):,} 行；列：{', '.join(df_x.columns)}")
    if len(df_x):
        st.download_button("下载 CSV", df_x.to_csv(index=False).encode("utf-8-sig"), f"{name}.csv", "text/csv")
    st.caption("日线类按代码保存在 缓存目录/px/ 下（指数为 index_代码.csv），格式 date,open,high,low,close,volume,amount。")
