# -*- coding: utf-8 -*-
"""给「① 避险板块」及共享读数层添加：
  1. loaders.py：支持 cache/px 向前补齐更早历史 + 共享前复权 helper (adjust="qfq"，CSV缓存避开 pyarrow)；
  2. common.py：close_panel 支持可选 tail 与 adjust 参数（默认行为不变）；
  3. fetchers.py：支持按指定起始日期回补 4大指数 + SECTORS + REF_ETFS + adj_factor，并将每日复权因子增量接入 update_all；
  4. views/dashboard.py：指数与板块读全历史、板块自动前复权、放开 250 日限制、对齐基准起点、页内一键回补至 2005/2000。

用法（在项目根目录执行）：
    python apply_sector_qfq_longhist_patch.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LOADERS_NEW_FUNCS = r'''def _candidates(code: str, kind=None) -> list:
    paths = _file_index().get(code, [])
    if kind == "index" or (kind is None and _is_index_code(code)):
        return [p for p in paths if _looks_like_index_file(p)]
    if kind == "stock":
        return [p for p in paths if not _looks_like_index_file(p)]
    return paths


def rescan():
    _file_index.cache_clear()


def has_local(code: str) -> bool:
    return bool(_candidates(code))


def _adj_cache_path(code: str) -> Path:
    d = config.CACHE_DIR / "adj_factor"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{code}.csv"


def read_adj_factor(code: str):
    """只读本地缓存的复权因子 CSV（不联网）。返回按日期升序排列的 Series 或 None。"""
    p = _adj_cache_path(code)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, dtype={"trade_date": str})
        if df.empty:
            return None
        s = pd.Series(
            pd.to_numeric(df["adj_factor"], errors="coerce").to_numpy(float),
            index=pd.to_datetime(df["trade_date"].astype(str), errors="coerce"),
        ).dropna().sort_index()
        return s[~s.index.duplicated(keep="last")]
    except Exception:  # noqa
        return None


def save_adj_factor(code: str, df_new: pd.DataFrame):
    """增量合并保存单只标的的复权因子到 CACHE_DIR/adj_factor/{code}.csv。"""
    if df_new is None or df_new.empty:
        return
    p = _adj_cache_path(code)
    sub = df_new[["trade_date", "adj_factor"]].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"].astype(str), errors="coerce").dt.strftime("%Y%m%d")
    sub["adj_factor"] = pd.to_numeric(sub["adj_factor"], errors="coerce")
    sub = sub.dropna(subset=["trade_date", "adj_factor"])
    if sub.empty:
        return
    if p.exists():
        try:
            old = pd.read_csv(p, dtype={"trade_date": str})
            sub = pd.concat([old[["trade_date", "adj_factor"]], sub], ignore_index=True)
        except Exception:  # noqa
            pass
    sub = sub.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    sub.to_csv(p, index=False)


def get_adj_factor(code: str, first_dt=None, last_dt=None, kind=None, fetch_missing="if_missing"):
    """获取个股/ETF 复权因子 Series。
    fetch_missing: True=缺/落后就联网拉取；"if_missing"=仅本地完全没有缓存时才尝试联网一次；False=仅读本地缓存。"""
    import os
    import tushare_src as ts_src

    s = read_adj_factor(code)
    need = False
    if fetch_missing is True:
        if s is None or s.empty:
            need = True
        elif first_dt is not None and s.index.min() > pd.Timestamp(first_dt) + pd.Timedelta(days=45):
            need = True
        elif last_dt is not None and s.index.max() < pd.Timestamp(last_dt):
            need = True
    elif fetch_missing == "if_missing":
        need = (s is None or s.empty)

    if need and (not config.DEMO) and ts_src.available() and not os.environ.get("PYTEST_CURRENT_TEST"):
        is_etf = (kind == "etf") or (kind is None and str(code).startswith(("51", "56", "58", "15", "16")))
        api = "fund_adj" if is_etf else "adj_factor"
        tscode = ts_src.ts_code(code, "etf" if is_etf else "stock")
        st_ = pd.Timestamp(first_dt).strftime("%Y%m%d") if first_dt is not None else "20000101"
        ed_ = pd.Timestamp(last_dt).strftime("%Y%m%d") if last_dt is not None else pd.Timestamp.today().strftime("%Y%m%d")
        if s is not None and not s.empty and first_dt is not None and s.index.min() <= pd.Timestamp(first_dt) + pd.Timedelta(days=45):
            st_ = (s.index.max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if st_ <= ed_:
            try:
                df_af = ts_src._windowed(api, st_, ed_, ts_code=tscode, fields="ts_code,trade_date,adj_factor")
                if df_af is not None and len(df_af):
                    save_adj_factor(code, df_af)
                    s = read_adj_factor(code)
            except Exception:  # noqa
                pass
    return s


def apply_qfq(df: pd.DataFrame, af: pd.Series) -> pd.DataFrame:
    """用复权因子对 OHLC 做前复权（以 df 最新交易日因子为基准=1）。volume / amount 不变。"""
    if df is None or df.empty or af is None or af.empty:
        return df
    aligned = af.reindex(af.index.union(df.index)).sort_index().ffill().bfill().reindex(df.index)
    last_af = float(aligned.iloc[-1])
    if not (last_af > 0 and np.isfinite(last_af)):
        return df
    ratio = aligned / last_af
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = out[c] * ratio
    return out


def load_price(code: str, adjust=None, kind=None, fetch_adj="if_missing"):
    """本地日线优先；若下载缓存里有更早或更新的日期，自动补齐两端（重叠日期仍以本地为准）。
    adjust="qfq" 时自动对 OHLC 做前复权（指数无复权因子会自动跳过）。"""
    paths = _candidates(code, kind=kind)
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            df = _read_any(p)
        except Exception:  # noqa
            continue
        if "close" not in df.columns and "amount" not in df.columns:
            continue
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            if config.CACHE_DIR not in p.parents:          # 自动下载的成交额已是"元"，只对本地文件换算
                df["amount"] = df["amount"] * config.AMOUNT_UNIT
        frames.append(df)
    if not frames:
        return None
    frames.sort(key=lambda d: "close" not in d.columns)    # 有收盘价的排前面（稳定排序，本地仍在前）
    base = frames[0]
    for extra in frames[1:]:
        cols = [c for c in extra.columns if c in base.columns]
        parts = [base]
        early = extra[extra.index < base.index.min()]
        if len(early):
            parts.insert(0, early[cols])
        late = extra[extra.index > base.index.max()]
        if len(late):
            parts.append(late[cols])
        if len(parts) > 1:
            base = pd.concat(parts).sort_index()
            base = base[~base.index.duplicated(keep="last")]
    for c in ("open", "high", "low", "close", "volume"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    if adjust == "qfq" and not (kind == "index" or (kind is None and _is_index_code(code))):
        af = get_adj_factor(code, base.index.min(), base.index.max(), kind=kind, fetch_missing=fetch_adj)
        if af is not None and len(af):
            base = apply_qfq(base, af)
    return base'''

FETCHERS_BACKFILL_FUNCS = r'''

def update_sector_adj(progress=None) -> list:
    """增量更新 SECTORS 成分股复权因子（近期缺日按交易日一次全市场取，每天仅1次调用）。"""
    if not tsrc.available() or getattr(config, "SOURCE", "auto") == "free":
        return []
    import loaders

    sec_codes = sorted({c for s in config.SECTORS.values() for c, _ in s["stocks"]})
    try:
        latest = tsrc.latest_ready_date()
    except Exception as e:  # noqa
        return [f"【板块复权因子】探测最新交易日失败：{explain(e)}"]

    latest_ts = pd.Timestamp(latest)
    lagging_recent, lagging_long = [], []
    for c in sec_codes:
        af = loaders.read_adj_factor(c)
        if af is None or af.empty:
            lagging_long.append(c)
        elif af.index.max() < latest_ts:
            if (latest_ts - af.index.max()).days <= 25:
                lagging_recent.append((c, af.index.max()))
            else:
                lagging_long.append(c)

    ok_days = ok_stocks = 0
    if lagging_recent:
        min_have = min(d for _, d in lagging_recent)
        days = tsrc.trade_dates(min_have + pd.Timedelta(days=1), latest_ts)
        need_set = {tsrc.ts_code(c, "stock"): c for c, _ in lagging_recent}
        for i, d in enumerate(days, 1):
            try:
                df = tsrc.call("adj_factor", trade_date=d.strftime("%Y%m%d"))
                if len(df):
                    sub = df[df["ts_code"].isin(need_set)]
                    for tsc, g in sub.groupby("ts_code"):
                        loaders.save_adj_factor(need_set[tsc], g)
                    ok_days += 1
            except Exception:  # noqa
                pass
            if progress:
                progress(i / max(len(days), 1), f"复权因子 {d.date()}")

    for i, c in enumerate(lagging_long, 1):
        px = loaders.load_price(c, kind="stock")
        st_ = px.index.min() if px is not None and len(px) else pd.Timestamp(config.PX_START)
        af = loaders.get_adj_factor(c, st_, latest_ts, kind="stock", fetch_missing=True)
        if af is not None and len(af):
            ok_stocks += 1
        if progress:
            progress(i / max(len(lagging_long), 1), f"复权因子 {c}")

    if not lagging_recent and not lagging_long:
        return ["【板块复权因子】已是最新"]
    return [f"【板块复权因子】按日补 {ok_days} 天（{len(lagging_recent)} 只），逐只补 {ok_stocks}/{len(lagging_long)} 只"]


def backfill_sector_and_index(start_date: str = "20050101", progress=None) -> list:
    """回补 4大指数 + SECTORS 成分股 + REF_ETFS 历史日线至 start_date，并同步拉齐复权因子。"""
    import loaders

    idx_items = [(v["code"], "index") for v in config.INDEXES.values()]
    sec_items = [(c, "stock") for c in sorted({c for s in config.SECTORS.values() for c, _ in s["stocks"]})]
    etf_items = [(c, "etf") for c in sorted(set(config.REF_ETFS.values()))]
    items = idx_items + sec_items + etf_items

    log = [f"【长历史回补】起始日 {start_date}，共 {len(items)} 个标的（4指数 + {len(sec_items)}板块股 + {len(etf_items)}参考ETF）"]
    total = len(items)
    for i, (code, kind) in enumerate(items):
        if i:
            time.sleep(getattr(config, "PX_SLEEP", 0.3) + random.random() * 0.2)
        px = loaders.load_price(code, kind=kind)
        need_px = (px is None or px.empty or px.index.min() > pd.Timestamp(start_date) + pd.Timedelta(days=20))
        if need_px:
            log.append(download_price(code, kind, start_date=start_date))
        else:
            log.append(f"  {code}({kind}): 本地已涵盖至 {px.index.min().date()}，跳过日线回补")

        if kind in ("stock", "etf") and tsrc.available():
            px2 = loaders.load_price(code, kind=kind)
            fdt = px2.index.min() if px2 is not None and len(px2) else pd.Timestamp(start_date)
            ldt = px2.index.max() if px2 is not None and len(px2) else pd.Timestamp.today()
            af = loaders.get_adj_factor(code, fdt, ldt, kind=kind, fetch_missing=True)
            if af is not None and len(af):
                log[-1] += f"；复权因子 {af.index.min().date()}~{af.index.max().date()}"
            elif kind == "stock":
                log[-1] += "；⚠ 未获取到复权因子"

        if progress:
            progress((i + 1) / total, f"回补 {code} ({i + 1}/{total})")

    loaders.rescan()
    return log
'''

DASHBOARD_SEC_BLOCK = r'''    for nm, c in config.REF_ETFS.items():
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
                st.text("\n".join(st.session_state["last_log"][1]))'''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    cnt = src.count(old)
    if cnt != 1:
        raise RuntimeError(f"{label}：期望找到 1 处锚点，实际 {cnt} 处。")
    return src.replace(old, new)


def patch_loaders(src: str):
    if "def apply_qfq(" in src:
        return None
    old_block = '''def _candidates(code: str) -> list:
    paths = _file_index().get(code, [])
    return [p for p in paths if _looks_like_index_file(p)] if _is_index_code(code) else paths


def rescan():
    _file_index.cache_clear()


def has_local(code: str) -> bool:
    return bool(_candidates(code))


def load_price(code: str):
    """本地日线优先；若下载缓存里有更新的日期，只补本地最后一天之后的行。
    读不出来/缺收盘列的文件会被跳过（原因见 diagnose），自动回退到下一个同代码文件。"""
    paths = _candidates(code)
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            df = _read_any(p)
        except Exception:  # noqa
            continue
        if "close" not in df.columns and "amount" not in df.columns:
            continue
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            if config.CACHE_DIR not in p.parents:          # 自动下载的成交额已是"元"，只对本地文件换算
                df["amount"] = df["amount"] * config.AMOUNT_UNIT
        frames.append(df)
    if not frames:
        return None
    frames.sort(key=lambda d: "close" not in d.columns)    # 有收盘价的排前面（稳定排序，本地仍在前）
    base = frames[0]
    for extra in frames[1:]:
        new = extra[extra.index > base.index.max()]
        if len(new):
            base = pd.concat([base, new[[c for c in new.columns if c in base.columns]]])
    for c in ("open", "high", "low", "close", "volume"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    return base'''
    src = replace_once(src, old_block, LOADERS_NEW_FUNCS, "loaders.py load_price")
    src = replace_once(
        src,
        '''def load_panel(codes, col="close", tail=None) -> pd.DataFrame:
    """返回 日期 x 代码 的宽表；本地没有的代码直接缺列。"""
    series = {}
    for c in codes:
        df = load_price(c)
        if df is not None and col in df.columns:
            s = df[col]
            series[c] = s.tail(tail) if tail else s''',
        '''def load_panel(codes, col="close", tail=None, adjust=None) -> pd.DataFrame:
    """返回 日期 x 代码 的宽表；本地没有的代码直接缺列。"""
    series = {}
    for c in codes:
        df = load_price(c, adjust=adjust)
        if df is not None and col in df.columns:
            s = df[col]
            series[c] = s.tail(tail) if tail else s''',
        "loaders.py load_panel",
    )
    return src


def patch_common(src: str):
    if "adjust=None" in src:
        return None
    return replace_once(
        src,
        '''@st.cache_data(show_spinner=False)
def close_panel(codes, stamp_key):
    return loaders.load_panel(list(codes), "close", tail=500)


@st.cache_data(show_spinner=False)
def amount_panel(codes, stamp_key):
    return loaders.load_panel(list(codes), "amount", tail=500)''',
        '''@st.cache_data(show_spinner=False)
def close_panel(codes, stamp_key, tail=500, adjust=None):
    return loaders.load_panel(list(codes), "close", tail=tail, adjust=adjust)


@st.cache_data(show_spinner=False)
def amount_panel(codes, stamp_key, tail=500):
    return loaders.load_panel(list(codes), "amount", tail=tail)''',
        "common.py close_panel",
    )


def patch_fetchers(src: str):
    if "def backfill_sector_and_index(" in src:
        return None
    src = replace_once(
        src,
        '''def download_price(code: str, kind: str) -> str:
    """增量下载单个代码，合并写入 cache/px/{code}.csv（成交额为"元"口径）。返回一行日志。"""
    px = config.CACHE_DIR / "px"
    px.mkdir(exist_ok=True)
    f = px / (f"index_{code}.csv" if kind == "index" else f"{code}.csv")      # 指数与个股同号时不会互相覆盖
    old = pd.read_csv(f, parse_dates=["date"]) if f.exists() else None
    start = config.PX_START if old is None or old.empty else (old["date"].max() - pd.Timedelta(days=7)).strftime("%Y%m%d")''',
        '''def download_price(code: str, kind: str, start_date: str = None) -> str:
    """增量下载单个代码，合并写入 cache/px/{code}.csv（成交额为"元"口径）。指定 start_date 时可向更早历史回补。"""
    px = config.CACHE_DIR / "px"
    px.mkdir(exist_ok=True)
    f = px / (f"index_{code}.csv" if kind == "index" else f"{code}.csv")      # 指数与个股同号时不会互相覆盖
    old = pd.read_csv(f, parse_dates=["date"]) if f.exists() else None
    base_start = start_date or config.PX_START
    if old is None or old.empty:
        start = base_start
    elif start_date is not None and old["date"].min() > pd.Timestamp(start_date) + pd.Timedelta(days=20):
        start = start_date
    else:
        start = (old["date"].max() - pd.Timedelta(days=7)).strftime("%Y%m%d")''',
        "fetchers.py download_price start_date",
    )
    src = replace_once(
        src,
        '    new = new[new["date"] >= pd.Timestamp(config.PX_START)]',
        '    new = new[new["date"] >= pd.Timestamp(base_start)]',
        "fetchers.py download_price filter",
    )
    src = replace_once(
        src,
        "# ───────────── ⑤ 外资代理指标 ─────────────",
        FETCHERS_BACKFILL_FUNCS.lstrip("\n") + "\n# ───────────── ⑤ 外资代理指标 ─────────────",
        "fetchers.py backfill_sector_and_index",
    )
    src = replace_once(
        src,
        '        ("④ ETF/板块个股日线", lambda p: download_prices("④", include_sector and not local, include_cons and not local, progress=p)),',
        '        ("④ ETF/板块个股日线", lambda p: download_prices("④", include_sector and not local, include_cons and not local, progress=p)),\n'
        '        ("板块复权因子", lambda p: update_sector_adj(p)),',
        "fetchers.py update_all",
    )
    return src


def patch_dashboard(src: str):
    if 'adjust="qfq"' in src:
        return None
    src = replace_once(
        src,
        "from common import (ETF_ALL, IDX, IDX_CODES, NAME, SEC_CODES, amount_panel, close_panel, ds, num, pct,\n"
        "                    render_freshness, show, stamp)",
        "from common import (ETF_ALL, IDX, IDX_CODES, NAME, SEC_CODES, amount_panel, close_panel, ds, num, pct,\n"
        "                    render_freshness, run_job, show, stamp)",
        "views/dashboard.py imports",
    )
    src = replace_once(
        src,
        "idx_close = close_panel(tuple(IDX_CODES), S)",
        "idx_close = close_panel(tuple(IDX_CODES), S, tail=None)",
        "views/dashboard.py idx_close",
    )
    src = replace_once(
        src,
        "sec_close = close_panel(tuple(SEC_CODES), S)",
        'sec_close = close_panel(tuple(SEC_CODES), S, tail=None, adjust="qfq")',
        "views/dashboard.py sec_close",
    )
    old_block = '''    for nm, c in config.REF_ETFS.items():
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
        st.plotly_chart(fig, width="stretch")'''
    src = replace_once(src, old_block, DASHBOARD_SEC_BLOCK, "views/dashboard.py sector chart")
    return src


def main():
    tasks = [
        (ROOT / "loaders.py", patch_loaders),
        (ROOT / "common.py", patch_common),
        (ROOT / "fetchers.py", patch_fetchers),
        (ROOT / "views" / "dashboard.py", patch_dashboard),
    ]

    planned = []
    for path, fn in tasks:
        if not path.exists():
            sys.exit(f"[中止] 找不到 {path}，请把脚本放到项目根目录后运行。")
        old = path.read_text(encoding="utf-8-sig")
        try:
            new = fn(old)
        except Exception as e:
            sys.exit(f"[中止] {path.name} 失败：{e}\n所有文件均未修改。")
        if new is not None:
            compile(new, str(path), "exec")
            planned.append((path, new))
        else:
            print(f"[跳过] {path.relative_to(ROOT)} 已经包含该更新")

    for path, new in planned:
        bak = path.with_suffix(path.suffix + ".bak_before_qfq_hist")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(new, encoding="utf-8")
        print(f"[更新] {path.relative_to(ROOT)}（备份：{bak.name}）")

    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("进入首页「① 避险板块」标签页后：")
    print("  1. 走势已自动启用「前复权」，默认窗口可在 60日 ~ 全部历史 自由切换；")
    print("  2. 展开图表下方的「长历史回补」，点击「📥 回补至 2005-01-01」（或 2000-01-01）即可补齐长周期曲线。")


if __name__ == "__main__":
    main()