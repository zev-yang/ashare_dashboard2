# -*- coding: utf-8 -*-
"""修复「🇺🇸 美股价值选股」点击全量下载时 Streamlit 断开/崩溃的问题：
  1. 主进程零 pyarrow/parquet 读取：VIX / JOLTS / 日历 / 股票池改存 .csv，data_status() 只看文件数与时间戳；
  2. 子进程优先使用 fastparquet + ARROW_USER_SIMD_LEVEL=NONE，避免 pyarrow 非法指令崩溃；
  3. 下载任务改为非阻塞后台子进程 + progress.json + @st.fragment(run_every="2s") 实时刷新，
     即使拉取 Top 1000 只美股耗时 20 分钟也不会阻塞或断开 Streamlit WebSocket。

用法（在项目根目录执行）：
    python fix_us_screen_crash_and_bg.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

US_DATA_NEW = r'''# -*- coding: utf-8 -*-
"""美股价值 + 质量选股 · 数据层（主进程零 pyarrow 读取 + 非阻塞后台子进程更新）。"""
from __future__ import annotations

import datetime as dt
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

import config
import tushare_src as ts_src

ROOT = Path(config.CACHE_DIR) / "us_value_screen"
DIRS = {
    "universe": ROOT / "universe",
    "macro": ROOT / "macro",
    "prices": ROOT / "prices",
    "indicators": ROOT / "indicators",
    "balance": ROOT / "balance",
    "cashflow": ROOT / "cashflow",
    "output": ROOT / "output",
    "logs": ROOT / "logs",
}
UNIVERSE_FILE = DIRS["universe"] / "us_universe.csv"
VIX_FILE = DIRS["macro"] / "vix.csv"
JOLTS_FILE = DIRS["macro"] / "jolts.csv"
CAL_FILE = DIRS["macro"] / "us_tradecal.csv"
PROG_FILE = DIRS["logs"] / "progress.json"
LOG_FILE = DIRS["logs"] / "last_update.log"

NASDAQ_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}

NATIVE_CRASH = {3221225477, 3221225501, 3221226505, -1073741819, -1073741795, -1073740791,
                -11, -4, -6, 139, 132, 134}

Progress = Optional[Callable[[float, str], None]]

DEFAULT_TOP_N = 1000
FIN_REFRESH_DAYS = 14
PX_REFRESH_HOURS = 16


def _read_pq(path: Path, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))


def _to_pq(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[c]):
            out[c] = out[c].dt.tz_localize(None)
        elif out[c].dtype == object:
            out[c] = out[c].fillna("").astype(str)
    out.to_parquet(path, index=False, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))


def _save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _age_hours(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 3600.0


def _retry_ak(fn, tries: int = 3, sleep_base: float = 0.4):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa
            last = e
            if i < tries - 1:
                time.sleep(sleep_base * (2 ** i) + random.random() * 0.25)
    raise last


def _safe_sym_fname(sym: str) -> str:
    return str(sym).replace("/", "_").replace(".", "_")


# ───────────────────────── 宏观与美股交易日历（纯 CSV）─────────────────────────
def latest_us_trade_day() -> str:
    today = dt.date.today()
    if CAL_FILE.exists() and _age_hours(CAL_FILE) < 24:
        try:
            cal = pd.read_csv(CAL_FILE, dtype={"cal_date": str, "is_open": str})
            op = cal[cal["is_open"].astype(str) == "1"]["cal_date"].astype(str)
            past = op[op <= today.strftime("%Y%m%d")]
            if len(past):
                return str(past.max())
        except Exception:  # noqa
            pass
    if ts_src.available():
        try:
            df = ts_src.call("us_tradecal", start_date=(today - dt.timedelta(days=40)).strftime("%Y%m%d"),
                             end_date=(today + dt.timedelta(days=5)).strftime("%Y%m%d"))
            if df is not None and len(df):
                _save_csv(df, CAL_FILE)
                op = df[df["is_open"].astype(str) == "1"]["cal_date"].astype(str)
                past = op[op <= today.strftime("%Y%m%d")]
                if len(past):
                    return str(past.max())
        except Exception:  # noqa
            pass
    return (today - dt.timedelta(days=1)).strftime("%Y%m%d")


def update_macro(progress: Progress = None) -> list:
    if not ts_src.available():
        return ["【美股宏观】未配置 Tushare token，跳过 VIX / JOLTS"]
    log = []
    start_vix = (dt.date.today() - dt.timedelta(days=365 * 6)).strftime("%Y%m%d")
    try:
        vix = ts_src.call("vix_index", start_date=start_vix)
        if vix is not None and len(vix):
            vix["trade_date"] = pd.to_datetime(vix["trade_date"].astype(str), errors="coerce")
            vix = vix.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            _save_csv(vix, VIX_FILE)
            log.append(f"【VIX】{len(vix)} 行，最新 {vix['trade_date'].max().date()} 收盘 {float(vix['close'].iloc[-1]):.2f}")
        else:
            log.append("【VIX】返回为空")
    except Exception as e:  # noqa
        log.append(f"【VIX】失败：{str(e)[:120]}")
    if progress:
        progress(0.5, "更新 JOLTS 职位空缺")
    try:
        jolts = ts_src.call("us_job_offers", start_date="20190101", end_date=dt.date.today().strftime("%Y%m%d"))
        if jolts is not None and len(jolts):
            jolts["ts_date"] = pd.to_datetime(jolts["ts_date"].astype(str), errors="coerce")
            jolts = jolts.dropna(subset=["ts_date"]).sort_values("ts_date")
            _save_csv(jolts, JOLTS_FILE)
            log.append(f"【JOLTS】{len(jolts)} 行，最新 {jolts['ts_date'].max().date()}")
        else:
            log.append("【JOLTS】返回为空")
    except Exception as e:  # noqa
        log.append(f"【JOLTS】失败：{str(e)[:120]}")
    if progress:
        progress(1.0, "宏观数据完成")
    return log


# ───────────────────────── 股票池与 GICS 行业（纯 CSV）─────────────────────────
def update_universe(progress: Progress = None) -> list:
    if progress:
        progress(0.1, "请求 Nasdaq Screener 全市场列表")
    r = requests.get(NASDAQ_URL, headers=UA_HEADERS, timeout=25)
    r.raise_for_status()
    js = r.json()["data"]
    rows = (js.get("table") or js)["rows"]
    nq = pd.DataFrame(rows)
    nq["symbol_raw"] = nq["symbol"].astype(str).str.strip()
    nq["symbol"] = nq["symbol_raw"].str.replace("/", ".", regex=False)
    nq["lastsale"] = pd.to_numeric(nq["lastsale"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    nq["market_cap"] = pd.to_numeric(nq["marketCap"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    nq["volume"] = pd.to_numeric(nq["volume"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    nq["dollar_vol"] = nq["lastsale"] * nq["volume"]
    nq["sector"] = nq["sector"].astype(str).str.strip()
    nq["industry"] = nq["industry"].astype(str).str.strip()
    nq["country"] = nq["country"].astype(str).str.strip()

    if progress:
        progress(0.55, "请求 Tushare us_basic 核验普通股/ADR")
    if ts_src.available():
        ub = ts_src.call("us_basic", list_status="L")
        if ub is not None and len(ub):
            ub_sub = ub[ub["classify"].isin(["EQ", "ADR"])][["ts_code", "classify", "list_date"]].drop_duplicates("ts_code")
            df = nq.merge(ub_sub, left_on="symbol", right_on="ts_code", how="inner")
        else:
            df = nq.assign(classify="EQ", list_date="")
    else:
        df = nq.assign(classify="EQ", list_date="")

    df = df[(df["market_cap"] > 0) & (df["lastsale"] > 0) & (df["sector"] != "") & (df["sector"] != "nan")].copy()
    df = df.sort_values("market_cap", ascending=False).drop_duplicates("symbol", keep="first").reset_index(drop=True)
    keep = ["symbol", "name", "sector", "industry", "country", "classify", "list_date",
            "lastsale", "volume", "dollar_vol", "market_cap"]
    df = df[[c for c in keep if c in df.columns]]
    _save_csv(df, UNIVERSE_FILE)
    if progress:
        progress(1.0, "美股股票池完成")
    return [f"【美股股票池】Nasdaq {len(nq)} 只 → 清洗后 EQ+ADR 共 {len(df)} 只（{df['sector'].nunique()} 个行业）"]


def _sina_candidates(sym: str) -> list:
    s = str(sym).strip()
    cands = [s]
    if "." in s:
        cands.append(s.replace(".", "-"))
        cands.append(s.replace(".", ""))
    return list(dict.fromkeys(cands))


def _fetch_us_price_one(sym: str) -> pd.DataFrame:
    import akshare as ak
    for sc in _sina_candidates(sym):
        try:
            df = _retry_ak(lambda sc=sc: ak.stock_us_hist_sina(symbol=sc, adjust="qfq"), tries=2, sleep_base=0.3)
            if df is not None and len(df):
                out = df[["date", "open", "high", "low", "close", "volume"]].copy()
                out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
                for c in ("open", "high", "low", "close", "volume"):
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                return out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        except Exception:  # noqa
            continue
    return pd.DataFrame()


def _fetch_us_indicator_one(sym: str) -> pd.DataFrame:
    import akshare as ak
    df = _retry_ak(lambda: ak.stock_financial_us_analysis_indicator_em(symbol=sym, indicator="单季报"), tries=3, sleep_base=0.4)
    if df is None or df.empty:
        return pd.DataFrame()
    keep = ["REPORT_DATE", "NOTICE_DATE", "CURRENCY_ABBR", "REPORT_TYPE",
            "OPERATE_INCOME", "GROSS_PROFIT", "PARENT_HOLDER_NETPROFIT",
            "GROSS_PROFIT_RATIO", "ROA", "ROE_AVG", "DEBT_ASSET_RATIO",
            "TOTAL_ASSETS_TR", "CURRENT_RATIO", "OCF_LIQDEBT"]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    for c in ("REPORT_DATE", "NOTICE_DATE"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").astype("datetime64[ns]")
    for c in cols:
        if c not in ("REPORT_DATE", "NOTICE_DATE", "CURRENCY_ABBR", "REPORT_TYPE"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE")


def _fetch_us_report_all_periods(sym: str, indicator_name: str) -> pd.DataFrame:
    import akshare as ak
    parts = []
    for per in ("年报", "单季报", "累计季报"):
        try:
            df = _retry_ak(lambda per=per: ak.stock_financial_us_report_em(symbol=sym, indicator=indicator_name, period=per),
                           tries=2, sleep_base=0.35)
            if df is not None and len(df):
                sub = df[["报告期", "项目名称", "金额"]].copy()
                sub["period_src"] = per
                parts.append(sub)
        except Exception:  # noqa
            continue
        time.sleep(0.05)
    if not parts:
        return pd.DataFrame()
    allx = pd.concat(parts, ignore_index=True)
    allx["报告期"] = pd.to_datetime(allx["报告期"], errors="coerce").astype("datetime64[ns]")
    allx["金额"] = pd.to_numeric(allx["金额"], errors="coerce")
    allx["项目名称"] = allx["项目名称"].astype(str).str.strip()
    allx = allx.dropna(subset=["报告期", "项目名称", "金额"])
    return allx.drop_duplicates(["报告期", "项目名称", "period_src"], keep="last").sort_values(["报告期", "项目名称"])


def select_stage1_symbols(top_n: int = DEFAULT_TOP_N, include_adr: bool = True) -> list:
    if not UNIVERSE_FILE.exists():
        return []
    u = pd.read_csv(UNIVERSE_FILE)
    if not include_adr and "classify" in u.columns:
        u = u[u["classify"] == "EQ"]
    u = u.sort_values("market_cap", ascending=False)
    if top_n and top_n > 0:
        u = u.head(int(top_n))
    return u["symbol"].astype(str).tolist()


def update_stage1(top_n: int = DEFAULT_TOP_N, include_adr: bool = True, force_fin: bool = False,
                  progress: Progress = None) -> list:
    if not UNIVERSE_FILE.exists():
        update_universe()
    syms = select_stage1_symbols(top_n=top_n, include_adr=include_adr)
    if not syms:
        return ["【Stage 1】股票池为空"]
    us_last = pd.to_datetime(latest_us_trade_day())

    px_ok = px_skip = px_fail = 0
    ind_ok = ind_skip = ind_fail = 0
    bs_ok = bs_skip = bs_fail = 0
    total = len(syms)

    for i, sym in enumerate(syms, 1):
        fn = _safe_sym_fname(sym)
        # 1) 前复权日线
        pf = DIRS["prices"] / f"{fn}.parquet"
        need_px = True
        if pf.exists():
            if _age_hours(pf) < PX_REFRESH_HOURS:
                need_px = False
            else:
                try:
                    dmax = _read_pq(pf, columns=["date"])["date"].max()
                    if pd.notna(dmax) and pd.Timestamp(dmax) >= us_last:
                        need_px = False
                except Exception:  # noqa
                    pass
        if need_px:
            try:
                df_p = _fetch_us_price_one(sym)
                if len(df_p):
                    _to_pq(df_p, pf)
                    px_ok += 1
                else:
                    px_fail += 1
            except Exception:  # noqa
                px_fail += 1
        else:
            px_skip += 1

        # 2) 单季报财务指标
        inf = DIRS["indicators"] / f"{fn}.parquet"
        need_ind = force_fin or (not inf.exists()) or (_age_hours(inf) >= FIN_REFRESH_DAYS * 24)
        latest_rep = None
        if need_ind:
            try:
                df_i = _fetch_us_indicator_one(sym)
                if len(df_i):
                    _to_pq(df_i, inf)
                    ind_ok += 1
                    latest_rep = pd.Timestamp(df_i["REPORT_DATE"].max())
                else:
                    ind_fail += 1
            except Exception:  # noqa
                ind_fail += 1
            time.sleep(0.06)
        else:
            ind_skip += 1
            try:
                latest_rep = pd.Timestamp(_read_pq(inf, columns=["REPORT_DATE"])["REPORT_DATE"].max())
            except Exception:  # noqa
                pass

        # 3) 资产负债表
        bsf = DIRS["balance"] / f"{fn}.parquet"
        need_bs = force_fin or (not bsf.exists())
        if not need_bs and latest_rep is not None:
            try:
                bs_max = pd.Timestamp(_read_pq(bsf, columns=["报告期"])["报告期"].max())
                if pd.isna(bs_max) or bs_max < latest_rep:
                    need_bs = True
            except Exception:  # noqa
                need_bs = True
        if need_bs:
            try:
                df_b = _fetch_us_report_all_periods(sym, "资产负债表")
                if len(df_b):
                    _to_pq(df_b, bsf)
                    bs_ok += 1
                else:
                    bs_fail += 1
            except Exception:  # noqa
                bs_fail += 1
            time.sleep(0.06)
        else:
            bs_skip += 1

        if progress:
            progress(i / total, f"Stage 1 {sym} ({i}/{total}) | 日线+{px_ok} 指标+{ind_ok} 负债+{bs_ok}")

    return [
        f"【Stage 1】覆盖 Top {len(syms)} 只美股：",
        f"  · 日线(Sina前复权): 更新 {px_ok}，已是最新跳过 {px_skip}，失败 {px_fail}",
        f"  · 财务指标(单季报): 更新 {ind_ok}，缓存有效跳过 {ind_skip}，失败 {ind_fail}",
        f"  · 资产负债表: 更新 {bs_ok}，已有最新财报跳过 {bs_skip}，失败 {bs_fail}",
    ]


def update_stage2(top_n: int = DEFAULT_TOP_N, include_adr: bool = True, excl_small_pct: float = 0.30,
                  cheap_pct: float = 0.10, amt_min_musd: float = 5.0, allow_neg_pb: bool = True,
                  force_cf: bool = False, progress: Progress = None) -> list:
    import us_value_screen_core as uvc

    if progress:
        progress(0.05, "计算估值分位，筛选 Stage 2 候选名单...")
    try:
        p = uvc.Params(
            top_n=int(top_n),
            excl_small_pct=float(excl_small_pct),
            cheap_pct=float(cheap_pct),
            amt_min_musd=float(amt_min_musd),
            include_adr=bool(include_adr),
            allow_neg_pb=bool(allow_neg_pb),
        )
        cheap_syms = uvc.get_cheap_symbols(p)
    except Exception as e:  # noqa
        return [f"【Stage 2】无法生成便宜候选列表：{e}（请先完成 Stage 1 数据下载）"]

    if not cheap_syms:
        return ["【Stage 2】没有满足估值与流动性门槛的便宜候选股"]

    ok = skip = fail = 0
    total = len(cheap_syms)
    for i, sym in enumerate(cheap_syms, 1):
        fn = _safe_sym_fname(sym)
        cff = DIRS["cashflow"] / f"{fn}.parquet"
        need = force_cf or (not cff.exists()) or (_age_hours(cff) >= FIN_REFRESH_DAYS * 24)
        if not need:
            try:
                inf = DIRS["indicators"] / f"{fn}.parquet"
                if inf.exists():
                    rep_max = pd.Timestamp(_read_pq(inf, columns=["REPORT_DATE"])["REPORT_DATE"].max())
                    cf_max = pd.Timestamp(_read_pq(cff, columns=["报告期"])["报告期"].max())
                    if pd.notna(rep_max) and (pd.isna(cf_max) or cf_max < rep_max):
                        need = True
            except Exception:  # noqa
                pass
        if need:
            try:
                df_c = _fetch_us_report_all_periods(sym, "现金流量表")
                if len(df_c):
                    _to_pq(df_c, cff)
                    ok += 1
                else:
                    fail += 1
            except Exception:  # noqa
                fail += 1
            time.sleep(0.06)
        else:
            skip += 1
        if progress:
            progress(i / total, f"Stage 2 现金流量表 {sym} ({i}/{total})")

    return [f"【Stage 2】最便宜候选 {total} 只：现金流量表更新 {ok}，缓存有效跳过 {skip}，失败 {fail}"]


def update_all(top_n: int = DEFAULT_TOP_N, include_adr: bool = True, excl_small_pct: float = 0.30,
               cheap_pct: float = 0.10, amt_min_musd: float = 5.0, allow_neg_pb: bool = True,
               progress: Progress = None) -> list:
    steps = [
        ("宏观(VIX/JOLTS)", lambda p: update_macro(p)),
        ("股票池(Nasdaq∩Tushare)", lambda p: update_universe(p)),
        ("Stage1(日线/指标/资产负债)", lambda p: update_stage1(top_n, include_adr, False, p)),
        ("Stage2(便宜候选现金流)", lambda p: update_stage2(top_n, include_adr, excl_small_pct, cheap_pct, amt_min_musd, allow_neg_pb, False, p)),
    ]
    log = []
    for i, (label, fn) in enumerate(steps):
        sub = (lambda frac, text, i=i, label=label: progress((i + frac) / len(steps), f"{label} · {text}")) if progress else None
        try:
            log += fn(sub)
        except Exception as e:  # noqa
            log.append(f"【{label}】中止：{e!r}（已保存分片保留，重跑自动续传）")
        gc.collect()
    return log


# ───────────────────────── 状态检查（零 pyarrow 读取！）─────────────────────────
def _count_files(key: str) -> int:
    d = DIRS[key]
    return len(list(d.glob("*.parquet"))) if d.exists() else 0


def missing_required() -> list:
    miss = []
    if not UNIVERSE_FILE.exists():
        miss.append("股票池")
    if _count_files("prices") < 20:
        miss.append("日线(Stage 1)")
    if _count_files("indicators") < 20:
        miss.append("财务指标(Stage 1)")
    return miss


def data_status() -> pd.DataFrame:
    rows = []
    if VIX_FILE.exists():
        v = pd.read_csv(VIX_FILE, usecols=["trade_date"])
        rows.append({"数据集": "宏观 VIX & JOLTS", "用途": "市场恐慌与就业背景", "数量": f"VIX {len(v)} 日",
                     "最新": str(pd.Timestamp(v["trade_date"].max()).date()), "状态": "✅"})
    else:
        rows.append({"数据集": "宏观 VIX & JOLTS", "用途": "市场恐慌与就业背景", "数量": "0", "最新": "—", "状态": "⚪ 可选"})

    if UNIVERSE_FILE.exists():
        u = pd.read_csv(UNIVERSE_FILE, usecols=["symbol"])
        rows.append({"数据集": "股票池 (Nasdaq∩Tushare)", "用途": "市值 / 行业 / EQ&ADR（必需）",
                     "数量": f"{len(u)} 只", "最新": dt.date.fromtimestamp(UNIVERSE_FILE.stat().st_mtime).isoformat(), "状态": "✅"})
    else:
        rows.append({"数据集": "股票池 (Nasdaq∩Tushare)", "用途": "市值 / 行业 / EQ&ADR（必需）",
                     "数量": "0", "最新": "—", "状态": "❌ 未下载"})

    for label, key, role in (
        ("Stage 1 日线 (Sina前复权)", "prices", "5年走势与历史最低价（必需）"),
        ("Stage 1 财务指标 (单季报)", "indicators", "PE_TTM 与 F-score（必需）"),
        ("Stage 1 资产负债表", "balance", "PB 与总资产（必需）"),
        ("Stage 2 现金流量表", "cashflow", "CFO / 分红 / 回购（仅便宜候选）"),
    ):
        n = _count_files(key)
        st_ = "✅" if n >= 20 else ("⚪ 建议下载" if key == "cashflow" else "❌ 未下载")
        d = DIRS[key]
        latest = dt.date.fromtimestamp(max(p.stat().st_mtime for p in d.glob("*.parquet"))).isoformat() if n else "—"
        rows.append({"数据集": label, "用途": role, "数量": f"{n} 只", "最新": latest, "状态": st_})
    return pd.DataFrame(rows)


# ───────────────────────── 非阻塞后台子进程管理 ─────────────────────────
def _subproc_env() -> dict:
    import importlib.util
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               ARROW_USER_SIMD_LEVEL="NONE")
    if importlib.util.find_spec("fastparquet") is not None:
        env["VS_PARQUET_ENGINE"] = "fastparquet"
    return env


def _write_prog(d: dict):
    DIRS["logs"].mkdir(parents=True, exist_ok=True)
    tmp = PROG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROG_FILE)


def read_progress() -> dict:
    if not PROG_FILE.exists():
        return {}
    try:
        return json.loads(PROG_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa
        return {}


def is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        except Exception:  # noqa
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def start_background_update(label: str, step: str = "all", top_n: int = DEFAULT_TOP_N, include_adr: bool = True,
                            excl_small_pct: float = 0.30, cheap_pct: float = 0.10,
                            amt_min_musd: float = 5.0, allow_neg_pb: bool = True):
    import subprocess
    import sys

    here = Path(__file__).resolve()
    DIRS["logs"].mkdir(parents=True, exist_ok=True)
    _write_prog({
        "state": "running", "label": label, "step": step,
        "frac": 0.0, "text": f"{label}：后台启动中…",
        "started_at": time.time(), "updated_at": time.time(), "logs": [], "pid": None,
    })
    cmd = [
        sys.executable, "-u", str(here),
        "--step", str(step),
        "--top-n", str(int(top_n)),
        "--excl", str(excl_small_pct),
        "--cheap", str(cheap_pct),
        "--amt-min", str(amt_min_musd),
    ]
    if include_adr:
        cmd.append("--include-adr")
    if allow_neg_pb:
        cmd.append("--allow-neg-pb")

    lf = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(here.parent), env=_subproc_env())
    cur = read_progress()
    cur["pid"] = proc.pid
    _write_prog(cur)
    return proc


def stop_background_update(proc=None):
    import subprocess
    cur = read_progress()
    pid = proc.pid if proc is not None else cur.get("pid")
    if proc is not None and proc.poll() is None:
        proc.terminate()
    if pid and is_pid_alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            try:
                import signal
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
    cur["state"] = "stopped"
    cur["text"] = "已手动中止（已保存的分片保留，下次可断点续传）"
    cur["updated_at"] = time.time()
    _write_prog(cur)


def read_log_tail(n: int = 25) -> str:
    if not LOG_FILE.exists():
        return ""
    try:
        return "\n".join(LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:  # noqa
        return ""


def _cli(argv=None) -> int:
    import argparse
    import traceback

    ap = argparse.ArgumentParser(description="us_value_screen_data 子进程更新入口")
    ap.add_argument("--step", default="all")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--excl", type=float, default=0.30)
    ap.add_argument("--cheap", type=float, default=0.10)
    ap.add_argument("--amt-min", type=float, default=5.0)
    ap.add_argument("--include-adr", action="store_true")
    ap.add_argument("--allow-neg-pb", action="store_true")
    a = ap.parse_args(argv)

    last_flush = [0.0]

    def cb(frac, text):
        now = time.time()
        f_val = min(max(float(frac), 0.0), 1.0)
        if now - last_flush[0] >= 0.8 or f_val >= 0.999:
            cur = read_progress()
            cur.update({
                "state": "running", "pid": os.getpid(),
                "frac": f_val, "text": text, "updated_at": now,
            })
            _write_prog(cur)
            last_flush[0] = now
        print(f"[{f_val * 100:5.1f}%] {text}", flush=True)

    try:
        if a.step == "macro":
            logs = update_macro(cb)
        elif a.step == "universe":
            logs = update_universe(cb)
        elif a.step == "stage1":
            logs = update_stage1(a.top_n, a.include_adr, False, cb)
        elif a.step == "stage2":
            logs = update_stage2(a.top_n, a.include_adr, a.excl, a.cheap, a.amt_min, a.allow_neg_pb, False, cb)
        else:
            logs = update_all(a.top_n, a.include_adr, a.excl, a.cheap, a.amt_min, a.allow_neg_pb, cb)
        cur = read_progress()
        cur.update({"state": "done", "frac": 1.0, "text": "更新完成", "updated_at": time.time(), "logs": logs})
        _write_prog(cur)
        return 0
    except Exception as e:  # noqa
        tb = traceback.format_exc()
        print(tb, flush=True)
        cur = read_progress()
        cur.update({"state": "error", "text": f"出错：{e}", "updated_at": time.time(), "logs": [tb]})
        _write_prog(cur)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
'''

US_VIEW_NEW = r'''"""美股价值 + 质量选股页：宏观背景(VIX/JOLTS) → 后台断点续传下载 → 选股结果与决策卡。"""
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
'''


def patch_core(src: str) -> str:
    # 1. UNIVERSE_FILE 改为读 CSV
    src = src.replace("u = _read_pq(ud.UNIVERSE_FILE)", "u = pd.read_csv(ud.UNIVERSE_FILE)")
    # 2. _env_attempts 优先用 fastparquet + 关闭 SIMD，避免先撞 pyarrow 崩溃
    old_env = '''def _env_attempts() -> list:
    import importlib.util
    seq = [("默认设置", {}), ("关闭 Arrow SIMD 加速", {"ARROW_USER_SIMD_LEVEL": "NONE"})]
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎", {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    return seq'''
    new_env = '''def _env_attempts() -> list:
    import importlib.util
    seq = []
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎", {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    seq.append(("关闭 Arrow SIMD 加速", {"ARROW_USER_SIMD_LEVEL": "NONE"}))
    seq.append(("默认设置", {}))
    return seq'''
    if old_env in src:
        src = src.replace(old_env, new_env, 1)
    return src


def main():
    p_data = ROOT / "us_value_screen_data.py"
    p_core = ROOT / "us_value_screen_core.py"
    p_view = ROOT / "views" / "us_value_screen.py"
    for p in (p_data, p_core, p_view):
        if not p.exists():
            sys.exit(f"[中止] 找不到 {p}，请先确认已运行 apply_us_value_screen_patch.py。")

    core_new = patch_core(p_core.read_text(encoding="utf-8-sig"))
    compile(US_DATA_NEW, str(p_data), "exec")
    compile(core_new, str(p_core), "exec")
    compile(US_VIEW_NEW, str(p_view), "exec")

    for p, content in ((p_data, US_DATA_NEW), (p_core, core_new), (p_view, US_VIEW_NEW)):
        bak = p.with_suffix(p.suffix + ".bak_before_bg")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(content, encoding="utf-8")
        print(f"[更新] {p.relative_to(ROOT)}（备份：{bak.name}）")

    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("现在在「🇺🇸 美股价值选股」页：")
    print("  · 主进程不再调用 pyarrow 读写 parquet；")
    print("  · 点击更新按钮会在后台子进程运行，页面每2秒自动刷新进度条与日志，可随时「⏹ 停止任务」或切换页面；")
    print("  · 也可以直接在命令行后台跑首次全量下载：python us_value_screen_data.py --step all --top-n 1000 --include-adr --allow-neg-pb")


if __name__ == "__main__":
    main()