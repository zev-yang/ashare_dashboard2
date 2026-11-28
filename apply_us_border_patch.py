# -*- coding: utf-8 -*-
"""一键添加「🇺🇸 美股价值选股」完整模块（数据下载 + 计算引擎 + Streamlit 页面 + 导航注册）。

包含文件：
  1. 新建 us_value_screen_data.py：Nasdaq Screener ∩ Tushare us_basic + Sina前复权日线 + Eastmoney美股财报 + VIX/JOLTS；
  2. 新建 us_value_screen_core.py：5年PE/PB分位(总市值/总利润避开EPS拆股与ADR比率陷阱)、行业+市值中性化、TTM F-score、最新价与历史最低价；
  3. 新建 views/us_value_screen.py：美股选股页面（含宏观面板、分级下载按钮、参数区、结果表、CSV导出）；
  4. 更新 app.py：注册「🇺🇸 美股价值选股」页面。

用法（在项目根目录执行）：
    python apply_us_value_screen_patch.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

US_DATA_PY = r'''# -*- coding: utf-8 -*-
"""美股价值 + 质量选股 · 数据层。

数据源组合（绕开 Tushare us_daily 1次/小时限制，零付费增补接口）：
  · 宏观政体：Tushare vix_index (VIX) + us_job_offers (JOLTS 职位空缺)
  · 股票池与GICS行业：Nasdaq Screener API ∩ Tushare us_basic (EQ/ADR，剔除ETF/优先股/OTC)
  · 历史日线(前复权)：ak.stock_us_hist_sina(symbol, adjust="qfq")
  · Stage 1 财务指标与资产负债：ak.stock_financial_us_analysis_indicator_em("单季报") + stock_financial_us_report_em("资产负债表")
  · Stage 2 现金流/分红/回购：仅针对最便宜 decile (~100只) 拉取 stock_financial_us_report_em("现金流量表")

落盘目录：data/cache/us_value_screen/
"""
from __future__ import annotations

import datetime as dt
import gc
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
}
UNIVERSE_FILE = DIRS["universe"] / "us_universe.parquet"
VIX_FILE = DIRS["macro"] / "vix.parquet"
JOLTS_FILE = DIRS["macro"] / "jolts.parquet"
CAL_FILE = DIRS["macro"] / "us_tradecal.parquet"

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
    df.to_parquet(path, index=False, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))


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


# ───────────────────────── 宏观与美股交易日历 ─────────────────────────
def latest_us_trade_day() -> str:
    """获取美股最近一个开市日 YYYYMMDD（优先缓存的 us_tradecal）。"""
    today = dt.date.today()
    if CAL_FILE.exists() and _age_hours(CAL_FILE) < 24:
        try:
            cal = _read_pq(CAL_FILE)
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
                _to_pq(df, CAL_FILE)
                op = df[df["is_open"].astype(str) == "1"]["cal_date"].astype(str)
                past = op[op <= today.strftime("%Y%m%d")]
                if len(past):
                    return str(past.max())
        except Exception:  # noqa
            pass
    return (today - dt.timedelta(days=1)).strftime("%Y%m%d")


def update_macro(progress: Progress = None) -> list:
    """更新 VIX 恐慌指数与 JOLTS 职位空缺（Tushare，约2次调用）。"""
    if not ts_src.available():
        return ["【美股宏观】未配置 Tushare token，跳过 VIX / JOLTS"]
    log = []
    start_vix = (dt.date.today() - dt.timedelta(days=365 * 6)).strftime("%Y%m%d")
    try:
        vix = ts_src.call("vix_index", start_date=start_vix)
        if vix is not None and len(vix):
            vix["trade_date"] = pd.to_datetime(vix["trade_date"].astype(str), errors="coerce")
            vix = vix.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
            _to_pq(vix, VIX_FILE)
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
            _to_pq(jolts, JOLTS_FILE)
            log.append(f"【JOLTS】{len(jolts)} 行，最新 {jolts['ts_date'].max().date()}")
        else:
            log.append("【JOLTS】返回为空")
    except Exception as e:  # noqa
        log.append(f"【JOLTS】失败：{str(e)[:120]}")
    if progress:
        progress(1.0, "宏观数据完成")
    return log


# ───────────────────────── 股票池与GICS行业 ─────────────────────────
def update_universe(progress: Progress = None) -> list:
    """Nasdaq Screener API ∩ Tushare us_basic (EQ/ADR) → 清洗后的美股投资域。"""
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
    _to_pq(df, UNIVERSE_FILE)
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
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
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
            out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in cols:
        if c not in ("REPORT_DATE", "NOTICE_DATE", "CURRENCY_ABBR", "REPORT_TYPE"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE")


def _fetch_us_report_all_periods(sym: str, indicator_name: str) -> pd.DataFrame:
    """拉取美股资产负债表或现金流量表的 年报 + 单季报 + 累计季报 并合并去重。"""
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
    allx["报告期"] = pd.to_datetime(allx["报告期"], errors="coerce")
    allx["金额"] = pd.to_numeric(allx["金额"], errors="coerce")
    allx["项目名称"] = allx["项目名称"].astype(str).str.strip()
    allx = allx.dropna(subset=["报告期", "项目名称", "金额"])
    return allx.drop_duplicates(["报告期", "项目名称", "period_src"], keep="last").sort_values(["报告期", "项目名称"])


def select_stage1_symbols(top_n: int = DEFAULT_TOP_N, include_adr: bool = True) -> list:
    if not UNIVERSE_FILE.exists():
        return []
    u = _read_pq(UNIVERSE_FILE)
    if not include_adr and "classify" in u.columns:
        u = u[u["classify"] == "EQ"]
    u = u.sort_values("market_cap", ascending=False)
    if top_n and top_n > 0:
        u = u.head(int(top_n))
    return u["symbol"].astype(str).tolist()


def update_stage1(top_n: int = DEFAULT_TOP_N, include_adr: bool = True, force_fin: bool = False,
                  progress: Progress = None) -> list:
    """Stage 1 增量更新：日线(Sina前复权) + 财务指标(单季报) + 资产负债表(仅新财报期才重拉)。"""
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
            df_p = _fetch_us_price_one(sym)
            if len(df_p):
                _to_pq(df_p, pf)
                px_ok += 1
            else:
                px_fail += 1
        else:
            px_skip += 1

        # 2) 单季报财务指标
        inf = DIRS["indicators"] / f"{fn}.parquet"
        need_ind = force_fin or (not inf.exists()) or (_age_hours(inf) >= FIN_REFRESH_DAYS * 24)
        latest_rep = None
        if need_ind:
            df_i = _fetch_us_indicator_one(sym)
            if len(df_i):
                _to_pq(df_i, inf)
                ind_ok += 1
                latest_rep = pd.Timestamp(df_i["REPORT_DATE"].max())
            else:
                ind_fail += 1
            time.sleep(0.06)
        else:
            ind_skip += 1
            try:
                latest_rep = pd.Timestamp(_read_pq(inf, columns=["REPORT_DATE"])["REPORT_DATE"].max())
            except Exception:  # noqa
                pass

        # 3) 资产负债表（仅缺文件或指标有了更新报告期时才拉）
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
            df_b = _fetch_us_report_all_periods(sym, "资产负债表")
            if len(df_b):
                _to_pq(df_b, bsf)
                bs_ok += 1
            else:
                bs_fail += 1
            time.sleep(0.06)
        else:
            bs_skip += 1

        if progress and (i % 5 == 0 or i == total):
            progress(i / total, f"Stage 1 {sym} ({i}/{total})")

    return [
        f"【Stage 1】覆盖 Top {len(syms)} 只美股：",
        f"  · 日线(Sina前复权): 更新 {px_ok}，已是最新跳过 {px_skip}，失败 {px_fail}",
        f"  · 财务指标(单季报): 更新 {ind_ok}，缓存有效跳过 {ind_skip}，失败 {ind_fail}",
        f"  · 资产负债表: 更新 {bs_ok}，已有最新财报跳过 {bs_skip}，失败 {bs_fail}",
    ]


def update_stage2(top_n: int = DEFAULT_TOP_N, include_adr: bool = True, excl_small_pct: float = 0.30,
                  cheap_pct: float = 0.10, amt_min_musd: float = 5.0, allow_neg_pb: bool = True,
                  force_cf: bool = False, progress: Progress = None) -> list:
    """Stage 2：先快速跑一次估值排序取出最便宜 decile，再仅为这些候选股拉取现金流量表。"""
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
            df_c = _fetch_us_report_all_periods(sym, "现金流量表")
            if len(df_c):
                _to_pq(df_c, cff)
                ok += 1
            else:
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


# ───────────────────────── 状态检查 ─────────────────────────
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
        v = _read_pq(VIX_FILE, columns=["trade_date"])
        rows.append({"数据集": "宏观 VIX & JOLTS", "用途": "市场恐慌与就业背景", "数量": f"VIX {len(v)} 日",
                     "最新": str(pd.Timestamp(v["trade_date"].max()).date()), "状态": "✅"})
    else:
        rows.append({"数据集": "宏观 VIX & JOLTS", "用途": "市场恐慌与就业背景", "数量": "0", "最新": "—", "状态": "⚪ 可选"})

    if UNIVERSE_FILE.exists():
        u = _read_pq(UNIVERSE_FILE, columns=["symbol"])
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


# ───────────────────────── 子进程隔离执行 ─────────────────────────
def _env_attempts() -> list:
    import importlib.util
    seq = [("默认设置", {}), ("关闭 Arrow SIMD 加速", {"ARROW_USER_SIMD_LEVEL": "NONE"})]
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎", {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    return seq


_GOOD_ENV: Optional[dict] = None


def run_update_in_subprocess(step: str = "all", top_n: int = DEFAULT_TOP_N, include_adr: bool = True,
                             excl_small_pct: float = 0.30, cheap_pct: float = 0.10,
                             amt_min_musd: float = 5.0, allow_neg_pb: bool = True,
                             progress: Progress = None, timeout: int = 7200) -> list:
    global _GOOD_ENV
    import pickle
    import subprocess
    import sys
    import tempfile

    here = Path(__file__).resolve()
    attempts = [("上次成功的设置", _GOOD_ENV)] if _GOOD_ENV is not None else _env_attempts()
    tried, last_err = [], ""
    for label, extra in attempts:
        with tempfile.TemporaryDirectory() as td:
            out_pkl = Path(td) / "us_upd_log.pkl"
            cmd = [
                sys.executable, "-u", str(here),
                "--step", str(step), "--out", str(out_pkl),
                "--top-n", str(int(top_n)),
                "--excl", str(excl_small_pct),
                "--cheap", str(cheap_pct),
                "--amt-min", str(amt_min_musd),
            ]
            if include_adr:
                cmd.append("--include-adr")
            if allow_neg_pb:
                cmd.append("--allow-neg-pb")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            env.update(extra)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(here.parent), env=env,
            )
            tail = []
            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip("\r\n")
                    if line.startswith("PROGRESS:"):
                        parts = line.split(":", 2)
                        if len(parts) == 3 and progress:
                            try:
                                progress(float(parts[1]), parts[2])
                            except Exception:  # noqa
                                pass
                    elif line:
                        tail.append(line)
                        if len(tail) > 40:
                            tail.pop(0)
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return [f"【更新超时】超过 {timeout} 秒未完成（已下载的分片保留，再点一次续传）"]

            if rc == 0 and out_pkl.exists():
                with open(out_pkl, "rb") as f:
                    logs = pickle.load(f)
                _GOOD_ENV = dict(extra)
                if tried:
                    logs.insert(0, f"⚠ 前 {len(tried)} 次因底层库崩溃失败（{'、'.join(tried)}），已改用：{label}")
                return logs

            tried.append(label)
            last_err = f"子进程退出码 {rc}\n" + "\n".join(tail[-20:])
            if rc not in NATIVE_CRASH:
                break
    _GOOD_ENV = None
    return [f"【更新中断】已尝试：{' → '.join(tried)}（已保存分片保留，再点一次会续传）。\n{last_err}"]


def _cli(argv=None) -> int:
    import argparse
    import pickle
    import traceback

    ap = argparse.ArgumentParser(description="us_value_screen_data 子进程更新入口")
    ap.add_argument("--step", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--excl", type=float, default=0.30)
    ap.add_argument("--cheap", type=float, default=0.10)
    ap.add_argument("--amt-min", type=float, default=5.0)
    ap.add_argument("--include-adr", action="store_true")
    ap.add_argument("--allow-neg-pb", action="store_true")
    a = ap.parse_args(argv)

    def cb(frac, text):
        print(f"PROGRESS:{min(max(float(frac), 0.0), 1.0):.4f}:{text}", flush=True)

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
    except Exception:  # noqa
        traceback.print_exc()
        return 1
    with open(a.out, "wb") as f:
        pickle.dump(logs, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
'''

US_CORE_PY = r'''# -*- coding: utf-8 -*-
"""美股价值 + 质量选股 · 计算层。

设计要点（针对美股会计与数据源陷阱定制）：
  1. 用「总市值 / 归母净利润TTM」和「总市值 / 归属母公司股东权益」计算 PE_TTM 与 PB：
     完全避开东财 BASIC_EPS 历史拆股断层（如 NVDA 2023-04 未复权 vs 2023-07 10:1复权）以及中概股 ADS 比例问题（如 BABA 1:8）；
  2. 处理东财比较期重述造成的 NOTICE_DATE 回填陷阱：若 NOTICE_DATE - REPORT_DATE > 120天，历史回溯改用 REPORT_DATE + 45天；
  3. 货币折算：USD=1，CNY 复用代理缓存里的 USDCNH 序列（无则7.2），其余主流ADR货币按汇率映射折算成 USD；
  4. 负PB兼容：美股大量优质回购股（AAPL/MCD/HD等）净资产为负，勾选 allow_neg_pb 时自动回退为仅用 PE 5年分位；
  5. GICS 行业 + ln(市值) OLS 中性化；
  6. 7项 TTM F-score + 现金流正年数 + 分红年数 + 净回购年数 + 股东回报率% + 最新价($) + 历史最低价(前复权$)。
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
import us_value_screen_data as ud

PCT_WIN = 1250
PCT_MINP = 252
CRIT = ["ROA>0", "CFO>0", "ΔROA>0", "CFO>NI", "Δ负债<0", "Δ毛利>0", "Δ周转>0"]

STATIC_FX_PER_USD = {
    "USD": 1.0, "HKD": 7.78, "EUR": 0.92, "GBP": 0.79, "CHF": 0.88, "CAD": 1.38,
    "AUD": 1.53, "JPY": 150.0, "TWD": 32.0, "KRW": 1380.0, "DKK": 6.90, "SEK": 10.6,
    "NOK": 10.8, "SGD": 1.34, "BRL": 5.60, "INR": 84.0, "ILS": 3.70, "MXN": 19.5, "ZAR": 18.0,
}


class DataMissing(RuntimeError):
    pass


@dataclass
class Params:
    top_n: int = 1000
    excl_small_pct: float = 0.30
    cheap_pct: float = 0.10
    fscore_min: int = 5
    amt_min_musd: float = 5.0
    include_adr: bool = True
    allow_neg_pb: bool = True


@dataclass
class Result:
    candidates: pd.DataFrame
    asof: Optional[pd.Timestamp]
    log: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    csv_path: Optional[Path] = None


def _read_pq(path: Path, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))


def latest_pct(w: pd.DataFrame, win: int = PCT_WIN, minp: int = PCT_MINP) -> pd.Series:
    tail = w.iloc[-win:]
    last = tail.iloc[-1]
    n = tail.notna().sum()
    less = tail.lt(last, axis=1).sum()
    eq = tail.eq(last, axis=1).sum()
    pct = (less + (eq + 1.0) / 2.0) / n
    return pct.where(last.notna() & (n >= minp))


def neutralize_one(y: pd.Series, ind_map: pd.Series, lncap: pd.Series) -> pd.Series:
    codes = y.index
    g = ind_map.reindex(codes).fillna("UNK").astype(str).to_numpy()
    lc = lncap.reindex(codes)
    lc = lc.fillna(lc.median()).to_numpy(float)
    yv = y.to_numpy(float)
    _, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    y_dm = yv - (np.bincount(inv, weights=yv) / cnt)[inv]
    l_dm = lc - (np.bincount(inv, weights=lc) / cnt)[inv]
    den = float((l_dm ** 2).sum())
    slope = float((l_dm * y_dm).sum() / den) if den > 1e-12 else 0.0
    return pd.Series(y_dm - slope * l_dm, index=codes)


def _load_usdcnh() -> pd.Series:
    p = Path(config.CACHE_DIR) / "proxy_daily.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p)
        sub = df[df["symbol"].isin(["USDCNH", "USDCNY_MID"])].copy()
        if sub.empty:
            return pd.Series(dtype=float)
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce").astype("datetime64[ns]")
        sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
        sub = sub.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        return pd.Series(sub["close"].to_numpy(float), index=pd.DatetimeIndex(sub["date"]))
    except Exception:  # noqa
        return pd.Series(dtype=float)


def _fx_series_for_dates(curr: str, dates: pd.DatetimeIndex, usdcnh: pd.Series) -> pd.Series:
    c = str(curr).upper().strip()
    if c == "CNY" and len(usdcnh):
        s = usdcnh.reindex(usdcnh.index.union(dates)).sort_index().ffill().bfill().reindex(dates)
        return s.fillna(7.2)
    if c == "CNY":
        return pd.Series(7.2, index=dates)
    rate = STATIC_FX_PER_USD.get(c, np.nan)
    return pd.Series(rate, index=dates, dtype=float)


def _eff_avail_date(rep_dt: pd.Series, not_dt: pd.Series) -> pd.Series:
    """处理东财比较期公告日回填陷阱：若 NOTICE_DATE 晚于 REPORT_DATE 超过 120 天，改按 REPORT_DATE+45 天作历史可用日。"""
    gap = (not_dt - rep_dt).dt.days
    use_rep = not_dt.isna() | (gap > 120) | (gap < 0)
    eff = not_dt.where(~use_rep, rep_dt + pd.Timedelta(days=45))
    return pd.to_datetime(eff).astype("datetime64[ns]")


def _extract_bs_series(bs_df: pd.DataFrame) -> pd.DataFrame:
    if bs_df is None or bs_df.empty:
        return pd.DataFrame()
    d = bs_df.copy()
    d["报告期"] = pd.to_datetime(d["报告期"], errors="coerce").astype("datetime64[ns]")
    d = d.dropna(subset=["报告期", "项目名称", "金额"])
    piv = d.pivot_table(index="报告期", columns="项目名称", values="金额", aggfunc="last").sort_index()
    eq_parent = piv["归属于母公司股东权益"] if "归属于母公司股东权益" in piv.columns else pd.Series(np.nan, index=piv.index)
    eq_tot = piv["股东权益合计"] if "股东权益合计" in piv.columns else pd.Series(np.nan, index=piv.index)
    assets = piv["总资产"] if "总资产" in piv.columns else pd.Series(np.nan, index=piv.index)
    liab = piv["总负债"] if "总负债" in piv.columns else pd.Series(np.nan, index=piv.index)
    out = pd.DataFrame({
        "REPORT_DATE": piv.index,
        "equity_raw": eq_parent.fillna(eq_tot),
        "total_assets": assets,
        "total_liab": liab,
    }).reset_index(drop=True)
    return out


def _build_stock_panel_and_ttm(sym: str, mv_now: float, usdcnh: pd.Series):
    """构建单只美股近5年日度 (PE_TTM, PB)、最新价格信息、以及按季度对齐的 TTM F-score 输入表。"""
    fn = ud._safe_sym_fname(sym)
    pf = ud.DIRS["prices"] / f"{fn}.parquet"
    inf = ud.DIRS["indicators"] / f"{fn}.parquet"
    bsf = ud.DIRS["balance"] / f"{fn}.parquet"
    if not (pf.exists() and inf.exists()):
        return None

    px = _read_pq(pf)
    if px.empty or len(px) < 60:
        return None
    px["date"] = pd.to_datetime(px["date"], errors="coerce").astype("datetime64[ns]")
    px = px.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    px_valid = px[px["close"] > 0].copy()
    if px_valid.empty:
        return None

    cur_px = float(px_valid["close"].iloc[-1])
    low_pos = px[px["low"] > 0]["low"]
    hist_low = float(low_pos.min()) if len(low_pos) else np.nan
    px_tail = px_valid.tail(PCT_WIN).copy()
    px_tail["dollar_vol"] = px_tail["close"] * px_tail["volume"]
    amt20_musd = float(px_tail["dollar_vol"].tail(20).mean() / 1e6) if len(px_tail) >= 5 else np.nan

    ind = _read_pq(inf)
    if ind.empty or "PARENT_HOLDER_NETPROFIT" not in ind.columns:
        return None
    ind = ind.copy()
    ind["REPORT_DATE"] = pd.to_datetime(ind["REPORT_DATE"], errors="coerce").astype("datetime64[ns]")
    ind["NOTICE_DATE"] = pd.to_datetime(ind["NOTICE_DATE"], errors="coerce").astype("datetime64[ns]") if "NOTICE_DATE" in ind.columns else pd.NaT
    ind = ind.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE").drop_duplicates("REPORT_DATE", keep="last").reset_index(drop=True)
    if len(ind) < 4:
        return None

    curr = str(ind["CURRENCY_ABBR"].dropna().iloc[-1]) if "CURRENCY_ABBR" in ind.columns and ind["CURRENCY_ABBR"].notna().any() else "USD"
    ind["avail_date"] = _eff_avail_date(ind["REPORT_DATE"], ind["NOTICE_DATE"])
    ind["np_ttm"] = ind["PARENT_HOLDER_NETPROFIT"].rolling(4, min_periods=4).sum()
    if "OPERATE_INCOME" in ind.columns:
        ind["rev_ttm"] = ind["OPERATE_INCOME"].rolling(4, min_periods=4).sum()
    else:
        ind["rev_ttm"] = np.nan
    if "GROSS_PROFIT" in ind.columns:
        ind["gp_ttm"] = ind["GROSS_PROFIT"].rolling(4, min_periods=4).sum()
    else:
        ind["gp_ttm"] = np.nan

    bs_tbl = _extract_bs_series(_read_pq(bsf)) if bsf.exists() else pd.DataFrame()
    if len(bs_tbl):
        bs_tbl["REPORT_DATE"] = pd.to_datetime(bs_tbl["REPORT_DATE"]).astype("datetime64[ns]")
        ind = pd.merge_asof(
            ind.sort_values("REPORT_DATE"),
            bs_tbl.sort_values("REPORT_DATE"),
            on="REPORT_DATE",
            direction="backward",
            tolerance=pd.Timedelta(days=375),
        )
    else:
        ind["equity_raw"] = np.nan
        ind["total_assets"] = np.nan
        ind["total_liab"] = np.nan

    # 补齐总资产/负债率兜底
    if "DEBT_ASSET_RATIO" not in ind.columns:
        ind["DEBT_ASSET_RATIO"] = np.nan
    miss_debt = ind["DEBT_ASSET_RATIO"].isna() & ind["total_liab"].notna() & (ind["total_assets"] > 0)
    ind.loc[miss_debt, "DEBT_ASSET_RATIO"] = ind.loc[miss_debt, "total_liab"] / ind.loc[miss_debt, "total_assets"] * 100.0

    ind["roa_ttm"] = np.where(ind["total_assets"] > 0, ind["np_ttm"] / ind["total_assets"] * 100.0, np.nan)
    ind["gm_ttm"] = np.where(ind["rev_ttm"] > 0, ind["gp_ttm"] / ind["rev_ttm"] * 100.0, np.nan)
    if "GROSS_PROFIT_RATIO" in ind.columns:
        ind["gm_ttm"] = ind["gm_ttm"].fillna(ind["GROSS_PROFIT_RATIO"])
    ind["at_ttm"] = np.where(ind["total_assets"] > 0, ind["rev_ttm"] / ind["total_assets"], np.nan)
    if "TOTAL_ASSETS_TR" in ind.columns:
        ind["at_ttm"] = ind["at_ttm"].fillna(ind["TOTAL_ASSETS_TR"])

    # 将可用季报对日线做 merge_asof
    q_for_px = ind[["avail_date", "np_ttm", "equity_raw"]].dropna(subset=["avail_date"]).sort_values("avail_date").drop_duplicates("avail_date", keep="last")
    if q_for_px.empty:
        return None
    daily_merged = pd.merge_asof(
        px_tail[["date", "close"]].sort_values("date"),
        q_for_px,
        left_on="date",
        right_on="avail_date",
        direction="backward",
    )
    dt_idx = pd.DatetimeIndex(daily_merged["date"])
    fx = _fx_series_for_dates(curr, dt_idx, usdcnh).to_numpy(float)
    mv_series = mv_now * (daily_merged["close"].to_numpy(float) / cur_px)
    np_usd = daily_merged["np_ttm"].to_numpy(float) / fx
    eq_usd = daily_merged["equity_raw"].to_numpy(float) / fx

    pe_s = pd.Series(np.where(np_usd > 0, mv_series / np_usd, np.nan), index=dt_idx)
    pb_s = pd.Series(np.where(eq_usd > 0, mv_series / eq_usd, np.nan), index=dt_idx)

    return {
        "pe": pe_s, "pb": pb_s, "ind": ind, "curr": curr,
        "cur_px": cur_px, "hist_low": hist_low, "amt20_musd": amt20_musd,
        "last_date": dt_idx[-1],
    }


def _stage1_rank(p: Params, log: list):
    if not ud.UNIVERSE_FILE.exists():
        raise DataMissing("股票池 (us_universe)")
    u = _read_pq(ud.UNIVERSE_FILE)
    if not p.include_adr and "classify" in u.columns:
        u = u[u["classify"] == "EQ"]
    u = u.sort_values("market_cap", ascending=False)
    if p.top_n and p.top_n > 0:
        u = u.head(int(p.top_n))
    u = u.set_index("symbol")

    usdcnh = _load_usdcnh()
    pe_dict, pb_dict, info_map = {}, {}, {}
    for sym, row in u.iterrows():
        res = _build_stock_panel_and_ttm(str(sym), float(row["market_cap"]), usdcnh)
        if res is None:
            continue
        pe_dict[str(sym)] = res["pe"]
        pb_dict[str(sym)] = res["pb"]
        info_map[str(sym)] = res

    if len(pe_dict) < 30:
        raise DataMissing("Stage 1 有效日线与财报数据不足 30 只，请先更新 Stage 1 数据")

    pe_wide = pd.DataFrame(pe_dict).sort_index().ffill(limit=5).iloc[-PCT_WIN:]
    pb_wide = pd.DataFrame(pb_dict).sort_index().ffill(limit=5).iloc[-PCT_WIN:]
    cur_date = pe_wide.index[-1]

    pe_pct = latest_pct(pe_wide.where(pe_wide > 0))
    pb_pct = latest_pct(pb_wide.where(pb_wide > 0))

    # 估值因子：要求 PE > 0；负 PB 回购股允许仅按 PE 分位计算
    val_both = 0.5 * (1.0 - pe_pct) + 0.5 * (1.0 - pb_pct)
    if p.allow_neg_pb:
        value_raw = val_both.where(pe_pct.notna() & pb_pct.notna(), 1.0 - pe_pct).where(pe_pct.notna()).dropna()
    else:
        value_raw = val_both.dropna()

    if len(value_raw) < 30:
        raise DataMissing("有效 PE/PB 5年分位不足 30 只（回看需至少 252 个交易日）")

    mc = u["market_cap"].reindex(value_raw.index).dropna()
    value_raw = value_raw.reindex(mc.index)
    lncap = np.log(mc.clip(lower=1.0))
    sector_map = u["sector"]
    value_neu = neutralize_one(value_raw, sector_map, lncap).rank(pct=True).dropna()

    # 域过滤：剔除最小 excl_small_pct + 20日均成交额低于阈值
    mc_dom = mc[mc >= mc.quantile(p.excl_small_pct)].index
    domain = [
        c for c in mc_dom
        if c in value_neu.index
        and (not np.isnan(info_map[c]["amt20_musd"]))
        and info_map[c]["amt20_musd"] >= p.amt_min_musd
    ]
    if len(domain) < 10:
        raise DataMissing(f"满足流动性门槛 (>=${p.amt_min_musd}M) 的股票仅 {len(domain)} 只")

    vd_ = value_neu.reindex(domain).dropna()
    vd_rank = vd_.rank(pct=True)
    cheap = vd_[vd_ >= vd_.quantile(1.0 - p.cheap_pct)].index.tolist()

    log.append(f"① 装载 {len(info_map)} 只美股（至 {cur_date.date()}）→ 域 {len(domain)} 只（剔小盘 {p.excl_small_pct:.0%} & 成交额 <${p.amt_min_musd}M）→ 最便宜 {p.cheap_pct:.0%}：{len(cheap)} 只")
    return {
        "u": u, "cur_date": cur_date, "pe_wide": pe_wide, "pb_wide": pb_wide,
        "pe_pct": pe_pct, "pb_pct": pb_pct, "vd_rank": vd_rank, "cheap": cheap,
        "info_map": info_map, "usdcnh": usdcnh, "n_domain": len(domain),
    }


def get_cheap_symbols(p: Optional[Params] = None) -> list:
    p = p or Params()
    return _stage1_rank(p, [])["cheap"]


def _cashflow_metrics(sym: str, today: pd.Timestamp, ind_df: pd.DataFrame, curr: str, mv_usd: float, usdcnh: pd.Series) -> dict:
    fn = ud._safe_sym_fname(sym)
    cff = ud.DIRS["cashflow"] / f"{fn}.parquet"
    out = {"cfo_cur": np.nan, "ni_cur": np.nan, "cf_pos_yrs": np.nan, "div_yrs": np.nan,
           "buyback_yrs": np.nan, "sh_yield_pct": np.nan}
    if not cff.exists():
        return out
    cf = _read_pq(cff)
    if cf.empty:
        return out
    cf["报告期"] = pd.to_datetime(cf["报告期"], errors="coerce").astype("datetime64[ns]")
    cf = cf[cf["报告期"] <= today]
    ann = cf[cf["period_src"] == "年报"].copy()
    if ann.empty:
        return out
    piv = ann.pivot_table(index="报告期", columns="项目名称", values="金额", aggfunc="last").sort_index()
    last3 = piv.tail(3)

    if "经营活动产生的现金流量净额" in last3.columns:
        cfo_s = last3["经营活动产生的现金流量净额"].dropna()
        out["cf_pos_yrs"] = int((cfo_s > 0).sum()) if len(cfo_s) else np.nan
        if len(cfo_s):
            out["cfo_cur"] = float(cfo_s.iloc[-1])
            latest_fy = cfo_s.index[-1]
            # 对齐同期年报归母净利润（ind_df 最近4季合计作为年报利润近似或精确对应年报日）
            sub_ind = ind_df[ind_df["REPORT_DATE"] <= latest_fy].sort_values("REPORT_DATE")
            if len(sub_ind) >= 4:
                out["ni_cur"] = float(sub_ind["PARENT_HOLDER_NETPROFIT"].tail(4).sum())

    if "股息支付" in last3.columns:
        dv = last3["股息支付"].dropna()
        out["div_yrs"] = int((dv.abs() > 0).sum()) if len(dv) else 0
    else:
        out["div_yrs"] = 0

    bb = last3["回购股份"].fillna(0) if "回购股份" in last3.columns else pd.Series(0.0, index=last3.index)
    iss = last3["发行股份"].fillna(0) if "发行股份" in last3.columns else pd.Series(0.0, index=last3.index)
    net_bb = -(bb + iss)
    out["buyback_yrs"] = int((net_bb > 0).sum()) if len(last3) else np.nan

    # 最新年报股东总回报率 % = (分红现金 + 净回购现金) / 总市值(USD)
    if len(last3) and mv_usd > 0:
        div_last = abs(float(last3["股息支付"].iloc[-1])) if ("股息支付" in last3.columns and pd.notna(last3["股息支付"].iloc[-1])) else 0.0
        bb_last = float(net_bb.iloc[-1])
        tot_ret_local = div_last + bb_last
        fx = float(_fx_series_for_dates(curr, pd.DatetimeIndex([last3.index[-1]]), usdcnh).iloc[-1])
        if fx > 0 and np.isfinite(fx):
            out["sh_yield_pct"] = round((tot_ret_local / fx) / mv_usd * 100.0, 2)
    return out


def _compute_fscore_one(sym: str, st_info: dict, today: pd.Timestamp, mv_usd: float, usdcnh: pd.Series) -> dict:
    ind = st_info["ind"]
    sub = ind[ind["NOTICE_DATE"].isna() | (ind["NOTICE_DATE"] <= today) | ((ind["NOTICE_DATE"] - ind["REPORT_DATE"]).dt.days > 120)].copy()
    sub = sub[sub["REPORT_DATE"] <= today].sort_values("REPORT_DATE").reset_index(drop=True)
    if len(sub) < 5:
        return {"fscore": 0, "detail": "·······", "period": "—", "roa": np.nan, "debt": np.nan, "cf_meta": {}}

    cur = sub.iloc[-1]
    cur_dt = cur["REPORT_DATE"]
    # 按报告期 −1 年（300~430天前最近一季）找上年同期，避免缺季度错位
    cand_py = sub[(sub["REPORT_DATE"] <= cur_dt - pd.Timedelta(days=300)) & (sub["REPORT_DATE"] >= cur_dt - pd.Timedelta(days=430))]
    py = cand_py.iloc[-1] if len(cand_py) else sub.iloc[-5]

    cf_meta = _cashflow_metrics(sym, today, sub, st_info["curr"], mv_usd, usdcnh)
    cfo_val = cf_meta.get("cfo_cur", np.nan)
    ni_val = cf_meta.get("ni_cur", np.nan)
    # 现金流表未下载时，兜底用 OCF_LIQDEBT > 0 判断 CFO>0
    if np.isnan(cfo_val) and "OCF_LIQDEBT" in cur.index and pd.notna(cur["OCF_LIQDEBT"]):
        cfo_pos = bool(cur["OCF_LIQDEBT"] > 0)
    else:
        cfo_pos = bool(cfo_val > 0) if not np.isnan(cfo_val) else False

    cfo_gt_ni = bool(cfo_val > ni_val) if (not np.isnan(cfo_val) and not np.isnan(ni_val)) else False

    checks = {
        "ROA>0": bool(pd.notna(cur["roa_ttm"]) and cur["roa_ttm"] > 0),
        "CFO>0": cfo_pos,
        "ΔROA>0": bool(pd.notna(cur["roa_ttm"]) and pd.notna(py["roa_ttm"]) and cur["roa_ttm"] > py["roa_ttm"]),
        "CFO>NI": cfo_gt_ni,
        "Δ负债<0": bool(pd.notna(cur["DEBT_ASSET_RATIO"]) and pd.notna(py["DEBT_ASSET_RATIO"]) and cur["DEBT_ASSET_RATIO"] < py["DEBT_ASSET_RATIO"]),
        "Δ毛利>0": bool(pd.notna(cur["gm_ttm"]) and pd.notna(py["gm_ttm"]) and cur["gm_ttm"] > py["gm_ttm"]),
        "Δ周转>0": bool(pd.notna(cur["at_ttm"]) and pd.notna(py["at_ttm"]) and cur["at_ttm"] > py["at_ttm"]),
    }
    fscore = sum(1 for k in CRIT if checks[k])
    detail = "".join("✓" if checks[k] else "·" for k in CRIT)
    return {
        "fscore": fscore,
        "detail": detail,
        "period": cur_dt.strftime("%Y-%m-%d"),
        "roa": float(cur["roa_ttm"]) if pd.notna(cur["roa_ttm"]) else np.nan,
        "debt": float(cur["DEBT_ASSET_RATIO"]) if pd.notna(cur["DEBT_ASSET_RATIO"]) else np.nan,
        "cf_meta": cf_meta,
    }


def run_screen(p: Optional[Params] = None, today=None) -> Result:
    p = p or Params()
    today = pd.Timestamp(today or dt.date.today())
    log = []

    st1 = _stage1_rank(p, log)
    u, cur_date = st1["u"], st1["cur_date"]
    pe_wide, pb_wide = st1["pe_wide"], st1["pb_wide"]
    pe_pct, pb_pct, vd_rank = st1["pe_pct"], st1["pb_pct"], st1["vd_rank"]
    cheap, info_map, usdcnh = st1["cheap"], st1["info_map"], st1["usdcnh"]

    rows = []
    cf_hit = 0
    for c in cheap:
        row_u = u.loc[c]
        st_info = info_map[c]
        mv_usd = float(row_u["market_cap"])
        f = _compute_fscore_one(c, st_info, today, mv_usd, usdcnh)
        if f["fscore"] < p.fscore_min:
            continue
        cfm = f["cf_meta"]
        if pd.notna(cfm.get("cfo_cur")):
            cf_hit += 1
        pe_now = float(pe_wide[c].iloc[-1]) if c in pe_wide.columns else np.nan
        pb_now = float(pb_wide[c].iloc[-1]) if c in pb_wide.columns else np.nan
        rows.append({
            "代码": c,
            "名称": str(row_u["name"]),
            "行业": str(row_u["sector"]),
            "国家": str(row_u["country"]),
            "类型": str(row_u.get("classify", "EQ")),
            "最新价($)": round(st_info["cur_px"], 2),
            "历史最低价(前复权$)": round(st_info["hist_low"], 2) if not np.isnan(st_info["hist_low"]) else np.nan,
            "PE(TTM)": round(pe_now, 1) if not np.isnan(pe_now) else np.nan,
            "PB": round(pb_now, 2) if not np.isnan(pb_now) else np.nan,
            "PE分位": round(float(pe_pct.get(c, np.nan)), 2),
            "PB分位": round(float(pb_pct.get(c, np.nan)), 2),
            "value便宜排名": round(float(vd_rank.get(c, np.nan)), 2),
            "F分": int(f["fscore"]),
            "7项明细": f["detail"],
            "财报期": f["period"],
            "货币": st_info["curr"],
            "成交额M$20d": round(st_info["amt20_musd"], 1) if not np.isnan(st_info["amt20_musd"]) else np.nan,
            "现金流正年数": cfm.get("cf_pos_yrs", np.nan),
            "分红年数": cfm.get("div_yrs", np.nan),
            "净回购年数": cfm.get("buyback_yrs", np.nan),
            "股东回报率%": cfm.get("sh_yield_pct", np.nan),
            "负债率%": round(f["debt"], 1) if not np.isnan(f["debt"]) else np.nan,
            "ROA%": round(f["roa"], 1) if not np.isnan(f["roa"]) else np.nan,
            "总市值亿$": round(mv_usd / 1e8, 1),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("value便宜排名", ascending=False).reset_index(drop=True)
    log.append(f"② F-score ≥ {p.fscore_min} 候选：{len(out)} 只（其中已匹配现金流量表 {cf_hit} 只）")
    if len(cheap) and cf_hit == 0:
        log.append("⚠ 最便宜候选股均未下载现金流量表，CFO>0 / CFO>NI / 现金流年数无法完整打分，请点击「更新 Stage 2 现金流」")

    counts = {"数据日": cur_date, "域": st1["n_domain"], "最便宜": len(cheap), "候选": len(out)}
    return Result(candidates=out, asof=cur_date, log=log, counts=counts)


def save_csv(res: Result) -> Optional[Path]:
    if res.candidates is None or res.candidates.empty:
        return None
    d = ud.DIRS["output"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"us_value_quality_screen_{pd.Timestamp(res.asof):%Y%m%d}.csv"
    res.candidates.to_csv(path, index=False, encoding="utf-8-sig")
    res.csv_path = path
    return path


# ───────────────────────── 子进程隔离运行（防 pyarrow 崩溃）─────────────────────────
class ScreenFailed(RuntimeError):
    def __init__(self, returncode, stderr="", trace=""):
        super().__init__(f"退出码 {returncode}")
        self.returncode, self.stderr, self.trace = returncode, stderr or "", trace or ""


def _env_attempts() -> list:
    import importlib.util
    seq = [("默认设置", {}), ("关闭 Arrow SIMD 加速", {"ARROW_USER_SIMD_LEVEL": "NONE"})]
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎", {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    return seq


_GOOD_ENV: Optional[dict] = None


def run_in_subprocess(p: Optional[Params] = None, timeout: int = 900) -> Result:
    global _GOOD_ENV
    p = p or Params()
    here = Path(__file__).resolve()
    attempts = [("上次成功的设置", _GOOD_ENV)] if _GOOD_ENV is not None else _env_attempts()
    tried, err = [], None
    for label, extra in attempts:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "us_res.pkl")
            cmd = [
                sys.executable, str(here),
                "--out", out, "--top-n", str(int(p.top_n)),
                "--excl", str(p.excl_small_pct), "--cheap", str(p.cheap_pct),
                "--fmin", str(int(p.fscore_min)), "--amt-min", str(p.amt_min_musd),
            ]
            if p.include_adr:
                cmd.append("--include-adr")
            if p.allow_neg_pb:
                cmd.append("--allow-neg-pb")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            env.update(extra)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=timeout, cwd=str(here.parent), env=env)
            except subprocess.TimeoutExpired as e:
                raise ScreenFailed("timeout", str(e.stderr or "")[-3000:]) from e
            if proc.returncode == 2:
                raise DataMissing((proc.stderr or "").replace("DATA_MISSING:", "").strip() or "数据不足")
            if proc.returncode == 0 and os.path.exists(out):
                with open(out, "rb") as f:
                    d = pickle.load(f)
                _GOOD_ENV = dict(extra)
                res = Result(candidates=d["candidates"], asof=d["asof"], log=d["log"], counts=d["counts"],
                             csv_path=Path(d["csv_path"]) if d.get("csv_path") else None)
                if tried:
                    res.log.insert(0, f"⚠ 前 {len(tried)} 次因底层库崩溃失败，已改用：{label}")
                return res
            err = ScreenFailed(proc.returncode, (proc.stderr or "")[-3000:])
            tried.append(label)
            if proc.returncode not in ud.NATIVE_CRASH:
                break
    _GOOD_ENV = None
    raise err


def _cli(argv=None) -> int:
    import argparse
    import traceback

    ap = argparse.ArgumentParser(description="美股价值+质量选股子进程入口")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=Params.top_n)
    ap.add_argument("--excl", type=float, default=Params.excl_small_pct)
    ap.add_argument("--cheap", type=float, default=Params.cheap_pct)
    ap.add_argument("--fmin", type=int, default=Params.fscore_min)
    ap.add_argument("--amt-min", type=float, default=Params.amt_min_musd)
    ap.add_argument("--include-adr", action="store_true")
    ap.add_argument("--allow-neg-pb", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = run_screen(Params(a.top_n, a.excl, a.cheap, a.fmin, a.amt_min, a.include_adr, a.allow_neg_pb))
        save_csv(res)
    except DataMissing as e:
        print(f"DATA_MISSING: {e}", file=sys.stderr)
        return 2
    except Exception:  # noqa
        traceback.print_exc()
        return 1
    payload = {"candidates": res.candidates, "asof": res.asof, "log": res.log,
               "counts": res.counts, "csv_path": str(res.csv_path) if res.csv_path else None}
    with open(a.out, "wb") as f:
        pickle.dump(payload, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
'''

US_VIEW_PY = r'''"""美股价值 + 质量选股页：宏观背景(VIX/JOLTS) → 分级增量下载 → 选股结果与决策卡。"""
import pandas as pd
import streamlit as st

import tushare_src as ts_src
import us_value_screen_core as uvc
import us_value_screen_data as ud

st.title("🇺🇸 美股价值 + 质量选股")
st.caption("便宜（总市值÷净利润TTM / 总市值÷净资产的 5 年自身分位，GICS 行业与市值中性化）∩ 财务健康（TTM F-score ≥ 5）。"
           "同时保留总股东回报（分红+净回购）与前复权历史最低价对照。")

# ───────────────────────── 0. 宏观背景：VIX & JOLTS ─────────────────────────
m1, m2 = st.columns(2)
if ud.VIX_FILE.exists():
    try:
        vix = ud._read_pq(ud.VIX_FILE).sort_values("trade_date")
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
        jo = ud._read_pq(ud.JOLTS_FILE).sort_values("ts_date")
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
st.caption("分两阶段省时省限流：Stage 1 仅对 Top N 市值股票拉取日线与财务/资产负债表；Stage 2 仅对最便宜的 ~10% 候选股拉取现金流量表。所有分片断点续传。")

with st.container(border=True):
    s1, s2, s3 = st.columns([1.2, 1.2, 1.2])
    top_n_opt = s1.selectbox("Stage 1 覆盖规模 (按市值 Top N)", [500, 800, 1000, 1500, 2200], index=2, key="us_topn")
    include_adr = s2.checkbox("包含中概股与外资 ADR", value=True, key="us_adr")
    allow_neg_pb = s3.checkbox("负 PB 回购股仅按 PE 排名", value=True, key="us_negpb",
                               help="如 AAPL / MCD / HD 因大额回购导致账面净资产为负，勾选后保留并仅按 PE 5年分位计算便宜度")

c_excl = st.session_state.get("us_excl", 0.30)
c_cheap = st.session_state.get("us_cheap", 0.10)
c_amt = st.session_state.get("us_amt", 5.0)


def _run_us_update(label, step_key):
    bar = st.progress(0.0, text=f"{label}：准备中…")

    def cb(frac, text):
        bar.progress(min(max(float(frac), 0.0), 1.0), text=text)

    logs = ud.run_update_in_subprocess(
        step=step_key,
        top_n=int(top_n_opt),
        include_adr=bool(include_adr),
        excl_small_pct=float(c_excl),
        cheap_pct=float(c_cheap),
        amt_min_musd=float(c_amt),
        allow_neg_pb=bool(allow_neg_pb),
        progress=cb,
    )
    bar.empty()
    st.session_state["us_update_log"] = logs
    st.rerun()


b0, b1, b2, b3, b4 = st.columns([1.4, 1, 1, 1.2, 1.2])
if b0.button("🔄 一键增量更新全部", type="primary", width="stretch", key="us_upd_all"):
    _run_us_update("美股全部数据", "all")
if b1.button("更新宏观(VIX/JOLTS)", width="stretch", key="us_upd_macro"):
    _run_us_update("美股宏观", "macro")
if b2.button("更新股票池(秒级)", width="stretch", key="us_upd_univ"):
    _run_us_update("美股股票池", "universe")
if b3.button("更新 Stage 1 (行情+财务)", width="stretch", key="us_upd_s1"):
    _run_us_update("Stage 1", "stage1")
if b4.button("更新 Stage 2 (便宜现金流)", width="stretch", key="us_upd_s2"):
    _run_us_update("Stage 2", "stage2")

if st.session_state.get("us_update_log"):
    with st.expander("上次更新日志", expanded=True):
        st.code("\n".join(st.session_state["us_update_log"]), language=None)

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

if st.button("▶ 运行美股选股", type="primary", disabled=bool(missing), key="us_run"):
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


def patch_app(src: str):
    if "views/us_value_screen.py" in src:
        return None
    anchor = '    st.Page("views/value_screen.py", title="价值质量选股", icon="💎"),'
    cnt = src.count(anchor)
    if cnt != 1:
        raise RuntimeError(f"app.py：期望找到 1 处 navigation 锚点，实际 {cnt} 处。")
    return src.replace(
        anchor,
        anchor + '\n    st.Page("views/us_value_screen.py", title="美股价值选股", icon="🇺🇸"),',
        1,
    )


def main():
    app_path = ROOT / "app.py"
    if not app_path.exists():
        sys.exit(f"[中止] 找不到 {app_path}，请将脚本放在项目根目录后运行。")

    files_to_create = [
        (ROOT / "us_value_screen_data.py", US_DATA_PY),
        (ROOT / "us_value_screen_core.py", US_CORE_PY),
        (ROOT / "views" / "us_value_screen.py", US_VIEW_PY),
    ]

    app_old = app_path.read_text(encoding="utf-8-sig")
    try:
        app_new = patch_app(app_old)
    except Exception as e:
        sys.exit(f"[中止] app.py 失败：{e}\n所有文件均未修改。")

    # 语法预检
    for p, content in files_to_create:
        compile(content, str(p), "exec")
    if app_new is not None:
        compile(app_new, str(app_path), "exec")

    # 写入文件
    for p, content in files_to_create:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"[写入] {p.relative_to(ROOT)}")

    if app_new is not None:
        bak = app_path.with_suffix(".py.bak_before_us_screen")
        if not bak.exists():
            shutil.copy2(app_path, bak)
        app_path.write_text(app_new, encoding="utf-8")
        print(f"[更新] app.py（备份：{bak.name}）")
    else:
        print("[跳过] app.py 已包含 美股价值选股 导航")

    print("\n全部完成！请重启 Streamlit：streamlit run app.py")
    print("打开新页面「🇺🇸 美股价值选股」后：")
    print("  1. 默认选择 Top 1000 市值美股（可在页面改为 500 / 800 / 1500 / 2200），点击「🔄 一键增量更新全部」；")
    print("  2. 数据下载在独立子进程中运行、支持断点续传（中断后重跑会自动跳过已完成股票）；")
    print("  3. 更新完成后点「▶ 运行美股选股」即可查看候选列表与 CSV 导出。")


if __name__ == "__main__":
    main()