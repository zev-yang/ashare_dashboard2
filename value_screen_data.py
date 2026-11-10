# -*- coding: utf-8 -*-
"""价值+质量选股 · 数据层。

移植自 G_Claude 的 fetch_* 系列，接入本项目已有的 Tushare 配置（tushare_src：token / 连接）。
和原脚本相比的小改进：
  · token、连接复用 tushare_src，不再单独读环境变量；
  · 自带限速器：遇到频控自动降速并重试，不会一直撞墙；
  · 只下载选股真正用到的数据：daily_basic 只取最近 BASIC_YEARS 年（5 年分位够用），
    不再下载 daily / adj_factor 的 2010 年起全历史；
  · 财务指标优先用 fina_indicator_vip 按报告期整市场一次取（约 12 次调用，原来逐只 ~5500 次）；
    没有该权限会自动退回逐只模式；
  · "近 3 个财年"随日期自动滚动（原来写死 2023-2025）；
  · 全部断点续传、增量更新；每个 update_* 返回日志列表（与 tushare_src.update_local_stock_daily 同风格）。

数据落地：data/cache/value_screen/<数据集>/*.parquet （需要 pyarrow 或 fastparquet）
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

import config
import tushare_src as ts_src

ROOT = Path(config.CACHE_DIR) / "value_screen"
DIRS = {
    "daily_basic": ROOT / "daily_basic",      # 每个交易日一个文件：ts_code,trade_date,pe_ttm,pb,circ_mv
    "daily_recent": ROOT / "daily_recent",    # 最近若干交易日的成交额（算 20 日均成交额）
    "fundamentals": ROOT / "fundamentals",    # 每个报告期一个文件：fina_indicator
    "cashflow": ROOT / "cashflow",            # 每个年报期一个文件：n_cashflow_act
    "dividend": ROOT / "dividend",            # 每个公告日一个文件：dividend
    "industry": ROOT / "industry",            # stock_industry.parquet
}
IND_FILE = DIRS["industry"] / "stock_industry.parquet"

CALLS_PER_MIN = int(os.environ.get("VS_CALLS_PER_MIN", "100"))  # 起始速度；撞到频控会自动减半
PAGE_LIMIT = 6000
BASIC_YEARS = 6        # 5 年分位 = 1250 个交易日，再留点余量
RECENT_GUARD = 3       # 最近 N 个交易日即使返回空也不锁定（数据可能还没入库）
REFETCH_PERIODS = 3    # 财报：每次重拉最近 N 个报告期（晚披露/重述）
AMT_DAYS = 25          # 近 25 个交易日成交额（取最近 20 日均值）

# 质量过滤要用的 fina_indicator 字段（与 G_Claude 一致）
FUND_FIELDS = ("ts_code,ann_date,end_date,roe,roa,debt_to_assets,"
               "ocfps,netprofit_yoy,or_yoy,grossprofit_margin,netprofit_margin,"
               "q_profit_yoy,profit_to_op,assets_turn,"
               "ocf_to_profit,ocf_to_or,"
               "q_roe,q_gsprofit_margin,assets_yoy")
FUND_KEEP = FUND_FIELDS.split(",")
FUND_NUM = [c for c in FUND_KEEP if c not in ("ts_code", "ann_date", "end_date")]

Progress = Optional[Callable[[float, str], None]]


# ───────────────────────── 限速 / 重试 / 分页 ─────────────────────────
class PermissionDenied(RuntimeError):
    """账号没有该接口权限（积分不够）。"""


class _Throttle:
    def __init__(self):
        self.reset()

    def reset(self, per_min: Optional[int] = None):
        self.interval = 60.0 / max(int(per_min or CALLS_PER_MIN), 1)
        self._last = 0.0

    def wait(self):
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()

    def slow_down(self):
        self.interval = min(self.interval * 2, 4.0)


_THR = _Throttle()
_RATE_KEYS = ("每分钟", "频率", "too many", "Too Many")


def query(api: str, **kw) -> pd.DataFrame:
    """一次逻辑调用：分页 + 限速 + 退避重试。

    注意：Tushare 的"频控"提示里也带"权限"二字（"…权限的具体详情访问…"），
    所以必须先判断频控，再判断是否真的没有权限。"""
    pro = ts_src.pro()
    frames, offset = [], 0
    while True:
        df = None
        for attempt in range(5):
            _THR.wait()
            try:
                df = pro.query(api, offset=offset, limit=PAGE_LIMIT, **kw)
                break
            except Exception as e:  # noqa
                msg = str(e)
                if any(k in msg for k in _RATE_KEYS):
                    if attempt == 4:
                        raise
                    _THR.slow_down()
                    time.sleep(65)
                elif "每天" in msg or "每小时" in msg:
                    raise  # 配额用完，重试没有意义
                elif "没有" in msg and "权限" in msg:
                    raise PermissionDenied(f"{api}: {msg[:150]}") from e
                elif attempt == 4:
                    raise
                else:
                    time.sleep(2.0 * (attempt + 1))
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ───────────────────────── 小工具 ─────────────────────────
def fiscal_years(today=None, n: int = 3) -> tuple:
    """最近 n 个"年报已披露"的财年。年报 4/30 前后披露完毕：
    4/30 之后 → 去年；之前 → 前年。2026-11 → (2023, 2024, 2025)。"""
    t = pd.Timestamp(today or dt.date.today())
    y = t.year - 1 if (t.month, t.day) >= (4, 30) else t.year - 2
    return tuple(range(y - n + 1, y + 1))


def _quarter_ends(y0: int, today) -> list:
    t = pd.Timestamp(today).strftime("%Y%m%d")
    ps = [f"{y}{md}" for y in range(y0, pd.Timestamp(today).year + 1)
          for md in ("0331", "0630", "0930", "1231")]
    return [p for p in ps if p <= t]


def _stems(d: Path, exts=(".parquet", ".empty")) -> set:
    return {p.stem for p in d.iterdir() if p.suffix in exts} if d.exists() else set()


def _age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400.0


def _trade_days(start: str, end: str) -> list:
    cal = query("trade_cal", exchange="SSE", start_date=start, end_date=end, is_open="1")
    return sorted(str(d) for d in cal["cal_date"]) if len(cal) else []


def _is_sh_sz(s: pd.Series) -> pd.Series:
    """只留沪深（剔北交所），与财务 / 行业数据的口径一致。"""
    return s.astype(str).str[-3:].isin([".SZ", ".SH"])


def _pull_by_date(api, out_dir: Path, days: list, label: str, progress: Progress,
                  key: str = "trade_date", **kw):
    """每个日期一次调用、一个 parquet 分片；已拉过的跳过；空日期打 .empty 标记（最近几天除外，留待重试）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    recent = set(days[-RECENT_GUARD:])
    for d in recent:
        (out_dir / f"{d}.empty").unlink(missing_ok=True)
    done = _stems(out_dir)
    todo = [d for d in days if d not in done]
    wrote = failed = 0
    for i, d in enumerate(todo, 1):
        try:
            df = query(api, **{key: d}, **kw)
        except PermissionDenied:
            raise
        except Exception:  # noqa  单日失败不影响其它日，下次自动重试
            failed += 1
            continue
        if df.empty:
            if d not in recent:
                (out_dir / f"{d}.empty").touch()
        else:
            df.to_parquet(out_dir / f"{d}.parquet", index=False)
            wrote += 1
        if progress and (i % 10 == 0 or i == len(todo)):
            progress(i / len(todo), f"{label} {i}/{len(todo)}（{d}）")
    return len(todo), wrote, failed


# ───────────────────────── 各数据集的更新函数 ─────────────────────────
def update_industry(progress: Progress = None) -> list:
    """个股申万一级行业（stock_basic.industry），一次调用，秒级。含已退市/暂停，剔北交所。"""
    _THR.reset()
    frames = []
    for status in ("L", "D", "P"):
        try:
            df = query("stock_basic", exchange="", list_status=status,
                       fields="ts_code,name,industry,market,list_date")
            if len(df):
                frames.append(df)
        except PermissionDenied as e:
            return [f"【行业】{e}"]
        except Exception as e:  # noqa
            return [f"【行业】{status} 拉取失败：{e!r}"]
    if not frames:
        return ["【行业】stock_basic 无数据（检查 token / 权限）"]
    allf = pd.concat(frames, ignore_index=True)
    allf = allf[_is_sh_sz(allf["ts_code"])]
    allf["code"] = allf["ts_code"].astype(str).str[:6]
    allf = allf.drop_duplicates("code", keep="first")
    DIRS["industry"].mkdir(parents=True, exist_ok=True)
    allf[["code", "ts_code", "name", "industry", "market", "list_date"]].to_parquet(IND_FILE, index=False)
    return [f"【行业】{len(allf)} 只股票，{allf['industry'].notna().sum()} 只有行业标签，"
            f"{allf['industry'].nunique()} 个一级行业"]


def update_daily_basic(progress: Progress = None) -> list:
    """每日 PE(TTM) / PB / 流通市值。最近 BASIC_YEARS 年，增量。首次约 1500 次调用。"""
    _THR.reset()
    today = dt.date.today()
    start = (today - dt.timedelta(days=int(365.25 * BASIC_YEARS))).strftime("%Y%m%d")
    days = _trade_days(start, today.strftime("%Y%m%d"))
    if not days:
        return ["【daily_basic】交易日历为空（检查 token / 网络）"]
    try:
        n, w, f = _pull_by_date("daily_basic", DIRS["daily_basic"], days, "daily_basic", progress,
                                fields="ts_code,trade_date,pe_ttm,pb,circ_mv")
    except PermissionDenied as e:
        return [f"【daily_basic】{e}（需约 2000 积分）"]
    return [f"【daily_basic】{days[0]}→{days[-1]}：待拉 {n} 个交易日，写入 {w}，失败 {f}（失败的下次自动重试）"]


def update_recent_daily(progress: Progress = None) -> list:
    """最近 AMT_DAYS 个交易日的成交额，只用来算"20 日均成交额"。"""
    _THR.reset()
    today = dt.date.today()
    days = _trade_days((today - dt.timedelta(days=60)).strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    days = days[-AMT_DAYS:]
    if not days:
        return ["【近期成交额】交易日历为空"]
    try:
        n, w, f = _pull_by_date("daily", DIRS["daily_recent"], days, "近期成交额", progress,
                                fields="ts_code,trade_date,amount")
    except PermissionDenied as e:
        return [f"【近期成交额】{e}"]
    return [f"【近期成交额】{days[0]}→{days[-1]}：待拉 {n} 天，写入 {w}，失败 {f}"]


def _save_fund(df: pd.DataFrame) -> int:
    """按报告期分片落盘；与已有分片合并，同 (ts_code,end_date) 保留 ann_date 最早的一条
    （"市场第一次知道"的时点，最保守）。返回写入的分片数。"""
    if df is None or df.empty:
        return 0
    df = df.copy()
    df = df[_is_sh_sz(df["ts_code"])]
    df["ann_date"] = df["ann_date"].astype(str)
    df["end_date"] = df["end_date"].astype(str)
    df = df.dropna(subset=["ann_date", "end_date"])
    for c in FUND_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    DIRS["fundamentals"].mkdir(parents=True, exist_ok=True)
    n = 0
    for per, g in df.groupby("end_date"):
        p = DIRS["fundamentals"] / f"{per}.parquet"
        if p.exists():
            g = pd.concat([pd.read_parquet(p), g], ignore_index=True)
        g = (g.sort_values(["ts_code", "ann_date"])
              .drop_duplicates(["ts_code", "end_date"], keep="first"))
        g[[c for c in FUND_KEEP if c in g.columns]].to_parquet(p, index=False)
        n += 1
    return n


def _stock_codes() -> list:
    frames = []
    for status in ("L", "D", "P"):  # 上市/退市/暂停（含退市，防幸存者偏差）
        try:
            sb = query("stock_basic", exchange="", list_status=status, fields="ts_code")
            if len(sb):
                frames.append(sb)
        except Exception:  # noqa
            continue
    if not frames:
        return []
    codes = pd.concat(frames, ignore_index=True)["ts_code"].astype(str)
    return sorted(codes[codes.str[-3:].isin([".SZ", ".SH"])].unique())


def _fund_per_stock(start: str, progress: Progress) -> str:
    """没有 fina_indicator_vip 权限时的退路：逐只股票取（慢，~5500 次调用），每 800 只落盘一次。"""
    codes = _stock_codes()
    if not codes:
        return "逐只模式：取不到股票列表（检查 stock_basic 权限）"
    today = dt.date.today().strftime("%Y%m%d")
    buf, wrote = [], 0
    for i, code in enumerate(codes, 1):
        try:
            df = query("fina_indicator", ts_code=code, start_date=start, end_date=today, fields=FUND_FIELDS)
        except PermissionDenied as e:
            return f"逐只模式也没有权限：{e}"
        except Exception:  # noqa
            continue
        if len(df):
            buf.append(df)
        if i % 800 == 0 or i == len(codes):
            if buf:
                wrote += _save_fund(pd.concat(buf, ignore_index=True))
                buf = []
        if progress and (i % 50 == 0 or i == len(codes)):
            progress(i / len(codes), f"财务指标（逐只）{i}/{len(codes)}")
    return f"逐只模式完成：{len(codes)} 只，写入/更新 {wrote} 个分片"


def update_fundamentals(progress: Progress = None) -> list:
    """季度财务指标（fina_indicator）→ F-score / 质量信息列。"""
    _THR.reset()
    today = dt.date.today()
    periods = _quarter_ends(today.year - 3, today)
    out = DIRS["fundamentals"]
    out.mkdir(parents=True, exist_ok=True)
    recent = set(periods[-REFETCH_PERIODS:])
    todo = [p for p in periods if p in recent or not (out / f"{p}.parquet").exists()]
    log, wrote = [], 0
    for i, p in enumerate(todo, 1):
        try:
            df = query("fina_indicator_vip", period=p, fields=FUND_FIELDS)
        except PermissionDenied:
            log.append("【财务指标】没有 fina_indicator_vip 权限，退回逐只模式（慢，首次约 5500 次调用）")
            log.append("【财务指标】" + _fund_per_stock(periods[0], progress))
            return log
        except Exception as e:  # noqa
            log.append(f"【财务指标】{p} 失败：{e!r}（下次重试）")
            continue
        wrote += _save_fund(df)
        if progress:
            progress(i / len(todo), f"财务指标 {p}（{i}/{len(todo)}）")
    log.insert(0, f"【财务指标】{periods[0]}→{periods[-1]}：处理 {len(todo)} 个报告期，写入 {wrote} 个分片")
    return log


def update_cashflow(progress: Progress = None) -> list:
    """近 3 个财年的经营现金流（cashflow_vip）→ "现金流为正年数"。信息列，缺了不影响选股。"""
    _THR.reset()
    years = fiscal_years()
    out = DIRS["cashflow"]
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for k, y in enumerate(years, 1):
        per = f"{y}1231"
        f = out / f"{per}.parquet"
        newest = (y == years[-1])
        if f.exists() and not (newest and _age_days(f) > 3):  # 旧年报已稳定；最新一期偶尔重拉
            continue
        try:
            df = query("cashflow_vip", period=per, report_type="1",
                       fields="ts_code,ann_date,f_ann_date,end_date,report_type,n_cashflow_act")
        except PermissionDenied as e:
            return log + [f"【现金流】{e}（需约 5000 积分；'现金流正年数'一列会留空，不影响选股）"]
        except Exception as e:  # noqa
            log.append(f"【现金流】{per} 失败：{e!r}")
            continue
        df = df[_is_sh_sz(df["ts_code"])] if len(df) else df
        if len(df):
            df.to_parquet(f, index=False)
        if progress:
            progress(k / len(years), f"现金流 {per}")
    return log + [f"【现金流】财年 {years[0]}–{years[-1]} 已就绪"]


def update_dividend(progress: Progress = None) -> list:
    """分红送股（dividend），按公告日逐日拉取 → "分红年数"。信息列，缺了不影响选股。首次约 1200 次调用。"""
    _THR.reset()
    years = fiscal_years()
    today = dt.date.today()
    days = _trade_days(f"{years[0] - 1}0101", today.strftime("%Y%m%d"))
    if not days:
        return ["【分红】交易日历为空"]
    try:
        n, w, f = _pull_by_date("dividend", DIRS["dividend"], days, "分红", progress, key="ann_date")
    except PermissionDenied as e:
        return [f"【分红】{e}"]
    return [f"【分红】{days[0]}→{days[-1]}：待拉 {n} 个公告日，写入 {w}，失败 {f}"]


STEPS = [
    ("行业分类", update_industry),
    ("daily_basic（PE/PB/市值）", update_daily_basic),
    ("财务指标（F-score）", update_fundamentals),
    ("现金流（信息列）", update_cashflow),
    ("分红（信息列）", update_dividend),
    ("近期成交额（信息列）", update_recent_daily),
]


def update_all(progress: Progress = None) -> list:
    """全部增量更新。某一步失败不影响其它步（分片已在盘上，重跑会续上）。"""
    log = []
    for label, fn in STEPS:
        try:
            log += fn(progress)
        except Exception as e:  # noqa
            log.append(f"【{label}】中止：{e!r}（已下载的分片保留，重跑会续上）")
    return log


# ───────────────────────── 数据状态 ─────────────────────────
REQUIRED = ("daily_basic", "fundamentals", "industry")


def _files(key: str) -> list:
    d = DIRS[key]
    return sorted(d.glob("*.parquet")) if d.exists() else []


def missing_required() -> list:
    """选股必需、但本地还没有的数据集。"""
    return [k for k in REQUIRED if not _files(k)]


def _expected_period(today, lag_days: int = 40) -> str:
    qs = _quarter_ends(pd.Timestamp(today).year - 1, today)
    ok = [q for q in qs if (pd.Timestamp(today) - pd.Timestamp(q)).days >= lag_days]
    return ok[-1] if ok else ""


def data_status(today=None) -> pd.DataFrame:
    today = pd.Timestamp(today or dt.date.today())
    rows = []

    def date_row(label, key, role, ok_age=5):
        fs = _files(key)
        if not fs:
            rows.append({"数据集": label, "用途": role, "最新": "—", "文件数": 0, "状态": "❌ 未下载"})
            return
        last = fs[-1].stem
        age = (today - pd.Timestamp(last)).days
        rows.append({"数据集": label, "用途": role, "最新": last, "文件数": len(fs),
                     "状态": "✅" if age <= ok_age else f"⚠️ 落后 {age} 天"})

    date_row("daily_basic", "daily_basic", "PE/PB 分位、流通市值（必需）")
    fs = _files("fundamentals")
    if fs:
        last, exp = fs[-1].stem, _expected_period(today)
        rows.append({"数据集": "财务指标", "用途": "F-score（必需）", "最新": last, "文件数": len(fs),
                     "状态": "✅" if last >= exp else f"⚠️ 应有 {exp}"})
    else:
        rows.append({"数据集": "财务指标", "用途": "F-score（必需）", "最新": "—", "文件数": 0, "状态": "❌ 未下载"})
    ind = IND_FILE
    rows.append({"数据集": "行业分类", "用途": "行业中性化（必需）",
                 "最新": dt.date.fromtimestamp(ind.stat().st_mtime).isoformat() if ind.exists() else "—",
                 "文件数": 1 if ind.exists() else 0, "状态": "✅" if ind.exists() else "❌ 未下载"})
    have = {p.stem[:4] for p in _files("cashflow")}
    need = {str(y) for y in fiscal_years(today)}
    rows.append({"数据集": "现金流", "用途": "现金流正年数（信息列）",
                 "最新": max(have) if have else "—", "文件数": len(have),
                 "状态": "✅" if need <= have else "⚪ 可选，未齐"})
    date_row("分红", "dividend", "分红年数（信息列）")
    date_row("近期成交额", "daily_recent", "20 日均成交额（信息列）")
    for r in rows:
        if r["数据集"] in ("分红", "近期成交额") and r["状态"].startswith("❌"):
            r["状态"] = "⚪ 可选，未下载"
    return pd.DataFrame(rows)
