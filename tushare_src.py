"""Tushare 数据源（需要 token）。

token 读取顺序：环境变量 TUSHARE_TOKEN → 项目目录下 tushare_token.txt。**不要把 token 提交到 GitHub**。
核心提速思路：stock/index/fund 的 daily 接口支持"按交易日取全市场"——补 N 个交易日只要 N 次调用，
而不是"每只股票一次"（5500 次）。
"""
import datetime as dt
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config

_pro = None
_last_call = [0.0]
STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "code", "name"]


# ───────────── token / 连接 / 限速 ─────────────
def get_token() -> str:
    t = os.environ.get("TUSHARE_TOKEN", "").strip()
    if t:
        return t
    p = Path(config.TOKEN_FILE)
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def save_token(token: str):
    Path(config.TOKEN_FILE).write_text(token.strip(), encoding="utf-8")
    reset()


def reset():
    global _pro
    _pro = None


def available() -> bool:
    return bool(get_token())


def pro():
    global _pro
    if _pro is None:
        try:
            import tushare as ts
        except ImportError as e:
            raise ImportError("未安装 tushare：请运行 pip install tushare") from e
        ts.set_token(get_token())
        _pro = ts.pro_api()
    return _pro


def call(api: str, **kw) -> pd.DataFrame:
    """带限速和"频控休眠重试"的统一调用。"""
    interval = 60.0 / max(int(config.TUSHARE_CALLS_PER_MIN), 1)
    for attempt in range(4):
        wait = interval - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            df = getattr(pro(), api)(**kw)
            return df if df is not None else pd.DataFrame()
        except Exception as e:  # noqa
            msg = str(e)
            if attempt < 3 and any(k in msg for k in ("每分钟", "50次", "频率", "too many", "Too Many")):
                time.sleep(65)
                continue
            raise


# ───────────── 工具 ─────────────
def ts_code(code: str, kind: str) -> str:
    if kind == "index":
        return code + (".SZ" if code.startswith("399") else ".SH")
    if kind == "etf":
        return code + (".SH" if code.startswith(("5", "6")) else ".SZ")
    if code.startswith(("6", "9")):
        return code + ".SH"
    return code + (".BJ" if code.startswith(("4", "8")) else ".SZ")


def _ymd(x) -> str:
    return pd.Timestamp(x).strftime("%Y%m%d")


def _windows(start, end, years=6):
    """把长区间切成若干段，避免单次返回行数超过接口上限。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s <= e:
        t = min(s + pd.DateOffset(years=years) - pd.Timedelta(days=1), e)
        out.append((_ymd(s), _ymd(t)))
        s = t + pd.Timedelta(days=1)
    return out


def _windowed(api, start, end, **kw) -> pd.DataFrame:
    parts = [call(api, start_date=a, end_date=b, **kw) for a, b in _windows(start, end)]
    parts = [p for p in parts if p is not None and len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def trade_dates(start, end) -> list:
    df = call("trade_cal", exchange="SSE", start_date=_ymd(start), end_date=_ymd(end), is_open="1")
    return sorted(pd.to_datetime(df["cal_date"]).tolist())


def latest_ready_date() -> str:
    """Tushare 库里已入库的最新交易日（避免盘中/未入库时的空数据）。"""
    df = call("daily", ts_code="000001.SZ", limit=1)
    if df.empty:
        raise ValueError("探测最新交易日失败：daily 返回为空（检查 token / 权限 / 网络）")
    return str(df.iloc[0]["trade_date"])


def to_px(df: pd.DataFrame) -> pd.DataFrame:
    """Tushare 日线 → date/open/high/low/close/volume/amount；成交额 千元→元，成交量保持"手"(与旧数据一致)。"""
    out = pd.DataFrame({"date": pd.to_datetime(df["trade_date"]), "open": df["open"], "high": df["high"],
                        "low": df["low"], "close": df["close"], "volume": df["vol"].astype(float),
                        "amount": (df["amount"].astype(float) * 1000.0).round(2)})
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


# ───────────── 日线：指数 / ETF / 个股（单代码，整段历史）─────────────
def fetch_price(code: str, kind: str, start: str) -> pd.DataFrame:
    api = {"index": "index_daily", "etf": "fund_daily"}.get(kind, "daily")
    df = _windowed(api, start, dt.date.today(), ts_code=ts_code(code, kind))
    if df.empty:
        raise ValueError(f"Tushare {api} 返回为空（代码不存在或没有该接口权限）")
    return to_px(df)


# ───────────── 融资融券 / 期指 / ETF份额 / 成分股 ─────────────
def margin_day(d) -> pd.DataFrame:
    df = call("margin_detail", trade_date=_ymd(d))
    if df.empty:
        raise ValueError("Tushare margin_detail 当日无数据（非交易日/未入库/无权限）")
    return pd.DataFrame({"date": pd.Timestamp(d), "code": df["ts_code"].astype(str).str[:6],
                         "rz_balance": pd.to_numeric(df["rzye"], errors="coerce"),
                         "rz_buy": pd.to_numeric(df["rzmre"], errors="coerce"),
                         "rz_repay": pd.to_numeric(df["rzche"], errors="coerce")})


def fut_daily_day(d) -> pd.DataFrame:
    df = call("fut_daily", trade_date=_ymd(d), exchange="CFFEX")
    if df.empty:
        raise ValueError("Tushare fut_daily 当日无数据")
    df = df[df["ts_code"].astype(str).str.match(r"^(IF|IH|IC|IM)\d{4}\.")].copy()
    out = pd.DataFrame({"date": pd.Timestamp(d), "contract": df["ts_code"].astype(str).str[:6],
                        "close": pd.to_numeric(df["close"], errors="coerce"),
                        "settle": pd.to_numeric(df["settle"], errors="coerce"),
                        "volume": pd.to_numeric(df["vol"], errors="coerce"),
                        "oi": pd.to_numeric(df["oi"], errors="coerce")})
    out["variety"] = out["contract"].str[:2]
    return out[["date", "contract", "variety", "close", "settle", "volume", "oi"]]


def fut_rank_day(d) -> pd.DataFrame:
    """fut_holding 是"会员一行"；转成程序用的"名次一行"（成交量/持买/持卖三组各取前20）。"""
    df = call("fut_holding", trade_date=_ymd(d), exchange="CFFEX")
    if df.empty:
        raise ValueError("Tushare fut_holding 当日无数据")
    df = df[df["symbol"].astype(str).str.match(r"^(IF|IH|IC|IM)\d{4}$")]
    df = df[~df["broker"].astype(str).str.contains("合计")]
    rows = []
    for con, g in df.groupby("symbol"):
        def top(col, chg):
            t = g.dropna(subset=[col]).sort_values(col, ascending=False).head(20)
            return t["broker"].tolist(), t[col].tolist(), t[chg].tolist()
        lm, lo, lc = top("long_hld", "long_chg")
        sm, so, sc = top("short_hld", "short_chg")
        vm, vo, vc = top("vol", "vol_chg")
        for r in range(max(len(lm), len(sm), len(vm))):
            pick = lambda a, i: a[i] if i < len(a) else np.nan  # noqa
            rows.append({"date": pd.Timestamp(d), "contract": con, "variety": con[:2], "rank": r + 1,
                         "long_member": pick(lm, r), "long_oi": pick(lo, r), "long_chg": pick(lc, r),
                         "short_member": pick(sm, r), "short_oi": pick(so, r), "short_chg": pick(sc, r),
                         "vol_member": pick(vm, r), "vol": pick(vo, r), "vol_chg": pick(vc, r)})
    if not rows:
        raise ValueError("Tushare fut_holding 没有可用的 IF/IH/IC/IM 持仓行")
    return pd.DataFrame(rows)


def etf_shares_history(codes, start) -> pd.DataFrame:
    """按代码取整段历史份额（fund_share，单位万份→份）。ETF 只有十几只，比按日取全市场更省调用。"""
    parts = []
    for c in codes:
        df = _windowed("fund_share", start, dt.date.today(), ts_code=ts_code(c, "etf"))
        if len(df):
            parts.append(pd.DataFrame({"date": pd.to_datetime(df["trade_date"]), "code": c, "name": "",
                                       "shares": pd.to_numeric(df["fd_share"], errors="coerce") * 1e4}))
    if not parts:
        raise ValueError("Tushare fund_share 全部为空（无权限或代码不对）")
    return pd.concat(parts, ignore_index=True).dropna(subset=["shares"])


def index_cons(index_code: str) -> pd.DataFrame:
    end = dt.date.today()
    df = call("index_weight", index_code=ts_code(index_code, "index"), start_date=_ymd(end - dt.timedelta(days=60)),
              end_date=_ymd(end))
    if df.empty:
        raise ValueError("Tushare index_weight 为空")
    df = df[df["trade_date"] == df["trade_date"].max()]
    return pd.DataFrame({"index_code": index_code, "code": df["con_code"].astype(str).str[:6], "name": ""})


# ───────────── 外资代理 ─────────────
def fx_hist(ts: str = "USDCNH.FXCM") -> pd.DataFrame:
    df = _windowed("fx_daily", config.PX_START, dt.date.today(), ts_code=ts)
    if df.empty:
        raise ValueError("Tushare fx_daily 为空（无权限或代码不对）")
    return pd.DataFrame({"date": pd.to_datetime(df["trade_date"]), "open": df["bid_open"], "high": df["bid_high"],
                         "low": df["bid_low"], "close": df["bid_close"]})


def index_global_hist(ts: str) -> pd.DataFrame:
    df = _windowed("index_global", config.PX_START, dt.date.today(), ts_code=ts)
    if df.empty:
        raise ValueError(f"Tushare index_global({ts}) 为空")
    return pd.DataFrame({"date": pd.to_datetime(df["trade_date"]), "open": df["open"], "high": df["high"],
                         "low": df["low"], "close": df["close"], "volume": df.get("vol")})


# ───────────── 本地个股日线批量更新（追加到你的日线目录）─────────────
def _parse_ymd(raw):
    if not raw:
        return None
    clean = str(raw).strip().strip("\"'").replace("/", "").replace("-", "")
    return clean if len(clean) == 8 and clean.isdigit() else None


def _last_date(path: Path):
    """读文件最后一行的日期（兼容 2026/04/01、2026-04-01、20260401）。"""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(-400, os.SEEK_END)
            except OSError:
                f.seek(0)
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return _parse_ymd(lines[-1].decode("utf-8", errors="ignore").split(",")[0]) if lines else None
    except Exception:  # noqa
        return None


def _last_close(path: Path):
    """读文件最后一行的收盘价；读不出返回 None。"""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            header = [c.strip() for c in f.readline().strip().split(",")]
        with open(path, "rb") as f:
            try:
                f.seek(-400, os.SEEK_END)
            except OSError:
                f.seek(0)
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return float(lines[-1].decode("utf-8", errors="ignore").split(",")[header.index("close")])
    except Exception:  # noqa
        return None


def _continuous(path: Path, g: pd.DataFrame) -> bool:
    """新数据首日的昨收(已按除权调整)与文件最后收盘价相差 4 倍以上 → 不是同一标的(如指数文件与同号个股)，不追加。
    送转股最多让比值到 2~3 倍，所以 4 倍是安全阈值。"""
    lc = _last_close(path)
    pc = g["pre_close"].iloc[0] if "pre_close" in g.columns and len(g) else np.nan
    if lc is None or lc != lc or pc != pc or pc <= 0 or lc <= 0:
        return True
    return 0.25 <= lc / pc <= 4


def _local_path(code: str) -> Path:
    import loaders
    for p in loaders._file_index().get(code, []):
        if config.CACHE_DIR not in p.parents:
            return p
    return Path(config.DAILY_DIR) / f"{code}.csv"


def _append_rows(path: Path, g: pd.DataFrame) -> bool:
    """追加：保持原表头列顺序，补换行；文件不存在则新建标准列。返回是否写入。"""
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        g[STD_COLS].to_csv(path, index=False, encoding="utf-8-sig")
        return True
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
    with open(path, "r", encoding="utf-8-sig") as f:
        header = [c.strip() for c in f.readline().strip().split(",")]
    cols = [c for c in header if c in g.columns]
    if "date" not in cols or "close" not in cols:
        return False                                              # 表头不是标准格式，不动它
    g[cols].to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    return True


def _to_std(df: pd.DataFrame, names: dict) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d"), "open": df["open"],
                        "high": df["high"], "low": df["low"], "close": df["close"], "volume": df["vol"].astype(float),
                        "amount": (df["amount"].astype(float) * 1000.0).round(2)})
    out["code"] = df["ts_code"].astype(str).str[:6]
    out["name"] = out["code"].map(names).fillna("")
    out["pre_close"] = pd.to_numeric(df["pre_close"], errors="coerce") if "pre_close" in df.columns else np.nan
    return out.sort_values(["code", "date"]).drop_duplicates(["code", "date"])


def update_local_stock_daily(progress=None, history_start=None, chunk_days=15) -> list:
    """把 DAILY_DIR 里每只股票的日线补到最新（追加，格式与旧文件一致：date,open,high,low,close,volume,amount,code,name）。
    缺 N 个交易日 → 按日取全市场 N 次调用；若更划算再退回逐只股票取。"""
    import loaders
    history_start = history_start or config.PX_START
    latest = latest_ready_date()
    basic = call("stock_basic", exchange="", list_status="L", fields="ts_code,symbol,name")
    names = dict(zip(basic["symbol"], basic["name"]))
    todo, fresh = {}, 0
    index_codes = {v["code"] for v in config.INDEXES.values()}
    skipped_idx = []
    for code in basic["symbol"]:
        if code in index_codes:                     # 000016/000905/000852 既是指数又是个股：绝不往可能是指数的文件里追加
            skipped_idx.append(code)
            continue
        p = _local_path(code)
        ld = _last_date(p) if p.exists() else None
        if ld is None:
            todo[code] = history_start
        elif ld < latest:
            todo[code] = (dt.datetime.strptime(ld, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
        else:
            fresh += 1
    log = [f"【⑥ 本地个股日线】目录 {config.DAILY_DIR}；最新入库日 {latest}；已是最新 {fresh} 只，需补 {len(todo)} 只"]
    if skipped_idx:
        log.append(f"  已跳过与指数同号的个股 {skipped_idx}（其本地文件可能是指数数据，不追加）")
    if not todo:
        return log
    days = trade_dates(min(todo.values()), latest)
    written, bad, discont = 0, 0, []
    if len(days) <= len(todo):
        log.append(f"  方式：按日批量（{len(days)} 次调用，替代逐只 {len(todo)} 次）")
        for i in range(0, len(days), chunk_days):
            frames = []
            for d in days[i:i + chunk_days]:
                df = call("daily", trade_date=_ymd(d))
                if len(df):
                    frames.append(df)
                if progress:
                    progress(min((days.index(d) + 1) / len(days), 1.0), f"本地日线 {d.date()}")
            if not frames:
                continue
            big = _to_std(pd.concat(frames, ignore_index=True), names)
            for code, g in big.groupby("code"):
                if code not in todo:
                    continue
                g = g[g["date"].str.replace("-", "") >= todo[code]]
                if g.empty:
                    continue
                path = _local_path(code)
                if path.exists() and not _continuous(path, g):
                    discont.append(code)
                    continue
                if _append_rows(path, g):
                    written += 1
                else:
                    bad += 1
    else:
        log.append(f"  方式：逐只（{len(todo)} 次调用，比按日批量 {len(days)} 次更少）")
        for i, (code, start) in enumerate(todo.items()):
            try:
                df = call("daily", ts_code=ts_code(code, "stock"), start_date=start, end_date=latest)
            except Exception as e:  # noqa
                log.append(f"  {code}: {e}")
                continue
            if len(df):
                g = _to_std(df, names)
                path = _local_path(code)
                if path.exists() and not _continuous(path, g):
                    discont.append(code)
                    continue
                if _append_rows(path, g):
                    written += 1
                else:
                    bad += 1
            if progress:
                progress((i + 1) / len(todo), f"本地日线 {code}")
    loaders.rescan()
    log.append(f"  完成：写入/追加 {written} 只" + (f"；{bad} 只文件表头不是标准格式，已跳过" if bad else ""))
    if discont:
        log.append(f"  ⚠️ {len(discont)} 只股票的本地文件最后收盘价与最新数据相差4倍以上，疑似文件内容不是该股票，已跳过不追加：{discont[:10]}")
    return log


# ───────────── 接口自检 ─────────────
def self_check() -> pd.DataFrame:
    """逐个测试程序用到的 Tushare 接口（取最近一个交易日/最近几天），返回 接口/状态/行数/说明。"""
    try:
        latest = latest_ready_date()
    except Exception as e:  # noqa
        return pd.DataFrame([{"接口": "daily(探测最新交易日)", "状态": "❌", "行数": 0, "说明": str(e)[:200]}])
    d = pd.Timestamp(latest)
    tests = [
        ("trade_cal 交易日历", lambda: trade_dates(d - pd.Timedelta(days=30), d)),
        ("daily 个股(按日取全市场)", lambda: call("daily", trade_date=latest)),
        ("index_daily 指数", lambda: fetch_price("000300", "index", _ymd(d - pd.Timedelta(days=30)))),
        ("fund_daily ETF日线", lambda: fetch_price("510300", "etf", _ymd(d - pd.Timedelta(days=30)))),
        ("margin_detail 融资融券明细", lambda: margin_day(d)),
        ("fut_daily 中金所日行情", lambda: fut_daily_day(d)),
        ("fut_holding 中金所持仓排名", lambda: fut_rank_day(d)),
        ("fund_share ETF份额", lambda: etf_shares_history(["510300"], d - pd.Timedelta(days=15))),
        ("index_weight 指数成分", lambda: index_cons("000300")),
        ("fx_daily USDCNH", lambda: fx_hist("USDCNH.FXCM")),
        ("index_global A50现货(XIN9)", lambda: index_global_hist("XIN9")),
    ]
    rows = []
    for name, fn in tests:
        try:
            r = fn()
            rows.append({"接口": name, "状态": "✅", "行数": len(r), "说明": ""})
        except Exception as e:  # noqa
            rows.append({"接口": name, "状态": "❌", "行数": 0, "说明": str(e)[:200]})
    return pd.DataFrame(rows)
