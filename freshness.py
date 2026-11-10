"""数据新鲜度：每类数据的最新日期、应有的最新交易日、落后几个交易日。纯函数，不依赖网页。"""
import numpy as np
import pandas as pd

import config
import fetchers
import loaders


def parse_cutoff(s: str):
    h, m = str(s).split(":")
    return int(h), int(m)


def expected_latest(now=None, cutoff=None):
    """现在应当已经有数据的最新交易日：交易日当天过了截止时间(默认17:30)算当天，否则算上一个交易日。"""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    h, m = parse_cutoff(cutoff or config.READY_CUTOFF)
    days = fetchers.cached_trading_days(now.normalize() - pd.Timedelta(days=30), now.normalize())
    if days and days[-1] == now.normalize() and (now.hour, now.minute) < (h, m):
        days = days[:-1]
    return days[-1] if days else None


def is_trading_day(now=None) -> bool:
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    days = fetchers.cached_trading_days(now.normalize() - pd.Timedelta(days=3), now.normalize())
    return bool(days) and days[-1] == now.normalize()


def lag_days(last, expected) -> int:
    """last 落后 expected 几个交易日（last>=expected 为 0）。"""
    if last is None or expected is None or last >= expected:
        return 0
    cal = fetchers.cached_trading_days(last, expected)
    return len([d for d in cal if d > last])


ALLOWED_LAG = {"融资融券": 1, "ETF份额": 1}          # 这两类披露本身常晚一天


def _row(name, last, expected, detail="", allowed=0, core=True):
    if last is None or last != last:
        return {"数据": name, "最新日期": None, "应有日期": expected, "落后(交易日)": None, "状态": "❌ 无数据",
                "说明": detail, "核心": core}
    lag = lag_days(last, expected)
    if lag == 0:
        st_ = "✅ 最新"
    elif lag <= allowed:
        st_ = f"✅ 最新（披露延迟 {lag} 日，正常）"
    else:
        st_ = f"⚠️ 落后 {lag} 个交易日"
    return {"数据": name, "最新日期": pd.Timestamp(last), "应有日期": expected, "落后(交易日)": lag, "状态": st_,
            "说明": detail, "核心": core}


def _group_last(codes, label):
    """一组日线代码：取最旧的最新日期（最慢的那只决定整体新鲜度）。"""
    ds = {c: loaders.last_date(c) for c in codes}
    have = {c: d for c, d in ds.items() if d is not None}
    if not have:
        return None, f"0/{len(codes)} 个有数据"
    oldest = min(have, key=have.get)
    detail = f"{len(have)}/{len(codes)} 个有数据；最旧：{oldest}（{have[oldest].date()}）"
    return min(have.values()), detail


def collect(now=None, expected=None) -> pd.DataFrame:
    expected = expected if expected is not None else expected_latest(now)
    rows = []
    idx_codes = [v["code"] for v in config.INDEXES.values()]
    last, det = _group_last(idx_codes, "指数")
    rows.append(_row("指数日线(沪深300/上证50/中证500/中证1000)", last, expected, det))
    etfs = sorted({c for v in config.ETF_GROUPS.values() for c in v})
    last, det = _group_last(etfs, "ETF")
    rows.append(_row("ETF日线", last, expected, det, core=False))
    sec = sorted({c for s in config.SECTORS.values() for c, _ in s["stocks"]})
    last, det = _group_last(sec, "板块个股")
    rows.append(_row("板块个股日线", last, expected, det, core=False))
    for name, key in (("融资融券", "margin"), ("期指持仓排名", "cffex_rank"), ("期指日行情", "cffex_daily"),
                      ("ETF份额", "etf_shares")):
        df = fetchers.read_ds(key)
        last = df["date"].max() if len(df) else None
        rows.append(_row(name, last, expected, f"{df['date'].nunique()} 个交易日" if len(df) else "", ALLOWED_LAG.get(name, 0)))
    pr = fetchers.read_ds("proxy_daily")
    for sym, g in pr.groupby("symbol"):
        last = g["date"].max()
        gap = (pd.Timestamp(expected) - last).days if expected is not None else 0
        rows.append({"数据": f"外资代理 {sym}", "最新日期": last, "应有日期": expected, "落后(交易日)": None,
                     "状态": "ℹ️ 外盘日历不同" if gap <= 4 else f"⚠️ 落后 {gap} 天", "说明": "按各市场自身交易日", "核心": False})
    return pd.DataFrame(rows)


def core_stale(df: pd.DataFrame) -> bool:
    """核心数据（指数日线、融资、期指、ETF份额）里是否有落后/缺失。"""
    core = df[df["核心"]]
    return bool(core["状态"].str.startswith(("⚠️", "❌")).any())
