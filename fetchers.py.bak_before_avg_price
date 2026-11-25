"""数据下载 / 导入 / 缓存。

说明：本文件里所有联网接口都【没有在开发环境里实测】（交易所网站和 akshare 接口会改版）。
设计原则：自动下载失败 → 返回清楚的失败原因 → 你可以把官网下载的文件直接上传导入（import_* 系列）。
"""
import io
import random
import re
import time
from datetime import date

import numpy as np
import pandas as pd

import config
import tushare_src as tsrc

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
VARIETIES = ("IF", "IH", "IC", "IM")

SCHEMAS = {
    "cffex_rank": ["date", "contract", "variety", "rank", "long_member", "long_oi", "long_chg",
                   "short_member", "short_oi", "short_chg", "vol_member", "vol", "vol_chg"],
    "cffex_daily": ["date", "contract", "variety", "close", "settle", "volume", "oi"],
    "margin": ["date", "code", "rz_balance", "rz_buy", "rz_repay"],
    "etf_shares": ["date", "code", "name", "shares"],
    "index_cons": ["index_code", "code", "name"],
    "proxy_daily": ["date", "symbol", "open", "high", "low", "close", "volume"],
}
KEYS = {"cffex_rank": ["date", "contract", "rank"], "cffex_daily": ["date", "contract"],
        "margin": ["date", "code"], "etf_shares": ["date", "code"], "index_cons": ["index_code", "code"],
        "proxy_daily": ["date", "symbol"]}


# ───────────── 缓存 ─────────────
def _path(name):
    return config.CACHE_DIR / f"{name}.csv"


def read_ds(name) -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        empty = pd.DataFrame(columns=SCHEMAS[name])
        if "date" in empty.columns:
            empty["date"] = pd.to_datetime(empty["date"])       # 空表也保持 datetime 类型，避免 .dt 报错
        return empty
    df = pd.read_csv(p, dtype={"code": str, "index_code": str})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def append_ds(name, new: pd.DataFrame) -> int:
    if new is None or new.empty:
        return 0
    old = read_ds(name)
    new = new[SCHEMAS[name]].copy()
    allx = pd.concat([old, new], ignore_index=True) if len(old) else new
    allx = allx.drop_duplicates(KEYS[name], keep="last")
    if "date" in allx.columns:
        allx = allx.sort_values("date")
    allx.to_csv(_path(name), index=False)
    return len(new)


def cache_stamp() -> float:
    return sum(p.stat().st_mtime for p in config.CACHE_DIR.glob("*.csv"))


def health() -> pd.DataFrame:
    rows = []
    for name in SCHEMAS:
        df = read_ds(name)
        if "date" in df.columns and len(df):
            rows.append({"数据集": name, "行数": len(df), "起": df["date"].min().date(),
                         "止": df["date"].max().date(), "交易日数": df["date"].nunique()})
        else:
            rows.append({"数据集": name, "行数": len(df), "起": None, "止": None, "交易日数": 0})
    return pd.DataFrame(rows)


# ───────────── 通用解析工具 ─────────────
def _clean_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[\s()（）]", "", str(c)) for c in df.columns]
    return df


def _find(cols, *must, any_of=(), none_of=()):
    for c in cols:
        if all(m in c for m in must) and (not any_of or any(a in c for a in any_of)) \
                and not any(n in c for n in none_of):
            return c
    return None


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "gbk"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


class ParseError(Exception):
    pass


def ak_version() -> str:
    try:
        import akshare as ak
        return getattr(ak, "__version__", "未知")
    except Exception:  # noqa
        return "未安装"


def explain(e: Exception) -> str:
    """把"akshare 缺接口"这类错误翻译成能直接照做的提示。"""
    msg = str(e)
    if isinstance(e, AttributeError) and "akshare" in msg:
        return f"{msg}。当前 akshare 版本 {ak_version()} 过旧，缺少该接口：请运行 pip install -U akshare 后重启网页"
    return msg


# ───────────── 中金所：前20会员持仓排名 ─────────────
def parse_cffex_rank(text: str, d) -> pd.DataFrame:
    """解析中金所"成交持仓排名"CSV（含多个合约，每个合约按名次列出成交量/持买/持卖三组）。"""
    df = _clean_cols(pd.read_csv(io.StringIO(text), skipinitialspace=True))
    cols = list(df.columns)
    c_contract = _find(cols, "合约")
    c_rank = _find(cols, "名次")
    spec = {}
    for key, kw in (("vol", "成交量"), ("long", "持买"), ("short", "持卖")):
        member = _find(cols, kw, any_of=("简称", "会员"))
        if member is None:
            spec[key] = (None, None, None)
            continue
        i = cols.index(member)               # 约定：会员列后面紧跟 数量列、增减列（增减列表头重复，不能靠名字找）
        spec[key] = (member, cols[i + 1], cols[i + 2]) if i + 2 < len(cols) else (None, None, None)
    if not c_contract or not c_rank or any(v is None for t in spec.values() for v in t):
        raise ParseError(f"无法识别中金所持仓排名文件的列。已识别：{cols}")
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "contract": df[c_contract].astype(str).str.strip(),
        "rank": _num(df[c_rank]),
    })
    for key, (m, v, c) in spec.items():
        out[f"{key}_member"] = df[m].astype(str).str.strip()
        out[f"{key}_oi" if key != "vol" else "vol"] = _num(df[v])
        out[f"{key}_chg" if key != "vol" else "vol_chg"] = _num(df[c])
    out = out[out["rank"].notna() & out["contract"].str.match(r"^(IF|IH|IC|IM)\d{4}$")].copy()
    out["variety"] = out["contract"].str[:2]
    return out[SCHEMAS["cffex_rank"]]


def _rank_via_akshare(d) -> pd.DataFrame:
    import akshare as ak
    res = ak.get_cffex_rank_table(date=pd.Timestamp(d).strftime("%Y%m%d"), vars_list=list(VARIETIES))
    frames = []
    for _, df in res.items():
        df = df.rename(columns={"long_party_name": "long_member", "long_open_interest": "long_oi",
                                "long_open_interest_chg": "long_chg", "short_party_name": "short_member",
                                "short_open_interest": "short_oi", "short_open_interest_chg": "short_chg",
                                "vol_party_name": "vol_member", "symbol": "contract"})
        df["date"] = pd.Timestamp(d)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=SCHEMAS["cffex_rank"])
    out = pd.concat(frames, ignore_index=True)
    out["variety"] = out["contract"].astype(str).str[:2]
    for c in SCHEMAS["cffex_rank"]:
        if c not in out.columns:
            raise ParseError(f"akshare 返回缺列 {c}，实际列：{list(out.columns)}")
    return out[SCHEMAS["cffex_rank"]]


def fetch_cffex_rank_day(d) -> pd.DataFrame:
    import requests
    d = pd.Timestamp(d)
    frames, last_err = [], None
    for v in VARIETIES:
        url = f"http://www.cffex.com.cn/sj/ccpm/{d:%Y%m}/{d:%d}/{v}_1.csv"
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code == 200 and len(r.content) > 80:
                frames.append(parse_cffex_rank(_decode(r.content), d))
        except Exception as e:  # noqa
            last_err = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    try:
        return _rank_via_akshare(d)
    except Exception as e:  # noqa
        raise ParseError(f"直连与 akshare 都失败（{last_err or '无响应'} / {e}）")


# ───────────── 中金所：日行情（算基差用）─────────────
def parse_cffex_daily(text: str, d) -> pd.DataFrame:
    df = _clean_cols(pd.read_csv(io.StringIO(text), skipinitialspace=True))
    cols = list(df.columns)
    c_con, c_close = _find(cols, "合约"), _find(cols, "收盘")
    c_set = _find(cols, "结算", none_of=("前",))
    c_vol, c_oi = _find(cols, "成交量"), _find(cols, "持仓量")
    if not all([c_con, c_close, c_oi]):
        raise ParseError(f"无法识别中金所日行情文件的列。已识别：{cols}")
    out = pd.DataFrame({"date": pd.Timestamp(d), "contract": df[c_con].astype(str).str.strip(),
                        "close": _num(df[c_close]), "settle": _num(df[c_set]) if c_set else np.nan,
                        "volume": _num(df[c_vol]) if c_vol else np.nan, "oi": _num(df[c_oi])})
    out = out[out["contract"].str.match(r"^(IF|IH|IC|IM)\d{4}$")].copy()
    out["variety"] = out["contract"].str[:2]
    return out[SCHEMAS["cffex_daily"]]


def fetch_cffex_daily_day(d) -> pd.DataFrame:
    import requests
    d = pd.Timestamp(d)
    url = f"http://www.cffex.com.cn/sj/hqsj/rtj/{d:%Y%m}/{d:%d}/{d:%Y%m%d}_1.csv"
    err = None
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code == 200 and len(r.content) > 80:
            return parse_cffex_daily(_decode(r.content), d)
    except Exception as e:  # noqa
        err = e
    try:
        import akshare as ak
        df = ak.get_futures_daily(start_date=d.strftime("%Y%m%d"), end_date=d.strftime("%Y%m%d"), market="CFFEX")
        df = df.rename(columns={"symbol": "contract", "open_interest": "oi"})
        df["date"] = d
        df["variety"] = df["contract"].astype(str).str[:2]
        df = df[df["contract"].astype(str).str.match(r"^(IF|IH|IC|IM)\d{4}$")]
        return df[SCHEMAS["cffex_daily"]]
    except Exception as e:  # noqa
        raise ParseError(f"直连与 akshare 都失败（{err or '无响应'} / {e}）")


# ───────────── 融资融券（交易所明细，非东财）─────────────
def normalize_margin(df: pd.DataFrame, d) -> pd.DataFrame:
    df = _clean_cols(df)
    cols = list(df.columns)
    c_code = _find(cols, "证券代码") or _find(cols, "代码")
    c_bal = _find(cols, "融资余额")
    c_buy = _find(cols, "融资买入额")
    c_rep = _find(cols, "融资偿还额")
    if not all([c_code, c_bal, c_buy]):
        raise ParseError(f"无法识别融资融券明细的列。已识别：{cols}")
    out = pd.DataFrame({"date": pd.Timestamp(d),
                        "code": df[c_code].astype(str).str.extract(r"(\d{6})")[0],
                        "rz_balance": _num(df[c_bal]), "rz_buy": _num(df[c_buy]),
                        "rz_repay": _num(df[c_rep]) if c_rep else np.nan})
    return out.dropna(subset=["code"])[SCHEMAS["margin"]]


def fetch_margin_day(d) -> pd.DataFrame:
    import akshare as ak
    s, frames, errs = pd.Timestamp(d).strftime("%Y%m%d"), [], []
    for fn in ("stock_margin_detail_sse", "stock_margin_detail_szse"):
        f = getattr(ak, fn, None)
        if f is None:
            errs.append(f"akshare 无 {fn}")
            continue
        try:
            df = f(date=s)
            if df is not None and len(df):
                frames.append(normalize_margin(df, d))
        except Exception as e:  # noqa
            errs.append(f"{fn}: {e}")
    if not frames:
        raise ParseError("；".join(errs) or "当日无数据（可能非交易日）")
    return pd.concat(frames, ignore_index=True)


# ───────────── ETF 份额 ─────────────
def normalize_etf_shares(df: pd.DataFrame, d) -> pd.DataFrame:
    df = _clean_cols(df)
    cols = list(df.columns)
    c_code = _find(cols, "代码")
    c_name = _find(cols, "简称") or _find(cols, "名称")
    c_sh = _find(cols, "份额")
    if not (c_code and c_sh):
        raise ParseError(f"无法识别 ETF 份额文件的列。已识别：{cols}")
    out = pd.DataFrame({"date": pd.Timestamp(d), "code": df[c_code].astype(str).str.extract(r"(\d{6})")[0],
                        "name": df[c_name].astype(str) if c_name else "", "shares": _num(df[c_sh])})
    out = out.dropna(subset=["code", "shares"])
    if len(out) and out["shares"].median() < 1e7:      # 多半是"万份"
        out["shares"] = out["shares"] * 1e4
    return out[SCHEMAS["etf_shares"]]


def _etf_shares_sse_direct(d) -> pd.DataFrame:
    """上交所官网查询接口直连（接口参数按 akshare 源码思路编写，未实测；失败会抛错并提示手动导入）。"""
    import requests
    d = pd.Timestamp(d)
    params = {"isPagination": "true", "pageHelp.pageSize": "10000", "pageHelp.pageNo": "1",
              "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "1",
              "sqlId": "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L", "STAT_DATE": d.strftime("%Y-%m-%d")}
    headers = dict(UA, Referer="http://www.sse.com.cn/market/funddata/volumn/etfvolumn/")
    r = requests.get("http://query.sse.com.cn/commonQuery.do", params=params, headers=headers, timeout=20)
    js = r.json()
    rows = js.get("result") or (js.get("pageHelp") or {}).get("data") or []
    if not rows:
        raise ParseError("上交所接口当日无数据（非交易日或尚未发布）")
    df = pd.DataFrame(rows)
    need = {"SEC_CODE", "TOT_VOL"}
    if not need <= set(df.columns):
        raise ParseError(f"上交所接口返回列与预期不符：{list(df.columns)}")
    out = pd.DataFrame({"date": d, "code": df["SEC_CODE"].astype(str).str.extract(r"(\d{6})")[0],
                        "name": df["SEC_NAME"].astype(str) if "SEC_NAME" in df.columns else "",
                        "shares": _num(df["TOT_VOL"]) * 1e4})          # TOT_VOL 单位：万份
    return out.dropna(subset=["code", "shares"])[SCHEMAS["etf_shares"]]


def fetch_etf_shares_day(d) -> pd.DataFrame:
    errs = []
    try:
        import akshare as ak
        f = getattr(ak, "fund_etf_scale_sse", None)
        if f is None:
            errs.append(f"akshare {ak_version()} 没有 fund_etf_scale_sse（建议 pip install -U akshare）")
        else:
            return normalize_etf_shares(_retry(lambda: f(date=pd.Timestamp(d).strftime("%Y%m%d")), tries=2), d)
    except Exception as e:  # noqa
        errs.append(f"akshare: {explain(e)}")
    try:
        return _etf_shares_sse_direct(d)
    except Exception as e:  # noqa
        errs.append(f"上交所直连: {e}")
    raise ParseError("；".join(errs))


# ───────────── 指数成分 ─────────────
def fetch_index_cons(index_code: str) -> pd.DataFrame:
    import akshare as ak
    df = _clean_cols(ak.index_stock_cons_csindex(symbol=index_code))
    c_code = _find(list(df.columns), "成分券代码") or _find(list(df.columns), "代码")
    c_name = _find(list(df.columns), "成分券名称") or _find(list(df.columns), "名称")
    if not c_code:
        raise ParseError(f"无法识别成分股文件的列：{list(df.columns)}")
    return pd.DataFrame({"index_code": index_code, "code": df[c_code].astype(str).str.extract(r"(\d{6})")[0],
                         "name": df[c_name] if c_name else ""}).dropna(subset=["code"])


# ───────────── 数据源选择：Tushare / 免费源 ─────────────
def _sources() -> list:
    mode = getattr(config, "SOURCE", "auto")
    ts_ok = tsrc.available()
    if mode == "tushare":
        return ["tushare"] if ts_ok else []
    if mode == "free":
        return ["free"]
    return (["tushare"] if ts_ok else []) + ["free"]


def _with_sources(free_fn, ts_fn):
    def fn(d):
        errs = []
        order = _sources()
        if not order:
            raise ParseError("数据源设为仅 Tushare，但没有配置 token")
        for src in order:
            try:
                return (ts_fn if src == "tushare" else free_fn)(d)
            except Exception as e:  # noqa
                errs.append(f"{'Tushare' if src == 'tushare' else '免费源'}: {explain(e)}")
        raise ParseError("；".join(errs))
    return fn


# ───────────── 交易日历（所有"按日下载"都以它为准，保证和日线同一颗粒度）─────────────
def trading_days(start, end) -> list:
    """A股交易日列表。优先用新浪交易日历（含节假日安排），失败则退化为工作日。"""
    p = config.CACHE_DIR / "trade_calendar.csv"
    end = pd.Timestamp(end).normalize()
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(p)["trade_date"])) if p.exists() else None
    if cal is None or cal.max() < end:
        got = None
        if "tushare" in _sources():
            try:
                got = pd.DatetimeIndex(tsrc.trade_dates("20150101", end + pd.Timedelta(days=365)))
            except Exception:  # noqa
                got = None
        if got is None:
            try:
                import akshare as ak
                got = pd.DatetimeIndex(pd.to_datetime(_retry(lambda: ak.tool_trade_date_hist_sina(), tries=2)["trade_date"]))
            except Exception:  # noqa
                got = None
        if got is not None and len(got):
            cal = got
            pd.DataFrame({"trade_date": cal}).to_csv(p, index=False)
        elif cal is None:
            cal = pd.bdate_range(pd.Timestamp(start), end)
    return list(cal[(cal >= pd.Timestamp(start)) & (cal <= end)])


def cached_trading_days(start, end) -> list:
    """只读本地缓存的交易日历（不联网）；没有缓存就退化为工作日。用于"数据新鲜度"判断。"""
    p = config.CACHE_DIR / "trade_calendar.csv"
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if p.exists():
        cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(p)["trade_date"]))
        if len(cal) and cal.max() >= e:
            return list(cal[(cal >= s) & (cal <= e)])
    return list(pd.bdate_range(s, e))


FETCHERS = {"cffex_rank": _with_sources(fetch_cffex_rank_day, tsrc.fut_rank_day),
            "cffex_daily": _with_sources(fetch_cffex_daily_day, tsrc.fut_daily_day),
            "margin": _with_sources(fetch_margin_day, tsrc.margin_day),
            "etf_shares": fetch_etf_shares_day}


REFRESH_LAST = 3     # 增量更新时，已入库的最近 N 个交易日也重抓覆盖（交易所/数据源可能先放出一部分，之后才补全）


def update_range(name: str, start, end, progress=None, refresh_last: int = 0) -> list:
    """按交易日逐日抓取并入库。已有的日期默认跳过；refresh_last>0 时，已入库的最近 N 个交易日会重新抓取并覆盖
    （避免"早上抓到的是不完整的当日数据，之后再点下载却因日期已存在被跳过"）。重抓不到（尚未发布）不算错误，旧数据保留。
    开头连续 3 天都因同一原因失败时提前停止，不再刷屏。"""
    cal = trading_days(start, end)
    have = set(read_ds(name)["date"].dt.date)
    have_days = [d for d in cal if d.date() in have]
    refresh = {d.date() for d in have_days[-refresh_last:]} if refresh_last else set()
    days = [d for d in cal if d.date() not in have or d.date() in refresh]
    log, ok, fail, refreshed, first_errs, stopped = [], 0, 0, 0, [], False
    for i, d in enumerate(days):
        again = d.date() in refresh
        try:
            n = append_ds(name, FETCHERS[name](d))
            if again:
                refreshed += 1 if n else 0
            else:
                ok += 1 if n else 0
        except Exception as e:  # noqa
            if again:
                pass        # 重抓不到：保留已有数据，不计失败
            else:
                fail += 1
                if fail <= 3:
                    log.append(f"  {d.date()}: {explain(e)}")
                if ok == 0:
                    first_errs.append(str(e))
                    if len(first_errs) >= 3 and len(set(first_errs[-3:])) == 1:
                        stopped = True
                        break
        if progress:
            progress((i + 1) / max(len(days), 1), f"{name} {d.date()}")
    head = f"{name}: 新增 {ok} 个交易日" + (f"，重抓最近 {refreshed} 天" if refresh_last else "") + f"，失败/尚未发布 {fail} 天"
    if stopped:
        head += f"；连续失败原因相同，已停止（共 {len(days)} 天待抓）。可修好后重试，或在侧边栏手动导入官网文件"
    log.insert(0, head)
    return log


def incremental_update(name: str, backfill_days: int = 60, progress=None) -> list:
    """增量：已有数据 → 从最后一天往前 10 个交易日起补（顺便补洞）；没有 → 回补最近 backfill_days 个交易日。"""
    end = pd.Timestamp.today().normalize()
    df = read_ds(name)
    if name == "etf_shares" and "tushare" in _sources():          # 按代码取整段历史，十几次调用
        start0 = (df["date"].max() - pd.Timedelta(days=15)) if len(df) else end - pd.Timedelta(days=int(backfill_days * 1.6) + 10)
        try:
            codes = sorted({c for v in config.ETF_GROUPS.values() for c in v})
            n = append_ds("etf_shares", tsrc.etf_shares_history(codes, start0))
            return [f"etf_shares: Tushare 按代码取 {len(codes)} 只 ETF，写入 {n} 行"]
        except Exception as e:  # noqa
            if _sources() == ["tushare"]:
                return [f"etf_shares: Tushare 失败 {explain(e)}"]
            first = f"etf_shares: Tushare 失败（{explain(e)[:120]}），改用免费源"
            return [first] + _incremental_by_day(name, df, end, backfill_days, progress)
    return _incremental_by_day(name, df, end, backfill_days, progress)


def _incremental_by_day(name, df, end, backfill_days, progress):
    if len(df):
        cal = trading_days(df["date"].max() - pd.Timedelta(days=30), end)
        start = cal[-11] if len(cal) > 10 else cal[0]
    else:
        cal = trading_days(end - pd.Timedelta(days=int(backfill_days * 1.6) + 10), end)
        start = cal[-backfill_days] if len(cal) > backfill_days else cal[0]
    return update_range(name, start, end, progress, refresh_last=REFRESH_LAST)


def update_index_cons(progress=None) -> list:
    log = []
    for nm, info in config.INDEXES.items():
        errs, ok = [], False
        for src in _sources():
            try:
                df = tsrc.index_cons(info["code"]) if src == "tushare" else fetch_index_cons(info["code"])
                append_ds("index_cons", df)
                log.append(f"  {nm}: 成分股 {len(df)} 只 [{'Tushare' if src == 'tushare' else '免费源'}]")
                ok = True
                break
            except Exception as e:  # noqa
                errs.append(f"{'Tushare' if src == 'tushare' else '免费源'}: {explain(e)[:140]}")
        if not ok:
            log.append(f"  {nm}: 失败 " + "；".join(errs))
    return log


# ───────────── ① 融资融券 ─────────────
def download_margin(backfill_days=60, progress=None) -> list:
    log = ["【① 融资融券】个股明细，每个交易日一行/只股票（与日线同一颗粒度）"]
    if read_ds("index_cons").empty:
        log += update_index_cons()
    return log + incremental_update("margin", backfill_days, progress)


# ───────────── ② 期指（持仓排名 + 日行情）─────────────
def download_futures(backfill_days=60, progress=None) -> list:
    log = ["【② 期指】前20会员持仓排名 + 合约日行情，按交易日"]
    log += incremental_update("cffex_daily", backfill_days, progress)
    log += incremental_update("cffex_rank", backfill_days, progress)
    return log


# ───────────── ③④ 日线价格（指数 / ETF / 个股）→ 统一格式 ─────────────
PX_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
_PX_ALIAS = {"date": ["date", "日期", "trade_date"], "open": ["open", "开盘", "今开"], "high": ["high", "最高"],
             "low": ["low", "最低"], "close": ["close", "收盘", "最新价", "latest", "最新"],
             "volume": ["volume", "成交量"], "amount": ["amount", "成交额", "成交金额"]}


def normalize_price(df: pd.DataFrame) -> pd.DataFrame:
    """任意来源日线 → date/open/high/low/close/volume/amount，一行一个交易日，按日期升序去重。"""
    cols = {str(c).strip().lower(): c for c in df.columns}
    out = pd.DataFrame()
    for std, names in _PX_ALIAS.items():
        src = next((cols[n.lower()] for n in names if n.lower() in cols), None)
        out[std] = df[src] if src is not None else np.nan
    if out["date"].isna().all():
        raise ParseError(f"日线缺日期列：{list(df.columns)}")
    out["date"] = pd.to_datetime(out["date"])
    for c in PX_COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _call_with_timeout(fn, seconds):
    """部分接口（如 akshare 的中行牌价）没设超时会挂 45 秒以上，这里强制超时。"""
    if not seconds:
        return fn()
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    except FTimeout:
        raise TimeoutError(f"请求超过 {seconds} 秒无响应")
    finally:
        ex.shutdown(wait=False)


def _retry(fn, tries=3, base=1.5, timeout=40):
    """东财接口对连续请求会直接断开连接(RemoteDisconnected)，间隔递增地重试几次；每次带超时保护。"""
    last = None
    for i in range(tries):
        try:
            return _call_with_timeout(fn, timeout)
        except Exception as e:  # noqa
            last = e
            if i < tries - 1:
                time.sleep(base * (2 ** i) + random.random())
    raise last


def _sina_symbol(code: str, kind: str) -> str:
    if kind == "etf":
        return ("sh" if code.startswith(("5", "6")) else "sz") + code
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _fetch_price_raw(code: str, kind: str, start: str):
    """返回 (原始DataFrame, 数据源名)。东财失败（重试后）自动改用新浪。"""
    import akshare as ak
    end = pd.Timestamp.today().strftime("%Y%m%d")
    if kind == "index":
        try:
            return _retry(lambda: ak.stock_zh_index_daily_em(symbol=f"sh{code}")), "东财"
        except Exception:  # noqa
            return _retry(lambda: ak.stock_zh_index_daily(symbol=f"sh{code}"), tries=2), "新浪"
    if kind == "etf":
        em = lambda: ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=end, adjust="")  # noqa
        sina = lambda: ak.fund_etf_hist_sina(symbol=_sina_symbol(code, kind))  # noqa
    else:
        em = lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")  # noqa
        sina = lambda: ak.stock_zh_a_daily(symbol=_sina_symbol(code, kind), adjust="")  # noqa
    try:
        return _retry(em), "东财"
    except Exception as e1:  # noqa
        try:
            return _retry(sina, tries=2), "新浪"
        except Exception as e2:  # noqa
            raise RuntimeError(f"东财失败({e1})；新浪也失败({e2})")


def download_price(code: str, kind: str) -> str:
    """增量下载单个代码，合并写入 cache/px/{code}.csv（成交额为"元"口径）。返回一行日志。"""
    px = config.CACHE_DIR / "px"
    px.mkdir(exist_ok=True)
    f = px / (f"index_{code}.csv" if kind == "index" else f"{code}.csv")      # 指数与个股同号时不会互相覆盖
    old = pd.read_csv(f, parse_dates=["date"]) if f.exists() else None
    start = config.PX_START if old is None or old.empty else (old["date"].max() - pd.Timedelta(days=7)).strftime("%Y%m%d")
    new, src, errs = None, "", []
    for s_ in _sources():
        try:
            if s_ == "tushare":
                new, src = tsrc.fetch_price(code, kind, start), "Tushare"
            else:
                raw, src = _fetch_price_raw(code, kind, start)
                new = normalize_price(raw)
            break
        except Exception as e:  # noqa
            errs.append(f"{'Tushare' if s_ == 'tushare' else '免费源'}: {explain(e)}")
    if new is None:
        return f"  {code}({kind}): 失败 " + ("；".join(errs) or "没有可用数据源（请配置 Tushare token）")
    note = ""
    if src == "新浪" and new["amount"].isna().all():            # 新浪源没有成交额：用 收盘价×成交量 近似
        new["amount"] = new["close"] * new["volume"]
        note = "（新浪源，成交额=收盘价×成交量 近似）"
    new = new[new["date"] >= pd.Timestamp(config.PX_START)]
    allx = pd.concat([old, new], ignore_index=True) if old is not None else new
    allx = allx.sort_values("date").drop_duplicates("date", keep="last")
    allx.to_csv(f, index=False)
    return f"  {code}({kind}): {len(allx)} 行，至 {allx['date'].max().date()} [{src}]{note}"


def price_targets(include_sector=True, include_cons=False) -> dict:
    t = {"③ 四大指数": [(v["code"], "index") for v in config.INDEXES.values()]}
    etfs = sorted({c for v in config.ETF_GROUPS.values() for c in v} | set(config.REF_ETFS.values()))
    t["④ ETF"] = [(c, "etf") for c in etfs]
    if include_sector:
        t["④ 板块个股"] = [(c, "stock") for s in config.SECTORS.values() for c, _ in s["stocks"]]
    if include_cons:
        cons = read_ds("index_cons")
        t["④ 指数成分股(融资占比分母)"] = [(c, "stock") for c in sorted(set(cons["code"]))]
    return t


def download_prices(group_prefix: str, include_sector=True, include_cons=False, progress=None) -> list:
    log = []
    for gname, items in price_targets(include_sector, include_cons).items():
        if not gname.startswith(group_prefix):
            continue
        log.append(f"【{gname}】共 {len(items)} 个，日线统一为 date/open/high/low/close/volume/amount")
        for i, (code, kind) in enumerate(items):
            if i:                                                  # 请求之间留间隔，降低被东财断开连接的概率
                time.sleep(getattr(config, "PX_SLEEP", 0.4) + random.random() * 0.3)
            log.append(download_price(code, kind))
            if progress:
                progress((i + 1) / len(items), f"{gname} {code}")
    return log


# ───────────── ⑤ 外资代理指标 ─────────────
def _boc_mid(ak):
    """备用：央行人民币对美元中间价（新浪-中行外汇牌价历史）。注意是在岸中间价，不是离岸即期，保存为 USDCNY_MID。"""
    raw = ak.currency_boc_sina(symbol="美元", start_date=config.PX_START, end_date=pd.Timestamp.today().strftime("%Y%m%d"))
    col = next((c for c in raw.columns if "中间价" in str(c)), None)
    dcol = next((c for c in raw.columns if "日期" in str(c) or str(c).lower() == "date"), None)
    if col is None or dcol is None:
        raise ParseError(f"中行牌价返回列与预期不符：{list(raw.columns)}")
    close = pd.to_numeric(raw[col], errors="coerce")
    if close.median() > 50:                                    # 牌价按"每100美元"报价
        close = close / 100
    return pd.DataFrame({"date": raw[dcol], "close": close})


def proxy_candidates():
    """每个指标按优先级给出多个来源：(指标, 说明, [(来源名, 取数函数, 保存时的代码[None=同指标]), ...])。
    已按实测结果剔除不可用来源（新浪 XIN9、新浪 HSAHP）。东财接口在部分网络下断连，所以排在后面。"""
    ts = "tushare" in _sources()
    a50 = [(f"新浪 {config.A50_SYMBOL}", lambda ak: ak.futures_foreign_hist(symbol=config.A50_SYMBOL), None),
           ("东财 国际期货", lambda ak: ak.futures_global_hist_em(symbol="富时中国A50指数当月连续"), None)]
    usd = [("东财", lambda ak: ak.forex_hist_em(symbol="USDCNH"), None),
           ("中行牌价·央行中间价(替代,存为USDCNY_MID)", _boc_mid, "USDCNY_MID")]
    hsi = [("新浪", lambda ak: ak.stock_hk_index_daily_sina(symbol="HSI"), None),
           ("东财", lambda ak: ak.stock_hk_index_daily_em(symbol="HSI"), None)]
    if ts:
        a50.append(("Tushare 富时A50现货指数(替代,存为A50_IDX)", lambda ak: tsrc.index_global_hist("XIN9"), "A50_IDX"))
        usd.insert(0, ("Tushare fx_daily", lambda ak: tsrc.fx_hist("USDCNH.FXCM"), None))
        hsi.append(("Tushare index_global", lambda ak: tsrc.index_global_hist("HSI"), None))
    return [
        ("A50", "富时中国A50期指", a50),
        ("USDCNH", "离岸人民币", usd),
        ("HSAHP", "恒生AH股溢价指数", [("东财", lambda ak: ak.stock_hk_index_daily_em(symbol="HSAHP"), None)]),
        ("HSI", "恒生指数", hsi),
        ("FXI", "中国大盘ETF(美股)", [("新浪", lambda ak: ak.stock_us_daily(symbol="FXI"), None)]),
        ("KWEB", "中概互联网ETF(美股)", [("新浪", lambda ak: ak.stock_us_daily(symbol="KWEB"), None)]),
    ]


def download_proxies(progress=None) -> list:
    """每个指标一行/其自身市场的交易日（A50/港股/美股日历与A股不同，这里保留原日期，对齐留到分析阶段）。
    每个指标依次尝试多个来源，成功即止；全部失败时在日志里列出每个来源的原因。"""
    import akshare as ak
    log = [f"【⑤ 外资代理】保留各市场原始交易日，不在下载时强行对齐A股日历（akshare {ak_version()}）"]
    specs = proxy_candidates()
    for i, (sym, desc, cands) in enumerate(specs):
        if i:
            time.sleep(getattr(config, "PX_SLEEP", 0.4) + random.random() * 0.3)
        errs, done = [], False
        for src, fn, save_as in cands:
            try:
                df = normalize_price(_retry(lambda: fn(ak), tries=2, base=3))
                df = df[df["date"] >= pd.Timestamp(config.PX_START)]
                if df.empty:
                    raise ParseError("返回为空")
                code = save_as or sym
                out = df.assign(symbol=code)[SCHEMAS["proxy_daily"]]
                append_ds("proxy_daily", out)
                note = "" if not save_as else "  ← 这是替代指标，不是离岸即期价"
                log.append(f"  {code}（{desc}）: {len(out)} 行，至 {out['date'].max().date()} [{src}]{note}")
                done = True
                break
            except Exception as e:  # noqa
                errs.append(f"{src}: {explain(e)[:160]}")
        if not done:
            log.append(f"  {sym}（{desc}）: 失败")
            log += [f"      · {e}" for e in errs]
        if progress:
            progress((i + 1) / len(specs), f"外资代理 {sym}")
    return log


# ───────────── 一键更新全部 ─────────────
def download_local_stocks(progress=None) -> list:
    """⑥ 用 Tushare 把你本地日线目录里的个股日线补到最新（追加）。需要 token。"""
    if not tsrc.available():
        return ["【⑥ 本地个股日线】未配置 Tushare token（侧边栏「数据源」里填写，或设置环境变量 TUSHARE_TOKEN）"]
    try:
        return tsrc.update_local_stock_daily(progress)
    except Exception as e:  # noqa
        return [f"【⑥ 本地个股日线】失败：{explain(e)}"]


def update_all(backfill_days=60, include_sector=True, include_cons=False, include_local=True, progress=None) -> list:
    local = include_local and "tushare" in _sources()
    steps = []
    if local:
        steps.append(("⑥ 本地个股日线(Tushare批量)", lambda p: download_local_stocks(p)))
    steps += [
        ("③ 四大指数日线", lambda p: download_prices("③", progress=p)),
        # 本地批量更新已覆盖板块个股和成分股，这里只下载 ETF
        ("④ ETF/板块个股日线", lambda p: download_prices("④", include_sector and not local, include_cons and not local, progress=p)),
        ("② 期指", lambda p: download_futures(backfill_days, p)),
        ("① 融资融券", lambda p: download_margin(backfill_days, p)),
        ("ETF份额", lambda p: incremental_update("etf_shares", backfill_days, p)),
        ("⑤ 外资代理", lambda p: download_proxies(p)),
    ]
    log = []
    for i, (name, fn) in enumerate(steps):
        sub = (lambda frac, text, i=i, name=name: progress((i + frac) / len(steps), f"{name} · {text}")) if progress else None
        try:
            log += fn(sub)
        except Exception as e:  # noqa
            log.append(f"{name}: 整步失败 {explain(e)}")
    return log


# ───────────── 早间补数据（只补次日早晨才披露的两类）─────────────
def update_morning(backfill_days=60, progress=None) -> list:
    """融资融券、ETF份额要到次日早晨才披露：早间只补这两类，不重跑日线 / 期指 / 外资代理。"""
    steps = [("① 融资融券", lambda p: download_margin(backfill_days, p)),
             ("ETF份额", lambda p: incremental_update("etf_shares", backfill_days, p))]
    log = []
    for i, (name, fn) in enumerate(steps):
        sub = (lambda frac, text, i=i, name=name: progress((i + frac) / len(steps), f"{name} · {text}")) if progress else None
        try:
            log += fn(sub)
        except Exception as e:  # noqa
            log.append(f"{name}: 整步失败 {explain(e)}")
    return log


# ───────────── 颗粒度对齐检查 ─────────────
def alignment_report() -> pd.DataFrame:
    """以沪深300日线的交易日为基准，检查每个数据集是否"一个交易日一份数据"、有无缺日/多日/滞后。"""
    import loaders
    bench = loaders.load_price(config.INDEXES[config.BENCH]["code"])
    if bench is None or bench.empty:
        return pd.DataFrame([{"数据集": "基准", "状态": "缺少沪深300日线，无法检查对齐（先点 ③）"}])
    cal_all = pd.DatetimeIndex(bench.index)
    rows = []

    def check(name, label, dates, foreign=False):
        dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(dates)))))
        if not len(dates):
            rows.append({"数据集": name, "对象": label, "状态": "无数据"})
            return
        cal = cal_all[(cal_all >= dates.min()) & (cal_all <= dates.max())]
        miss, extra = cal.difference(dates), dates.difference(cal_all)
        lag = int((cal_all > dates.max()).sum())
        ok = not len(miss) and not len(extra)
        rows.append({"数据集": name, "对象": label, "起": dates.min().date(), "止": dates.max().date(),
                     "应有交易日": len(cal), "实有": len(dates.intersection(cal)), "缺失": len(miss),
                     "缺失示例": ", ".join(str(d.date()) for d in miss[:3]), "多出非交易日": len(extra),
                     "落后基准(交易日)": lag,
                     "状态": ("ℹ️ 外盘日历与A股不同，正常" if foreign else ("✅ 对齐" if ok and lag <= 1 else "⚠️ 需补数据"))})

    for name in ("cffex_rank", "cffex_daily", "etf_shares", "margin"):
        df = read_ds(name)
        if len(df):
            check(name, "全部", df["date"])
        else:
            rows.append({"数据集": name, "对象": "全部", "状态": "无数据"})
    m = read_ds("margin")
    if len(m):
        cnt = m.groupby("date")["code"].nunique()
        bad = cnt[cnt < 0.5 * cnt.median()]
        rows.append({"数据集": "margin", "对象": "单日股票数", "实有": int(cnt.median()), "缺失": len(bad),
                     "缺失示例": ", ".join(str(d.date()) for d in bad.index[:3]),
                     "状态": "✅ 正常" if bad.empty else "⚠️ 个别交易日股票数偏少(可能只抓到一个交易所)"})
    px_codes = [v["code"] for v in config.INDEXES.values()] + sorted(
        {c for v in config.ETF_GROUPS.values() for c in v} | set(config.REF_ETFS.values()))
    for c in px_codes:
        d = loaders.load_price(c)
        if d is not None and len(d):
            check("日线", c, d.index)
    pr = read_ds("proxy_daily")
    for sym, g in pr.groupby("symbol"):
        check("外资代理", sym, g["date"], foreign=True)
    return pd.DataFrame(rows)


# ───────────── 手动导入（官网下载的文件）─────────────
def import_file(name: str, uploaded_name: str, content: bytes, d=None) -> str:
    """name 取 cffex_rank / cffex_daily / margin / etf_shares / index_cons。"""
    if uploaded_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
        text = df.to_csv(index=False)
    else:
        text = _decode(content)
        df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    if name == "cffex_rank":
        n = append_ds(name, parse_cffex_rank(text, d))
    elif name == "cffex_daily":
        n = append_ds(name, parse_cffex_daily(text, d))
    elif name == "margin":
        n = append_ds(name, normalize_margin(df, d))
    elif name == "etf_shares":
        n = append_ds(name, normalize_etf_shares(df, d))
    elif name == "index_cons":
        df = _clean_cols(df)
        c = _find(list(df.columns), "成分券代码") or _find(list(df.columns), "代码")
        idc = d  # 这里 d 传指数代码
        n = append_ds(name, pd.DataFrame({"index_code": idc, "code": df[c].astype(str).str.extract(r"(\d{6})")[0],
                                          "name": ""}).dropna(subset=["code"]))
    else:
        raise ValueError(name)
    return f"{uploaded_name}: 导入 {n} 行"
