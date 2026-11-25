"""本地日线读取：指数 / ETF / 个股 csv（或 parquet），列名中英文都认。"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config

_ALIAS = {
    "date": ["date", "trade_date", "日期", "交易日期", "时间", "datetime"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价", "close_price"],
    "volume": ["volume", "vol", "成交量"],
    "amount": ["amount", "turnover", "成交额", "成交金额"],
}
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


_SUBSTR = {"date": ["date", "日期", "交易日"], "close": ["收盘", "close"], "open": ["开盘", "open"],
           "high": ["最高", "high"], "low": ["最低", "low"], "volume": ["成交量", "volume"],
           "amount": ["成交额", "成交金额", "amount", "turnover"]}


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """先按别名精确匹配；匹配不上再按关键字子串兜底（如"收盘价(点)"、"Close Price"）。"""
    rename, used = {}, set()
    lower = {str(c).strip().lower(): c for c in df.columns}
    for std, names in _ALIAS.items():
        for n in names:
            if n.lower() in lower and lower[n.lower()] not in used:
                rename[lower[n.lower()]] = std
                used.add(lower[n.lower()])
                break
    for std, keys in _SUBSTR.items():
        if std in rename.values():
            continue
        for low, orig in lower.items():
            if orig not in used and "前" not in low and "adj" not in low and any(k in low for k in keys):
                rename[orig] = std
                used.add(orig)
                break
    return df.rename(columns=rename)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = None
        for enc in ("utf-8-sig", "gbk"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"无法读取 {path}")
    df = _std_cols(df)
    if "date" not in df.columns:
        raise ValueError(f"{path.name} 缺少日期列，已识别列：{list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    return df


@lru_cache(maxsize=1)
def _file_index() -> dict:
    """扫描一次本地目录，建立 代码 -> 文件路径 的索引。"""
    idx = {}
    roots = [config.DAILY_DIR, config.CACHE_DIR / "px"]
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in (".csv", ".parquet"):
                m = _CODE_RE.search(p.stem)
                if m:
                    idx.setdefault(m.group(1), []).append(p)     # 本地目录在前，下载缓存在后
    return idx


def _is_index_code(code: str) -> bool:
    return code in {v["code"] for v in config.INDEXES.values()}


def _looks_like_index_file(p: Path) -> bool:
    """指数代码 000016/000905/000852 同时也是个股代码(深康佳A/厦门港务/石化机械)，
    所以指数文件必须放在名字含 index/idx/指数 的子目录，或文件名含这些词（自动下载的是 index_代码.csv）。"""
    for root in (config.DAILY_DIR, config.CACHE_DIR / "px"):
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        return any(k in part.lower() for part in rel.parts for k in ("index", "idx", "指数"))
    return False


def _candidates(code: str) -> list:
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
    return base


def diagnose(codes) -> pd.DataFrame:
    """逐个代码说明：找到哪些文件、识别出哪些列、为什么读不出来。"""
    rows = []
    for c in codes:
        paths = _candidates(c)
        for q in set(_file_index().get(c, [])) - set(paths):
            rows.append({"代码": c, "状态": "⚠️ 已忽略：该代码同时是个股代码，无法确认是指数。指数日线请放进 index 子目录，或点 ③ 自动下载",
                         "文件": str(q), "原始列": "", "最后日期": None})
        if not paths:
            rows.append({"代码": c, "状态": "❌ 没有可用的指数文件（点 ③ 下载，或放进 index 子目录）" if _is_index_code(c)
                         else "❌ 目录里没有文件名含该6位代码的 csv/parquet", "文件": "", "原始列": "", "最后日期": None})
            continue
        for p in paths:
            try:
                raw = pd.read_csv(p, encoding="utf-8-sig", nrows=3) if p.suffix.lower() == ".csv" else pd.read_parquet(p).head(3)
            except Exception:  # noqa
                try:
                    raw = pd.read_csv(p, encoding="gbk", nrows=3)
                except Exception as e:  # noqa
                    rows.append({"代码": c, "状态": f"❌ 文件读不出来：{e}", "文件": str(p), "原始列": "", "最后日期": None})
                    continue
            std = _std_cols(raw)
            miss = [k for k in ("date", "close") if k not in std.columns]
            last = None
            if not miss:
                try:
                    last = _read_any(p).index.max().date()
                except Exception:  # noqa
                    pass
            rows.append({"代码": c, "状态": "✅ 可用" if not miss else f"❌ 没识别出列：{'、'.join(miss)}",
                         "文件": str(p), "原始列": ", ".join(map(str, raw.columns)), "最后日期": last})
    return pd.DataFrame(rows)


def load_panel(codes, col="close", tail=None) -> pd.DataFrame:
    """返回 日期 x 代码 的宽表；本地没有的代码直接缺列。"""
    series = {}
    for c in codes:
        df = load_price(c)
        if df is not None and col in df.columns:
            s = df[col]
            series[c] = s.tail(tail) if tail else s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def tail_date(path: Path):
    """只读文件末尾取最后一个日期（5000 多个文件也很快）。"""
    try:
        if path.suffix.lower() == ".parquet":
            return _read_any(path).index.max()
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            header = [h.strip().strip('"').lower() for h in f.readline().strip().split(",")]
        di = next((i for i, h in enumerate(header) if h in [a.lower() for a in _ALIAS["date"]]), 0)
        with open(path, "rb") as f:
            try:
                f.seek(-600, 2)
            except OSError:
                f.seek(0)
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        raw = lines[-1].decode("utf-8", errors="ignore").split(",")[di].strip().strip('"')
        ts = pd.to_datetime(raw, errors="coerce")
        return None if ts is pd.NaT else ts
    except Exception:  # noqa
        return None


def last_date(code: str):
    """该代码可用日线里最晚的日期（本地 + 下载缓存取最大）。没有则 None。"""
    ds = [tail_date(p) for p in _candidates(code)]
    ds = [d for d in ds if d is not None and d == d]
    return max(ds) if ds else None


def scan_local_dir() -> pd.DataFrame:
    """扫描日线目录里每个文件的最后日期：代码, 最后日期, 路径。"""
    rows = []
    for code, paths in _file_index().items():
        for p in paths:
            if config.CACHE_DIR in p.parents:
                continue
            rows.append({"代码": code, "最后日期": tail_date(p), "路径": str(p)})
    return pd.DataFrame(rows, columns=["代码", "最后日期", "路径"])
