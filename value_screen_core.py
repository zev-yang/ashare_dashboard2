# -*- coding: utf-8 -*-
"""价值+质量选股 · 计算层（移植自 G_Claude 的 build_value_quality_screen.py + eval_value_longhist.py）。

定位：研究工具，不是机械组合。保证"便宜 + 财务健康"的下限（避雷），最终挑 10-15 只靠你判断。
预注册口径（默认值，改动即偏离预注册）：
  域   = 全市场 剔 ST、剔流通市值最小 30%
  候选 = 域内"最便宜"的 value decile（10%） ∩ F-score >= 5
  value = 0.5*(1-PE 5年分位) + 0.5*(1-PB 5年分位)，再对 行业哑变量 + ln(流通市值) 回归取残差、转横截面 rank
  F-score 7 项：ROA>0 / CFO>0 / ΔROA>0 / CFO>NI / Δ负债率<0 / Δ毛利率>0 / Δ周转率>0（Δ = 同比上年同期）
新增的三列（现金流正年数 / 分红年数 / 20 日成交额）只作决策卡【信息】，不叠硬门槛。

和原脚本的数值等价点（tests/test_value_screen.py 里有对照测试）：
  · 只算"最后一个交易日"的 5 年分位（原来对 2010 年起每个交易日滚动排名，只取最后一行）——结果相同，快得多；
  · 同比上年同期按"报告期 −1 年"对齐（原来按行号 shift(4)）——季度连续时结果相同，缺季度时更准确。
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

import value_screen_data as vd

PCT_WIN = 1250    # 5 年滚动分位（~252*5）
PCT_MINP = 252    # 至少 1 年数据才出分位
CRIT = ['ROA>0', 'CFO>0', 'ΔROA>0', 'CFO>NI', 'Δ负债<0', 'Δ毛利>0', 'Δ周转>0']


class DataMissing(RuntimeError):
    pass


_TRACE = None   # 子进程里指向 trace 文件；每个阶段写一行（原生崩溃时最后一行就是崩溃位置）


def _mark(msg: str):
    if _TRACE is not None:
        _TRACE.write(f"{dt.datetime.now():%H:%M:%S}  {msg}\n")
        _TRACE.flush()
        try:
            os.fsync(_TRACE.fileno())
        except Exception:  # noqa
            pass


def trace_path() -> Path:
    return vd.ROOT / "output" / "last_run_trace.log"


@dataclass
class Params:
    excl_small_pct: float = 0.30   # 剔流通市值最小 30%
    cheap_pct: float = 0.10        # 最便宜 decile
    fscore_min: int = 5            # F-score 健康线
    amt_min_yi: float = 0.5        # 20 日均成交额低于它（亿元）标"流动性提示"
    exclude_bj: bool = False       # True=daily_basic 里剔除北交所（原脚本不剔，默认保持一致）


@dataclass
class Result:
    candidates: pd.DataFrame
    asof: Optional[pd.Timestamp]
    log: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    csv_path: Optional[Path] = None


# ───────────────────────── 读数据 ─────────────────────────
def _read_dir(d: Path, last: Optional[int] = None, columns=None) -> pd.DataFrame:
    fs = sorted(d.glob("*.parquet")) if d.exists() else []
    if last:
        fs = fs[-last:]
    parts = [pd.read_parquet(f, columns=columns) if columns else pd.read_parquet(f) for f in fs]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@dataclass
class Basic:
    """最近 PCT_WIN 个交易日的 PE/PB 宽表（日期×代码）+ 最新交易日的流通市值。"""
    pe: pd.DataFrame
    pb: pd.DataFrame
    circ: pd.Series      # 最新交易日 code -> 流通市值（万元）
    cur: pd.Timestamp


def load_daily_basic(exclude_bj: bool = False) -> Basic:
    """逐个交易日读入、直接写进 float 矩阵（~5500 股 × 1250 日 ≈ 55MB/张），不保留字符串大表。
    只保留"最新交易日还在交易"的股票：其它股票最后一天没有分位，本来就进不了选股。"""
    fs = sorted(vd.DIRS["daily_basic"].glob("*.parquet"))[-PCT_WIN:]
    if not fs:
        raise DataMissing("daily_basic")
    cols = ["ts_code", "pe_ttm", "pb", "circ_mv"]

    def _read(f):
        df = pd.read_parquet(f, columns=cols)
        if exclude_bj:
            df = df[~df["ts_code"].astype(str).str.endswith(".BJ")]
        df = df.assign(code=df["ts_code"].astype(str).str[:6]).drop_duplicates("code", keep="last")
        return df.set_index("code")

    last = _read(fs[-1])
    universe = last.index
    pe = np.full((len(fs), len(universe)), np.nan)
    pb = np.full_like(pe, np.nan)
    dates = []
    for i, f in enumerate(fs):
        df = last if i == len(fs) - 1 else _read(f).reindex(universe)
        pe[i] = pd.to_numeric(df["pe_ttm"], errors="coerce").to_numpy(float)
        pb[i] = pd.to_numeric(df["pb"], errors="coerce").to_numpy(float)
        dates.append(pd.Timestamp(f.stem))
    idx = pd.DatetimeIndex(dates)
    return Basic(pe=pd.DataFrame(pe, index=idx, columns=universe),
                 pb=pd.DataFrame(pb, index=idx, columns=universe),
                 circ=pd.to_numeric(last["circ_mv"], errors="coerce"), cur=idx[-1])


def load_industry() -> pd.DataFrame:
    if not vd.IND_FILE.exists():
        raise DataMissing("industry")
    return pd.read_parquet(vd.IND_FILE).drop_duplicates("code").set_index("code")


# ───────────────────────── value：5 年分位 + 中性化 ─────────────────────────
def latest_pct(w: pd.DataFrame, win: int = PCT_WIN, minp: int = PCT_MINP) -> pd.Series:
    """等价于 w.rolling(win, min_periods=minp).rank(pct=True).iloc[-1]，但只算最后一行。
    当前值在自身最近 win 个交易日里的分位（并列取平均名次）；有效观测不足 minp 则为 NaN。"""
    tail = w.iloc[-win:]
    last = tail.iloc[-1]
    n = tail.notna().sum()
    less = tail.lt(last, axis=1).sum()
    eq = tail.eq(last, axis=1).sum()          # 含自己
    pct = (less + (eq + 1.0) / 2.0) / n
    return pct.where(last.notna() & (n >= minp))


def value_percentiles(basic: Basic):
    """PE(TTM)、PB 各自的 5 年分位（亏损/负值不参与）。返回 (pe_pct, pb_pct)，index=code。"""
    return (latest_pct(basic.pe.where(basic.pe > 0)),
            latest_pct(basic.pb.where(basic.pb > 0)))


def neutralize_one(y: pd.Series, ind_map: pd.Series, lncap: pd.Series) -> pd.Series:
    """单个截面：y 对 [截距 + 行业哑变量 + ln(流通市值)] 做 OLS，取残差 → 剥掉行业 / 市值倾斜。
    用 Frisch–Waugh–Lovell 实现（行业内去均值 + 一个斜率），与"哑变量 + lstsq"的残差完全相同，
    但纯 numpy 向量运算：不建 5500×30 的稠密矩阵，也不调用 LAPACK（少一个原生崩溃点）。"""
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


# ───────────────────────── F-score + 三个信息列 ─────────────────────────
def fscore_table(today=None) -> pd.DataFrame:
    """每只股票"今天已公告的最新一期"财报的 F-score 及 7 项明细。"""
    df = _read_dir(vd.DIRS["fundamentals"])
    if df.empty:
        raise DataMissing("fundamentals")
    today = pd.Timestamp(today or dt.date.today())
    df["code"] = df["ts_code"].astype(str).str[:6]
    df["ann"] = pd.to_datetime(df["ann_date"].astype(str), errors="coerce")
    df["end"] = pd.to_datetime(df["end_date"].astype(str), errors="coerce")
    base = ['roa', 'ocfps', 'ocf_to_profit', 'debt_to_assets', 'grossprofit_margin', 'assets_turn']
    for c in base:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ann", "end", "roa"])
    df = df.sort_values(["code", "end", "ann"]).drop_duplicates(["code", "end"], keep="first")
    # 同比上年同期：把上年同期的行平移一年再按 (code, end) 对齐
    dcols = ['roa', 'debt_to_assets', 'grossprofit_margin', 'assets_turn']
    prev = df[["code", "end"] + dcols].copy()
    prev["end"] = prev["end"] + pd.DateOffset(years=1)
    df = df.merge(prev.rename(columns={c: c + "_py" for c in dcols}), on=["code", "end"], how="left")
    cc = {'ROA>0': df['roa'] > 0, 'CFO>0': df['ocfps'] > 0,
          'ΔROA>0': (df['roa'] - df['roa_py']) > 0, 'CFO>NI': df['ocf_to_profit'] > 1,
          'Δ负债<0': (df['debt_to_assets'] - df['debt_to_assets_py']) < 0,
          'Δ毛利>0': (df['grossprofit_margin'] - df['grossprofit_margin_py']) > 0,
          'Δ周转>0': (df['assets_turn'] - df['assets_turn_py']) > 0}
    for k, v in cc.items():
        df[k] = v.astype(int)
    df["fscore"] = df[list(cc)].sum(axis=1)
    sub = df[df["ann"] <= today]
    last = sub.loc[sub.groupby("code")["end"].idxmax()].set_index("code")
    last["财报期"] = last["end"].dt.strftime("%Y-%m-%d")
    return last[["fscore"] + CRIT + ["debt_to_assets", "ocfps", "roa", "财报期"]]


def cashflow_pos_years(years) -> pd.Series:
    df = _read_dir(vd.DIRS["cashflow"], columns=["ts_code", "end_date", "n_cashflow_act"])
    if df.empty:
        return pd.Series(dtype=float)
    df["code"] = df["ts_code"].astype(str).str[:6]
    df["yr"] = df["end_date"].astype(str).str[:4]
    df = df[df["yr"].isin([str(y) for y in years])].drop_duplicates(["code", "yr"], keep="last")
    df["pos"] = pd.to_numeric(df["n_cashflow_act"], errors="coerce") > 0
    return df.groupby("code")["pos"].sum().astype(int)


def dividend_years(years) -> pd.Series:
    df = _read_dir(vd.DIRS["dividend"])
    if df.empty:
        return pd.Series(dtype=float)
    df["code"] = df["ts_code"].astype(str).str[:6]
    if "div_proc" in df.columns:
        df = df[df["div_proc"].astype(str).str.contains("实施", na=False)]
    cc = "cash_div_tax" if "cash_div_tax" in df.columns else ("cash_div" if "cash_div" in df.columns else None)
    if cc is None:
        return pd.Series(dtype=float)
    df["cash"] = pd.to_numeric(df[cc], errors="coerce")
    df = df[df["cash"] > 0]
    dc = "ex_date" if "ex_date" in df.columns else "ann_date"
    df["yr"] = pd.to_datetime(df[dc].astype(str), errors="coerce").dt.year
    df = df[df["yr"].isin(list(years))]
    return df.groupby("code")["yr"].nunique()


def avg_amount_20d_yi():
    """近 20 个交易日日均成交额（亿元）。Tushare daily.amount 单位是千元：1 亿 = 1e5 千元。
    返回 (Series, 最新成交额日期)。"""
    df = _read_dir(vd.DIRS["daily_recent"], last=20, columns=["ts_code", "trade_date", "amount"])
    if df.empty:
        return pd.Series(dtype=float), None
    df["code"] = df["ts_code"].astype(str).str[:6]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.drop_duplicates(["trade_date", "code"], keep="last")
    w = df.pivot(index="trade_date", columns="code", values="amount")
    return w.mean() / 1e5, pd.Timestamp(str(df["trade_date"].astype(str).max()))


# ───────────────────────── 主流程 ─────────────────────────
def run_screen(p: Optional[Params] = None, today=None) -> Result:
    p = p or Params()
    today = pd.Timestamp(today or dt.date.today())
    log = []
    years = vd.fiscal_years(today)

    log.append("① 读 daily_basic / 行业，构建中性化 value、5 年分位、F-score、质量信息列 ...")
    _mark("读 daily_basic（逐日读 parquet，约 1250 个文件）")
    basic = load_daily_basic(p.exclude_bj)
    _mark("读行业表")
    ind_df = load_industry()
    ind, name_map = ind_df["industry"], ind_df["name"]
    st = set(ind_df.index[ind_df["name"].astype(str).str.contains("ST")])
    cur = basic.cur
    log.append(f"   数据至 {cur.date()}（回看 {len(basic.pe)} 个交易日，{basic.pe.shape[1]} 只股票）")

    _mark("计算 PE/PB 5 年分位")
    pe_pct, pb_pct = value_percentiles(basic)
    value_raw = (0.5 * (1.0 - pe_pct) + 0.5 * (1.0 - pb_pct)).dropna()
    cm = basic.circ
    lncap = np.log(cm.clip(lower=1))
    del basic
    if len(value_raw) < 50:
        raise DataMissing("daily_basic（有效 PE/PB 分位不足 50 只，历史太短？）")
    _mark("行业+市值中性化")
    value_neu = neutralize_one(value_raw, ind, lncap).rank(pct=True).dropna()

    _mark("读成交额 / 现金流 / 分红")
    amt20, amt_date = avg_amount_20d_yi()
    cfy, divy = cashflow_pos_years(years), dividend_years(years)
    log.append(f"   质量列覆盖：现金流年数 {len(cfy)} 股 | 分红年数 {len(divy)} 股 | 成交额 {int(amt20.notna().sum())} 股")
    if amt_date is not None and amt_date < cur:
        log.append(f"   ⚠ 成交额数据只到 {amt_date.date()}，比 daily_basic（{cur.date()}）旧，请更新「近期成交额」")

    domain = cm[cm >= cm.quantile(p.excl_small_pct)].index
    domain = [c for c in domain if c not in st and c in value_neu.index]
    vd_ = value_neu.reindex(domain).dropna()
    vd_rank = vd_.rank(pct=True)
    cheap = vd_[vd_ >= vd_.quantile(1 - p.cheap_pct)].index
    log.append(f"② 域 {len(domain)} 只（剔 ST {len(st)}、剔市值最小 {p.excl_small_pct:.0%}）→ 最便宜 {p.cheap_pct:.0%}：{len(cheap)} 只")

    _mark("计算 F-score")
    fdet = fscore_table(today)
    cand = [c for c in cheap if c in fdet.index and fdet.loc[c, "fscore"] >= p.fscore_min]
    log.append(f"③ 候选 = 便宜 ∩ F-score ≥ {p.fscore_min}：{len(cand)} 只")
    counts = {"数据日": cur, "域": len(domain), "最便宜": len(cheap), "候选": len(cand)}

    _mark("组装结果表")
    rows = []
    for c in cand:
        f = fdet.loc[c]
        a = float(amt20.get(c, np.nan))
        rows.append({
            "代码": c, "名称": str(name_map.get(c, c)), "行业": str(ind.get(c, "")),
            "PE分位": round(float(pe_pct.get(c, np.nan)), 2), "PB分位": round(float(pb_pct.get(c, np.nan)), 2),
            "value便宜排名": round(float(vd_rank.get(c, np.nan)), 2),
            "F分": int(f["fscore"]), "7项明细": "".join("✓" if f[k] else "·" for k in CRIT),
            "财报期": f["财报期"],
            "成交额亿20d": round(a, 2) if not np.isnan(a) else np.nan,
            "流动性提示": "⚠<%.0f万" % (p.amt_min_yi * 1e4) if (not np.isnan(a) and a < p.amt_min_yi) else "",
            "现金流正年数": int(cfy.get(c, 0)), "分红年数": int(divy.get(c, 0)),
            "负债率": round(float(f["debt_to_assets"]), 1),
            "现金流ocfps": round(float(f["ocfps"]), 2), "ROA": round(float(f["roa"]), 1),
            "流通市值亿": round(float(cm.get(c, np.nan)) / 1e4, 1),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("value便宜排名", ascending=False).reset_index(drop=True)
    _mark("完成")
    return Result(candidates=out, asof=cur, log=log, counts=counts)


def save_csv(res: Result) -> Optional[Path]:
    """把结果存到 data/cache/value_screen/output/，返回路径。"""
    if res.candidates is None or res.candidates.empty:
        return None
    d = vd.ROOT / "output"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"value_quality_screen_{pd.Timestamp(res.asof):%Y%m%d}.csv"
    res.candidates.to_csv(path, index=False, encoding="utf-8-sig")
    res.csv_path = path
    return path


# ───────────────────────── 在独立进程里运行（页面用） ─────────────────────────
class ScreenFailed(RuntimeError):
    """选股子进程异常退出（含被系统杀掉 / 原生库崩溃）。"""

    def __init__(self, returncode, stderr="", trace=""):
        super().__init__(f"退出码 {returncode}")
        self.returncode, self.stderr, self.trace = returncode, stderr or "", trace or ""

    def hint(self) -> str:
        rc = self.returncode
        if rc in (-9, 137):
            return "进程被系统强行终止，最常见原因是内存不足（OOM）。"
        if rc in (3221225477, -1073741819, -11, 139):
            return "底层库崩溃（访问冲突 / 段错误）。请把下面的错误信息和 pandas / numpy / pyarrow 版本发给我。"
        if rc == "timeout":
            return "运行超时（默认 15 分钟）。"
        return "脚本报错，详见下面的错误信息。"


def _read_trace(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")[-3000:]
    except Exception:  # noqa
        return ""


def run_in_subprocess(p: Optional[Params] = None, timeout: int = 900, cmd_prefix=None) -> Result:
    """在独立 Python 进程里跑选股（读数据、算分位、回归都在子进程）。
    好处：内存吃紧或底层库崩溃时只会让子进程退出，网页服务不会掉线，页面还能显示失败原因。"""
    p = p or Params()
    here = Path(__file__).resolve()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "result.pkl")
        tpath = trace_path()
        tpath.parent.mkdir(parents=True, exist_ok=True)
        cmd = list(cmd_prefix or [sys.executable, str(here)]) + [
            "--out", out, "--trace", str(tpath), "--excl", str(p.excl_small_pct), "--cheap", str(p.cheap_pct),
            "--fmin", str(int(p.fscore_min)), "--amt-min", str(p.amt_min_yi)]
        if p.exclude_bj:
            cmd.append("--exclude-bj")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, cwd=str(here.parent), env=env)
        except subprocess.TimeoutExpired as e:
            raise ScreenFailed("timeout", str(e.stderr or "")[-3000:], _read_trace(tpath)) from e
        if proc.returncode == 2:
            raise DataMissing((proc.stderr or "").replace("DATA_MISSING:", "").strip() or "数据不足")
        if proc.returncode != 0 or not os.path.exists(out):
            raise ScreenFailed(proc.returncode, (proc.stderr or "")[-3000:], _read_trace(tpath))
        with open(out, "rb") as f:
            d = pickle.load(f)
    return Result(candidates=d["candidates"], asof=d["asof"], log=d["log"], counts=d["counts"],
                  csv_path=Path(d["csv_path"]) if d.get("csv_path") else None)


# ───────────────────────── 命令行入口（页面用子进程调用，崩溃不会拖垮网页服务） ─────────────────────────
def _cli(argv=None) -> int:
    import argparse
    import traceback

    ap = argparse.ArgumentParser(description="价值+质量选股（结果写入 --out 指定的 pickle）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--excl", type=float, default=Params.excl_small_pct)
    ap.add_argument("--cheap", type=float, default=Params.cheap_pct)
    ap.add_argument("--fmin", type=int, default=Params.fscore_min)
    ap.add_argument("--amt-min", type=float, default=Params.amt_min_yi)
    ap.add_argument("--exclude-bj", action="store_true")
    a = ap.parse_args(argv)
    global _TRACE
    import faulthandler
    tp = Path(a.trace) if a.trace else trace_path()
    tp.parent.mkdir(parents=True, exist_ok=True)
    _TRACE = open(tp, "w", encoding="utf-8")
    faulthandler.enable(file=_TRACE, all_threads=True)    # 底层崩溃（段错误 / Windows 访问冲突）时把 Python 调用栈写进 trace
    _mark(f"启动 python {sys.version.split()[0]} pandas {pd.__version__} numpy {np.__version__}")
    try:
        res = run_screen(Params(a.excl, a.cheap, a.fmin, a.amt_min, a.exclude_bj))
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
    print("\n".join(res.log))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
