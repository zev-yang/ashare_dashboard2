# -*- coding: utf-8 -*-
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
    u = pd.read_csv(ud.UNIVERSE_FILE)
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
    seq = []
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎", {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    seq.append(("关闭 Arrow SIMD 加速", {"ARROW_USER_SIMD_LEVEL": "NONE"}))
    seq.append(("默认设置", {}))
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
