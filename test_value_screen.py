# -*- coding: utf-8 -*-
"""价值质量选股：数据层 + 计算层测试（不联网；用合成数据和假的 Tushare 接口）。
运行：python -m pytest tests/test_value_screen.py -q      （需要 pyarrow 或 fastparquet）
重点是"和 G_Claude 原脚本的算法逐项对照"：同一份数据，新代码的候选名单 / 分位 / F-score 必须一致。"""
import datetime as dt

import numpy as np
import pandas as pd

import value_screen_core as vc
import value_screen_data as vd

TODAY = pd.Timestamp("2026-11-10")


# ───────────────────────── 参考实现：G_Claude 原算法（逐字照搬逻辑） ─────────────────────────
def ref_build_value(db):
    db = db.set_index(["date", "code"]).sort_index()
    parts = []
    for col in ("pe_ttm", "pb"):
        w = db[col].where(db[col] > 0).unstack("code").sort_index()
        pct = w.rolling(vc.PCT_WIN, min_periods=vc.PCT_MINP).rank(pct=True)
        parts.append((1.0 - pct).stack())
    return (0.5 * parts[0] + 0.5 * parts[1]).rename("value")


def ref_neutralize(fac, ind_map, lncap, dates):
    out = []
    for d in dates:
        y = fac.xs(d, level="date").dropna()
        codes = y.index
        ind = ind_map.reindex(codes).fillna("UNK").astype("category")
        lc = lncap.xs(d, level="date").reindex(codes)
        D = pd.get_dummies(ind, drop_first=True).to_numpy(float)
        lc = lc.fillna(lc.median()).to_numpy(float).reshape(-1, 1)
        lc = (lc - np.nanmean(lc)) / (np.nanstd(lc) + 1e-9)
        X = np.column_stack([np.ones(len(y)), D, lc])
        yv = y.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        out.append(pd.Series(yv - X @ beta, index=pd.MultiIndex.from_arrays(
            [np.repeat(d, len(codes)), codes.to_numpy()], names=["date", "code"])))
    return pd.concat(out)


def ref_fscore(df, today):
    df = df.copy()
    df["code"] = df["ts_code"].astype(str).str[:6]
    df["ann"] = pd.to_datetime(df["ann_date"].astype(str), errors="coerce")
    df["end"] = pd.to_datetime(df["end_date"].astype(str), errors="coerce")
    for c in ["roa", "ocfps", "ocf_to_profit", "debt_to_assets", "grossprofit_margin", "assets_turn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ann", "end", "roa"]).sort_values(["code", "end"])
    g = df.groupby("code")
    cc = {"ROA>0": df["roa"] > 0, "CFO>0": df["ocfps"] > 0,
          "ΔROA>0": (df["roa"] - g["roa"].shift(4)) > 0, "CFO>NI": df["ocf_to_profit"] > 1,
          "Δ负债<0": (df["debt_to_assets"] - g["debt_to_assets"].shift(4)) < 0,
          "Δ毛利>0": (df["grossprofit_margin"] - g["grossprofit_margin"].shift(4)) > 0,
          "Δ周转>0": (df["assets_turn"] - g["assets_turn"].shift(4)) > 0}
    for k, v in cc.items():
        df[k] = v.astype(int)
    df["fscore"] = df[list(cc)].sum(axis=1)
    sub = df[df["ann"] <= today]
    return sub.loc[sub.groupby("code")["end"].idxmax()].set_index("code")[["fscore"] + vc.CRIT]


# ───────────────────────── 合成数据 ─────────────────────────
def make_universe(n=260, n_days=1320, seed=7, gap_codes=()):
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i}" if i % 2 == 0 else f"{1 + i:06d}" for i in range(n)]
    suffix = {c: (".SH" if c.startswith("6") else ".SZ") for c in codes}
    inds = [f"行业{k}" for k in range(9)]
    names = [("*ST样本%d" % i if i % 37 == 0 else "样本%d" % i) for i in range(n)]
    ind_df = pd.DataFrame({"code": codes, "ts_code": [c + suffix[c] for c in codes], "name": names,
                           "industry": [inds[i % 9] for i in range(n)], "market": "主板", "list_date": "20100101"})
    dates = pd.bdate_range(end="2026-11-09", periods=n_days)
    base_pe = rng.lognormal(2.8, 0.7, n)
    pe = base_pe * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n)), axis=0))
    pe[rng.random((n_days, n)) < 0.03] = -5.0            # 亏损 → 负 PE，不参与分位
    pe[rng.random((n_days, n)) < 0.01] = np.nan
    pb = rng.lognormal(0.7, 0.5, n) * np.exp(np.cumsum(rng.normal(0, 0.015, (n_days, n)), axis=0))
    pb = np.round(pb, 2)                                  # 取整制造并列名次
    pe = np.round(pe, 1)
    mv = rng.lognormal(12.0, 1.0, n) * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n)), axis=0))
    db = pd.DataFrame({"trade_date": np.repeat(dates.strftime("%Y%m%d"), n),
                       "ts_code": np.tile([c + suffix[c] for c in codes], n_days),
                       "pe_ttm": pe.ravel(), "pb": pb.ravel(), "circ_mv": mv.ravel()})
    # 财务：2023Q1 – 2026Q3 连续季度
    periods = vd._quarter_ends(2023, TODAY)
    rows = []
    for c in codes:
        for p in periods:
            if c in gap_codes and p == "20250630":
                continue                                     # 制造缺季度
            end = pd.Timestamp(p)
            rows.append({"ts_code": c + suffix[c], "ann_date": (end + pd.Timedelta(days=int(rng.integers(20, 55)))).strftime("%Y%m%d"),
                         "end_date": p, "roe": rng.normal(8, 5), "roa": rng.normal(3, 3),
                         "debt_to_assets": rng.uniform(20, 80), "ocfps": rng.normal(0.3, 0.5),
                         "netprofit_yoy": rng.normal(5, 20), "or_yoy": rng.normal(5, 15),
                         "grossprofit_margin": rng.normal(25, 8), "netprofit_margin": rng.normal(8, 5),
                         "q_profit_yoy": rng.normal(5, 20), "profit_to_op": rng.normal(90, 10),
                         "assets_turn": rng.uniform(0.2, 1.2), "ocf_to_profit": rng.normal(1.0, 1.0),
                         "ocf_to_or": rng.normal(8, 6), "q_roe": rng.normal(2, 2),
                         "q_gsprofit_margin": rng.normal(25, 8), "assets_yoy": rng.normal(5, 8)})
    fund = pd.DataFrame(rows)
    return db, ind_df, fund, dates


def write_universe(db, ind_df, fund, dates, rng_seed=3):
    rng = np.random.default_rng(rng_seed)
    for d, g in db.groupby("trade_date"):
        vd.DIRS["daily_basic"].mkdir(parents=True, exist_ok=True)
        g.to_parquet(vd.DIRS["daily_basic"] / f"{d}.parquet", index=False)
    vd.DIRS["industry"].mkdir(parents=True, exist_ok=True)
    ind_df.to_parquet(vd.IND_FILE, index=False)
    vd.DIRS["fundamentals"].mkdir(parents=True, exist_ok=True)
    for per, g in fund.groupby("end_date"):
        g.to_parquet(vd.DIRS["fundamentals"] / f"{per}.parquet", index=False)
    # 现金流 / 分红 / 近期成交额
    vd.DIRS["cashflow"].mkdir(parents=True, exist_ok=True)
    for y in vd.fiscal_years(TODAY):
        cf = pd.DataFrame({"ts_code": ind_df["ts_code"], "end_date": f"{y}1231",
                           "n_cashflow_act": rng.normal(1e8, 3e8, len(ind_df))})
        cf.to_parquet(vd.DIRS["cashflow"] / f"{y}1231.parquet", index=False)
    vd.DIRS["dividend"].mkdir(parents=True, exist_ok=True)
    dv = []
    for c in ind_df["ts_code"]:
        for y in vd.fiscal_years(TODAY):
            if rng.random() < 0.6:
                dv.append({"ts_code": c, "ann_date": f"{y}0420", "div_proc": "实施",
                           "cash_div_tax": 0.1, "ex_date": f"{y}0620"})
    dv = pd.DataFrame(dv)
    dv.to_parquet(vd.DIRS["dividend"] / "20240420.parquet", index=False)
    vd.DIRS["daily_recent"].mkdir(parents=True, exist_ok=True)
    for d in dates[-25:]:
        pd.DataFrame({"ts_code": ind_df["ts_code"], "trade_date": d.strftime("%Y%m%d"),
                      "amount": rng.lognormal(10, 1, len(ind_df))}
                     ).to_parquet(vd.DIRS["daily_recent"] / f"{d:%Y%m%d}.parquet", index=False)


def use_tmp_dirs(tmp_path, monkeypatch):
    root = tmp_path / "value_screen"
    monkeypatch.setattr(vd, "ROOT", root)
    monkeypatch.setattr(vd, "DIRS", {k: root / v.name for k, v in vd.DIRS.items()})
    monkeypatch.setattr(vd, "IND_FILE", root / "industry" / "stock_industry.parquet")


# ───────────────────────── 计算层测试 ─────────────────────────
def test_latest_pct_matches_pandas_rolling_rank():
    rng = np.random.default_rng(1)
    w = pd.DataFrame(np.round(rng.lognormal(1, 0.5, (400, 40)), 1))          # 有并列
    w = w.mask(rng.random(w.shape) < 0.05)                                   # 有缺失
    w.iloc[:, 0] = np.nan                                                    # 整列缺失
    w.iloc[-1, 1] = np.nan                                                   # 最后一天缺失
    w.iloc[:150, 2] = np.nan                                                 # 有效观测不足
    for win, minp in ((250, 100), (1250, 252), (60, 30)):
        ref = w.rolling(win, min_periods=minp).rank(pct=True).iloc[-1]
        got = vc.latest_pct(w, win=win, minp=minp)
        pd.testing.assert_series_equal(got.astype(float), ref.astype(float), check_names=False)


def test_fscore_yoy_alignment_equals_shift4_when_quarters_contiguous(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    db, ind_df, fund, dates = make_universe(n=80, n_days=300)
    write_universe(db, ind_df, fund, dates)
    new = vc.fscore_table(TODAY)
    ref = ref_fscore(fund, TODAY)
    pd.testing.assert_frame_equal(new[["fscore"] + vc.CRIT].sort_index(), ref.sort_index(), check_dtype=False)


def test_fscore_with_missing_quarter_is_not_misaligned(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    db, ind_df, fund, dates = make_universe(n=40, n_days=300, gap_codes=("600000",))
    write_universe(db, ind_df, fund, dates)
    new = vc.fscore_table(TODAY)
    # 600000 缺 2025Q2：最新期 2026Q3 的上年同期是 2025Q3，应按"报告期−1年"取到，而不是按行号错位
    row = fund[(fund.ts_code == "600000.SH") & (fund.end_date.isin(["20260930", "20250930"]))].sort_values("end_date")
    cur, prev = row.iloc[1], row.iloc[0]
    assert new.loc["600000", "ΔROA>0"] == int(cur.roa - prev.roa > 0)
    assert new.loc["600000", "Δ周转>0"] == int(cur.assets_turn - prev.assets_turn > 0)


def test_run_screen_matches_original_pipeline(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    db, ind_df, fund, dates = make_universe()
    write_universe(db, ind_df, fund, dates)
    res = vc.run_screen(vc.Params(), today=TODAY)

    # —— 原脚本流程 ——
    d = db.copy()
    d["date"] = pd.to_datetime(d["trade_date"])
    d["code"] = d["ts_code"].str[:6]
    value_raw = ref_build_value(d[["date", "code", "pe_ttm", "pb"]])
    lncap = np.log(d.set_index(["date", "code"])["circ_mv"].clip(lower=1))
    cur = value_raw.index.get_level_values("date").max()
    ind = ind_df.set_index("code")["industry"]
    st = set(ind_df["code"][ind_df["name"].str.contains("ST")])
    neu = ref_neutralize(value_raw, ind, lncap, [cur])
    value_neu = neu.groupby(level="date").rank(pct=True).xs(cur, level="date").dropna()
    cm = d[d.date == cur].set_index("code")["circ_mv"]
    domain = [c for c in cm[cm >= cm.quantile(0.30)].index if c not in st and c in value_neu.index]
    vd_ = value_neu.reindex(domain).dropna()
    cheap = vd_[vd_ >= vd_.quantile(0.9)].index
    f = ref_fscore(fund, TODAY)
    ref_cand = {c for c in cheap if c in f.index and f.loc[c, "fscore"] >= 5}

    assert res.counts["域"] == len(domain)
    assert res.counts["最便宜"] == len(cheap)
    assert set(res.candidates["代码"]) == ref_cand and len(ref_cand) > 0
    got_rank = res.candidates.set_index("代码")["value便宜排名"]
    ref_rank = vd_.rank(pct=True).reindex(got_rank.index).round(2)
    pd.testing.assert_series_equal(got_rank, ref_rank, check_names=False)
    got_f = res.candidates.set_index("代码")["F分"]
    pd.testing.assert_series_equal(got_f, f.loc[got_rank.index, "fscore"], check_names=False, check_dtype=False)
    # 信息列都有值、形状合理
    c = res.candidates
    assert c["现金流正年数"].between(0, 3).all() and c["分红年数"].between(0, 3).all()
    assert c["成交额亿20d"].notna().all()
    # 合成数据里公告日 = 期末 + 20~55 天，2026-11-10 时 Q3 只有一部分已公告 → 取"今天已公告的最新一期"
    assert set(c["财报期"]) <= {"2026-06-30", "2026-09-30"}


def test_save_csv_and_empty_result(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    db, ind_df, fund, dates = make_universe(n=120, n_days=300)
    write_universe(db, ind_df, fund, dates)
    res = vc.run_screen(vc.Params(fscore_min=8), today=TODAY)     # F 分最高 7 → 空
    assert res.candidates.empty and vc.save_csv(res) is None
    res = vc.run_screen(vc.Params(fscore_min=3, cheap_pct=0.3), today=TODAY)
    p = vc.save_csv(res)
    assert p is not None and p.exists() and len(pd.read_csv(p)) == len(res.candidates)


# ───────────────────────── 数据层测试（假的 Tushare 接口） ─────────────────────────
class FakePro:
    """tables: api -> DataFrame；按 trade_date / ann_date / period 过滤，支持 offset/limit。"""
    def __init__(self, tables, deny=(), rate_fail_n=0, rate_msg="抱歉，您每分钟最多访问该接口50次，权限的具体详情访问：https://tushare.pro"):
        self.t, self.deny, self.rate_fail_n, self.rate_msg, self.calls = tables, set(deny), rate_fail_n, rate_msg, []

    def query(self, api, offset=0, limit=6000, **kw):
        self.calls.append((api, dict(kw), offset))
        if api in self.deny:
            raise Exception("抱歉，您没有接口访问权限，权限的具体详情访问：https://tushare.pro")
        if self.rate_fail_n > 0:
            self.rate_fail_n -= 1
            raise Exception(self.rate_msg)
        df = self.t[api]
        for k in ("trade_date", "ann_date", "period", "ts_code", "list_status"):
            if k in kw and k in df.columns:
                df = df[df[k] == kw[k]]
        if api == "trade_cal":
            df = df[(df.cal_date >= kw["start_date"]) & (df.cal_date <= kw["end_date"])]
        df = df.iloc[offset:offset + limit]
        f = kw.get("fields")
        return df[[c for c in f.split(",") if c in df.columns]] if f else df


def _install_fake(monkeypatch, pro):
    import tushare_src as ts_src
    monkeypatch.setattr(ts_src, "_pro", pro)
    monkeypatch.setattr(vd.time, "sleep", lambda s: None)
    vd._THR.reset(100000)


def test_query_paginates_and_rate_limit_is_retried_not_misread_as_permission(monkeypatch):
    big = pd.DataFrame({"a": range(23)})
    pro = FakePro({"x": big}, rate_fail_n=2)
    _install_fake(monkeypatch, pro)
    monkeypatch.setattr(vd, "PAGE_LIMIT", 10)
    out = vd.query("x")
    assert len(out) == 23 and out["a"].tolist() == list(range(23))
    assert vd._THR.interval >= 60.0 / 100000 * 4          # 撞过两次频控 → 已降速


def test_query_permission_denied_is_detected(monkeypatch):
    _install_fake(monkeypatch, FakePro({"x": pd.DataFrame({"a": [1]})}, deny=["x"]))
    try:
        vd.query("x")
        raise AssertionError("should raise")
    except vd.PermissionDenied:
        pass


def test_update_daily_basic_resume_empty_marker_and_recent_guard(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    today = dt.date.today()
    days = [(today - dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(12, -1, -1)]
    cal = pd.DataFrame({"cal_date": days, "is_open": 1})
    rows = pd.DataFrame({"ts_code": ["000001.SZ"] * len(days[:-1]), "trade_date": days[:-1],     # 最后一天还没入库
                         "pe_ttm": 5.0, "pb": 1.0, "circ_mv": 1e6})
    rows = rows[rows.trade_date != days[3]]                                                       # 中间某天本来就没数据
    pro = FakePro({"trade_cal": cal, "daily_basic": rows})
    _install_fake(monkeypatch, pro)
    log = vd.update_daily_basic()
    d = vd.DIRS["daily_basic"]
    assert (d / f"{days[3]}.empty").exists()                      # 空日子打了标记
    assert not (d / f"{days[-1]}.empty").exists()                 # 最近几天的空不打标记，留待重试
    n_calls = len([c for c in pro.calls if c[0] == "daily_basic"])
    vd.update_daily_basic()                                       # 第二次：只会重试最近未入库的那一天
    assert len([c for c in pro.calls if c[0] == "daily_basic"]) - n_calls <= vd.RECENT_GUARD
    assert "daily_basic" in log[0]


def test_update_fundamentals_vip_by_period_and_dedup_earliest_ann(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    today = dt.date.today()
    per = vd._quarter_ends(today.year - 3, today)
    rows = []
    for p in per:
        for code in ("000001.SZ", "600000.SH", "830799.BJ"):
            rows.append({"ts_code": code, "ann_date": f"{p[:4]}{int(p[4:6]) + 1:02d}15" if p[4:6] != "12" else f"{int(p[:4]) + 1}0301",
                         "end_date": p, "period": p, **{c: 1.0 for c in vd.FUND_NUM}})
    tbl = pd.DataFrame(rows)
    pro = FakePro({"fina_indicator_vip": tbl})
    _install_fake(monkeypatch, pro)
    vd.update_fundamentals()
    got = pd.read_parquet(vd.DIRS["fundamentals"] / f"{per[0]}.parquet")
    assert set(got.ts_code) == {"000001.SZ", "600000.SH"}                      # 北交所被剔除
    # 同一 (ts_code,end_date) 之后又来一个更晚的公告日（修正版）→ 仍保留最早
    late = tbl[tbl.end_date == per[-1]].copy()
    late["ann_date"] = "20991231"
    pro.t["fina_indicator_vip"] = late
    vd.update_fundamentals()
    g2 = pd.read_parquet(vd.DIRS["fundamentals"] / f"{per[-1]}.parquet")
    assert (g2.ann_date != "20991231").all()
    # 旧报告期不会被重拉
    n_before = len(pro.calls)
    vd.update_fundamentals()
    periods_called = {c[1].get("period") for c in pro.calls[n_before:] if c[0] == "fina_indicator_vip"}
    assert periods_called == set(per[-vd.REFETCH_PERIODS:])


def test_update_fundamentals_falls_back_to_per_stock_without_vip(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    today = dt.date.today()
    per = vd._quarter_ends(today.year - 3, today)[:3]
    rows = [{"ts_code": c, "ann_date": p[:4] + "0830", "end_date": p, **{k: 1.0 for k in vd.FUND_NUM}}
            for p in per for c in ("000001.SZ", "600000.SH")]
    sb = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"], "list_status": "L"})
    pro = FakePro({"fina_indicator_vip": pd.DataFrame(rows), "fina_indicator": pd.DataFrame(rows), "stock_basic": sb},
                  deny=["fina_indicator_vip"])
    _install_fake(monkeypatch, pro)
    log = vd.update_fundamentals()
    assert any("逐只" in x for x in log)
    assert {p.stem for p in vd.DIRS["fundamentals"].glob("*.parquet")} == set(per)


def test_fiscal_years_roll_over_with_date():
    assert vd.fiscal_years("2026-11-10") == (2023, 2024, 2025)
    assert vd.fiscal_years("2027-04-29") == (2023, 2024, 2025)      # 年报还没披露完
    assert vd.fiscal_years("2027-04-30") == (2024, 2025, 2026)
    assert vd.fiscal_years("2027-01-05") == (2023, 2024, 2025)


def test_status_and_missing_required(tmp_path, monkeypatch):
    use_tmp_dirs(tmp_path, monkeypatch)
    assert set(vd.missing_required()) == {"daily_basic", "fundamentals", "industry"}
    st = vd.data_status(TODAY)
    assert (st["状态"].str.contains("未下载")).sum() >= 3
    db, ind_df, fund, dates = make_universe(n=30, n_days=300)
    write_universe(db, ind_df, fund, dates)
    assert vd.missing_required() == []
    st = vd.data_status(TODAY)
    assert st.set_index("数据集").loc["daily_basic", "状态"] == "✅"
    assert st.set_index("数据集").loc["财务指标", "状态"] == "✅"
