"""生成【模拟数据】到 data_demo/，用来在接真实数据前预览网页。全部是随机数，不代表任何真实行情。
用法：python make_demo_data.py   然后   set ASHARE_DEMO=1 && streamlit run app.py   (PowerShell: $env:ASHARE_DEMO=1)
"""
import os
os.environ["ASHARE_DEMO"] = "1"

import numpy as np
import pandas as pd

import analytics as an
import config
import fetchers

rng = np.random.default_rng(7)
END = pd.Timestamp("2026-11-06")
DATES = pd.bdate_range(end=END, periods=260)
MEMBERS = ["中信期货", "国泰君安", "海通期货", "华泰期货", "东证期货", "银河期货", "中信建投", "广发期货",
           "国信期货", "中泰期货", "光大期货", "招商期货", "永安期货", "申银万国", "方正中期", "南华期货",
           "中金财富", "瑞达期货", "兴证期货", "五矿期货"]


def walk(n, start, vol, common=None, beta=0.5):
    r = rng.normal(0.0002, vol, n)
    if common is not None:
        r = r + beta * common
    return start * np.cumprod(1 + r)


def write_px(code, close, amount_base, sub=None):
    amt = amount_base * (1 + rng.normal(0, 0.2, len(close))).clip(0.3)
    d = config.DAILY_DIR / sub if sub else config.DAILY_DIR
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": DATES, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                  "volume": amt / close, "amount": amt}).to_csv(d / f"{code}.csv", index=False)


def main():
    n = len(DATES)
    mkt = rng.normal(0, 0.008, n)
    idx_close = {}
    for nm, info in config.INDEXES.items():
        c = walk(n, {"000300": 4600, "000016": 3000, "000905": 7200, "000852": 7400}[info["code"]], 0.004, mkt, 0.8)
        idx_close[info["fut"]] = pd.Series(c, index=DATES)
        write_px(info["code"], c, 3e11, sub="index")     # 指数代码与个股同号，必须放 index 子目录
    for codes in config.ETF_GROUPS.values():
        for c in codes:
            write_px(c, walk(n, 4.0, 0.004, mkt, 0.8), 8e8)
    for sec in config.SECTORS.values():
        for code, _ in sec["stocks"]:
            write_px(code, walk(n, 10.0, 0.012, mkt * 0.4, 0.5), 5e8)
    for code in config.REF_ETFS.values():
        write_px(code, walk(n, 1.2, 0.004, -mkt * 0.2, 0.5), 3e8)

    # 虚拟成分股（代码用 9xxxxx，不会和真实代码冲突）
    cons_rows, margin_rows = [], []
    k = 0
    for ii, (nm, info) in enumerate(config.INDEXES.items()):
        for j in range(40):
            code = f"9{ii}{j:04d}"
            c = walk(n, 20.0, 0.015, mkt, 0.9)
            write_px(code, c, 4e8)
            cons_rows.append({"index_code": info["code"], "code": code, "name": f"模拟{nm}{j}"})
            bal = 3e8 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
            buy = bal * rng.uniform(0.02, 0.06, n)
            margin_rows.append(pd.DataFrame({"date": DATES, "code": code, "rz_balance": bal, "rz_buy": buy,
                                             "rz_repay": buy * 0.97}))
    fetchers.append_ds("index_cons", pd.DataFrame(cons_rows))
    fetchers.append_ds("margin", pd.concat(margin_rows, ignore_index=True))

    # ETF 份额
    sh_rows = []
    for codes in config.ETF_GROUPS.values():
        for c in codes:
            sh = 3e9 * np.cumprod(1 + rng.normal(0, 0.01, n))
            sh_rows.append(pd.DataFrame({"date": DATES, "code": c, "name": f"模拟ETF{c}", "shares": sh}))
    fetchers.append_ds("etf_shares", pd.concat(sh_rows, ignore_index=True))

    # 期指：日行情 + 前20会员排名
    daily_rows, rank_rows = [], []
    state = {}
    for dt in DATES:
        for nm, info in config.INDEXES.items():
            v = info["fut"]
            S = idx_close[v].loc[dt]
            for con in an.active_contracts(v, dt):
                days = max((an.contract_expiry(con) - dt.date()).days, 1)
                F = S * (1 - 0.04 * days / 365 + rng.normal(0, 0.001))
                daily_rows.append({"date": dt, "contract": con, "variety": v, "close": F, "settle": F,
                                   "volume": rng.integers(5e4, 2e5), "oi": rng.integers(8e4, 2.5e5)})
                for r, m in enumerate(MEMBERS, 1):
                    key = (con, m)
                    lo, so = state.get(key, (rng.integers(500, 12000), rng.integers(500, 12000)))
                    nlo = max(int(lo * (1 + rng.normal(0, 0.03))), 100)
                    nso = max(int(so * (1 + rng.normal(0, 0.03))), 100)
                    state[key] = (nlo, nso)
                    rank_rows.append({"date": dt, "contract": con, "variety": v, "rank": r,
                                      "long_member": m, "long_oi": nlo, "long_chg": nlo - lo,
                                      "short_member": m, "short_oi": nso, "short_chg": nso - so,
                                      "vol_member": m, "vol": 1000, "vol_chg": 0})
    fetchers.append_ds("cffex_daily", pd.DataFrame(daily_rows))
    fetchers.append_ds("cffex_rank", pd.DataFrame(rank_rows))
    print("演示数据已生成：", config.DATA_ROOT)


if __name__ == "__main__":
    main()
