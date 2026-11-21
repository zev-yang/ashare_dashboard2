# -*- coding: utf-8 -*-
"""给「价值质量选股」(value_screen_core.py) 新增两列：
  1. 最新价：来自本地日线目录 (E:\\ashare_data\\daily) 最新一个交易日的收盘价；
  2. 历史最低价(前复权)：取本地日线全历史最低价，并用 Tushare adj_factor 前复权至最新日口径。
其它筛选逻辑、页面布局与已有列均保持不变。

用法（在项目根目录运行）：
    python update_value_screen_prices.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "value_screen_core.py"

HELPER_CODE = r'''

# ───────────────────────── 日线最新价 & 全历史前复权最低价 ─────────────────────────
def _load_stock_daily_local(code: str) -> Optional[pd.DataFrame]:
    """从本地日线目录读取个股日线（排除指数子目录，避免 000016/000905/000852 同号混淆）。"""
    import config
    import loaders

    all_paths = loaders._file_index().get(code, [])
    paths = [p for p in all_paths if not loaders._looks_like_index_file(p) and config.CACHE_DIR not in p.parents]
    if not paths:
        paths = [p for p in all_paths if not loaders._looks_like_index_file(p)]
    for p in paths:
        try:
            df = loaders._read_any(p)
            if "close" in df.columns and "low" in df.columns:
                df = df.copy()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["low"] = pd.to_numeric(df["low"], errors="coerce")
                df = df[(df["close"] > 0) & (df["low"] > 0)].sort_index()
                if len(df):
                    return df
        except Exception:  # noqa
            continue
    return None


def _get_adj_factor(code: str, first_dt: pd.Timestamp, last_dt: pd.Timestamp) -> Optional[pd.Series]:
    """读取/增量更新单只股票的 Tushare 复权因子（缓存为 parquet）。"""
    import tushare_src as ts_src

    cache_dir = vd.ROOT / "adj_factor"
    cache_file = cache_dir / f"{code}.parquet"
    df_af = None
    if cache_file.exists():
        try:
            df_af = _read_pq(cache_file)
        except Exception:  # noqa
            df_af = None

    need_fetch = (
        df_af is None
        or df_af.empty
        or pd.to_datetime(df_af["trade_date"].astype(str)).max() < last_dt
        or pd.to_datetime(df_af["trade_date"].astype(str)).min() > first_dt + pd.Timedelta(days=30)
    )
    if need_fetch and ts_src.available() and not os.environ.get("PYTEST_CURRENT_TEST"):
        tscode = ts_src.ts_code(code, "stock")
        # 增量：若已有早期数据且仅尾部落后，只补新段；否则拉全段（约3800行，1次调用）
        if df_af is not None and not df_af.empty and pd.to_datetime(df_af["trade_date"].astype(str)).min() <= first_dt + pd.Timedelta(days=30):
            s_dt = (pd.to_datetime(df_af["trade_date"].astype(str)).max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            s_dt = first_dt.strftime("%Y%m%d")
        e_dt = last_dt.strftime("%Y%m%d")
        if s_dt <= e_dt:
            fresh = vd.query("adj_factor", ts_code=tscode, start_date=s_dt, end_date=e_dt,
                             fields="ts_code,trade_date,adj_factor")
            if fresh is not None and len(fresh):
                parts = [df_af, fresh[["trade_date", "adj_factor"]]] if (df_af is not None and len(df_af)) else [fresh[["trade_date", "adj_factor"]]]
                df_af = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date", keep="last")
                df_af = df_af.sort_values("trade_date")
                cache_dir.mkdir(parents=True, exist_ok=True)
                df_af.to_parquet(cache_file, index=False)

    if df_af is None or df_af.empty:
        return None
    s = pd.Series(
        pd.to_numeric(df_af["adj_factor"], errors="coerce").to_numpy(float),
        index=pd.to_datetime(df_af["trade_date"].astype(str)),
    ).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def _cand_price_stats(cand: list, log: list) -> dict:
    """只对最终候选股读取本地日线最新价与全历史前复权最低价。返回 {code: (latest_close, qfq_low)}。"""
    out = {}
    n_px = n_adj = 0
    adj_err = None
    for c in cand:
        df = _load_stock_daily_local(c)
        if df is None or df.empty:
            continue
        n_px += 1
        cur_close = float(df["close"].iloc[-1])
        qfq_low = np.nan
        try:
            af = _get_adj_factor(c, df.index[0], df.index[-1])
            if af is not None and len(af):
                aligned = af.reindex(af.index.union(df.index)).sort_index().ffill().bfill().reindex(df.index)
                last_af = float(aligned.iloc[-1])
                if last_af > 0 and np.isfinite(last_af):
                    qfq_low = float((df["low"] * (aligned / last_af)).min())
                    n_adj += 1
        except Exception as e:  # noqa
            if adj_err is None:
                adj_err = str(e)
        out[c] = (cur_close, qfq_low)
    log.append(f"   日线现价/前复权最低价：匹配本地日线 {n_px}/{len(cand)} 只，前复权成功 {n_adj}/{len(cand)} 只")
    if adj_err:
        log.append(f"   ⚠ 获取复权因子部分失败（{adj_err[:120]}），相应股票历史最低价留空")
    return out
'''

OLD_MARK = '    _mark("组装结果表")\n    rows = []\n    for c in cand:\n        f = fdet.loc[c]\n        a = float(amt20.get(c, np.nan))\n        rows.append({\n            "代码": c, "名称": str(name_map.get(c, c)), "行业": str(ind.get(c, "")),'

NEW_MARK = '    _mark("读取候选股本地日线最新价与前复权最低价")\n    px_stats = _cand_price_stats(cand, log)\n\n    _mark("组装结果表")\n    rows = []\n    for c in cand:\n        f = fdet.loc[c]\n        a = float(amt20.get(c, np.nan))\n        cur_px, qfq_low = px_stats.get(c, (np.nan, np.nan))\n        rows.append({\n            "代码": c, "名称": str(name_map.get(c, c)), "行业": str(ind.get(c, "")),\n            "最新价": round(cur_px, 2) if not np.isnan(cur_px) else np.nan,\n            "历史最低价(前复权)": round(qfq_low, 2) if not np.isnan(qfq_low) else np.nan,'


def main():
    if not TARGET.exists():
        sys.exit(f"[中止] 找不到 {TARGET}，请将脚本放在项目根目录运行。")

    src = TARGET.read_text(encoding="utf-8-sig")
    if "_cand_price_stats" in src:
        print("[跳过] value_screen_core.py 已经包含最新价与前复权历史最低价列。")
        return

    anchor_main = "# ───────────────────────── 主流程 ─────────────────────────"
    if src.count(anchor_main) != 1:
        sys.exit("[中止] 未找到主流程锚点，文件未修改。")
    if src.count(OLD_MARK) != 1:
        sys.exit("[中止] 未找到组装结果表锚点，文件未修改。")

    new_src = src.replace(anchor_main, HELPER_CODE.lstrip("\n") + "\n\n" + anchor_main, 1)
    new_src = new_src.replace(OLD_MARK, NEW_MARK, 1)
    compile(new_src, str(TARGET), "exec")

    bak = TARGET.with_suffix(".py.bak_before_prices")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
    TARGET.write_text(new_src, encoding="utf-8")
    print(f"[更新] {TARGET.name}（备份：{bak.name}）")
    print("完成！请重启 Streamlit 后进入「💎 价值质量选股」重新点击「▶ 运行选股」即可看到新增两列：")
    print("  · 最新价（本地日线最新收盘价）")
    print("  · 历史最低价(前复权)（本地日线全历史最低价按 Tushare 复权因子前复权至最新日口径）")


if __name__ == "__main__":
    main()