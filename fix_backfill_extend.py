# -*- coding: utf-8 -*-
"""修复 fetchers.py：当你在「数据」页调大「首次回补交易日数」（例如从 60 调成 250 或 500）时，
即使本地已有旧数据，也会自动向前补齐缺少的更早交易日（已有的日期跳过，不重复下载）。

用法（在项目根目录运行）：
    python fix_backfill_extend.py
"""
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "fetchers.py"

OLD_BLOCK = '''def incremental_update(name: str, backfill_days: int = 60, progress=None) -> list:
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
    return update_range(name, start, end, progress, refresh_last=REFRESH_LAST)'''

NEW_BLOCK = '''def _target_backfill_start(end, backfill_days: int):
    cal = trading_days(end - pd.Timedelta(days=int(backfill_days * 1.7) + 25), end)
    return cal[-backfill_days] if len(cal) > backfill_days else cal[0]


def incremental_update(name: str, backfill_days: int = 60, progress=None) -> list:
    """增量：若已有数据起点覆盖不够 backfill_days 个交易日，自动向前补齐缺口；已够则仅从最后10天增量补洞。"""
    end = pd.Timestamp.today().normalize()
    df = read_ds(name)
    target_start = _target_backfill_start(end, backfill_days)
    if name == "etf_shares" and "tushare" in _sources():          # 按代码取整段历史，十几次调用
        if len(df) and df["date"].min() <= target_start + pd.Timedelta(days=7):
            start0 = df["date"].max() - pd.Timedelta(days=15)
        else:
            start0 = target_start
        try:
            codes = sorted({c for v in config.ETF_GROUPS.values() for c in v})
            n = append_ds("etf_shares", tsrc.etf_shares_history(codes, start0))
            return [f"etf_shares: Tushare 按代码取 {len(codes)} 只 ETF（自 {pd.Timestamp(start0).date()}），写入 {n} 行"]
        except Exception as e:  # noqa
            if _sources() == ["tushare"]:
                return [f"etf_shares: Tushare 失败 {explain(e)}"]
            first = f"etf_shares: Tushare 失败（{explain(e)[:120]}），改用免费源"
            return [first] + _incremental_by_day(name, df, end, backfill_days, progress)
    return _incremental_by_day(name, df, end, backfill_days, progress)


def _incremental_by_day(name, df, end, backfill_days, progress):
    target_start = _target_backfill_start(end, backfill_days)
    if len(df) and df["date"].min() <= target_start + pd.Timedelta(days=7):
        cal = trading_days(df["date"].max() - pd.Timedelta(days=30), end)
        start = cal[-11] if len(cal) > 10 else cal[0]
    else:
        start = target_start
    return update_range(name, start, end, progress, refresh_last=REFRESH_LAST)'''


def main():
    if not TARGET.exists():
        sys.exit(f"[中止] 找不到 {TARGET}，请将脚本放在项目根目录后运行。")
    src = TARGET.read_text(encoding="utf-8-sig")
    if "def _target_backfill_start(" in src:
        print("[跳过] fetchers.py 已包含历史回补扩展修复")
        return
    cnt = src.count(OLD_BLOCK)
    if cnt != 1:
        sys.exit(f"[中止] fetchers.py 中期望找到 1 处目标代码块，实际找到 {cnt} 处。未修改文件。")

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    compile(new_src, str(TARGET), "exec")

    bak = TARGET.with_suffix(".py.bak_before_backfill_extend")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
    TARGET.write_text(new_src, encoding="utf-8")
    print(f"[更新] {TARGET.name}（备份：{bak.name}）")
    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("现在到「🗄️ 数据」页将「首次回补交易日数」设为 250（或 500）后：")
    print("  · 点「🔄 更新全部数据」（或单独点 ① 融资融券 / ② 期指 / ④ ETF份额），会自动补齐更早缺失的历史；")
    print("  · 已有日期自动跳过，且每天按日落盘（注：250 日约需 ~10-13 分钟，500 日约 ~25-30 分钟，补齐后日常增量恢复几秒钟）。")


if __name__ == "__main__":
    main()