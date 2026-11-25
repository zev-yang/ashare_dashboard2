# -*- coding: utf-8 -*-
"""修复 tushare_src.py 中 update_local_stock_daily 因个别新股/缺文件拉低 min(todo) 而重下2019全市场的bug。

特点：
  1. 新股按 max(PX_START, list_date) 起补，不再对 2026 新股要 2019 数据；
  2. 自动按总 API 调用次数最小化，将 todo 拆分为「按日批量(近期大多数股票)」+「逐只兜底(少数长期缺日离群股)」；
  3. 仅对按日批量的全市场数据调用 record_avg_from_daily()，不污染全A平均股价缓存；
  4. 不改变文件格式、表头或复权口径，不影响「① 避险板块(2000/2005起)」与「💎 价值质量选股」。

用法（在项目根目录运行）：
    python fix_tushare_daily_incremental.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "tushare_src.py"

OLD_HEAD = '''    basic = call("stock_basic", exchange="", list_status="L", fields="ts_code,symbol,name")
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
            frames, day_frames = [], []
            for d in days[i:i + chunk_days]:
                df = call("daily", trade_date=_ymd(d))
                if len(df):
                    frames.append(df)
                    day_frames.append((_ymd(d), df))
                if progress:
                    progress(min((days.index(d) + 1) / len(days), 1.0), f"本地日线 {d.date()}")
            if day_frames:
                record_avg_from_daily(day_frames)
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
                progress((i + 1) / len(todo), f"本地日线 {code}")'''

NEW_HEAD = '''    import bisect

    basic = call("stock_basic", exchange="", list_status="L", fields="ts_code,symbol,name,list_date")
    names = dict(zip(basic["symbol"], basic["name"]))
    todo, fresh = {}, 0
    index_codes = {v["code"] for v in config.INDEXES.values()}
    skipped_idx = []
    for code, list_d in zip(basic["symbol"], basic["list_date"]):
        if code in index_codes:                     # 000016/000905/000852 既是指数又是个股：绝不往可能是指数的文件里追加
            skipped_idx.append(code)
            continue
        p = _local_path(code)
        ld = _last_date(p) if p.exists() else None
        if ld is None:
            ld_est = _parse_ymd(list_d)
            s_dt = max(history_start, ld_est) if ld_est else history_start
            if s_dt <= latest:
                todo[code] = s_dt
            else:
                fresh += 1
        elif ld < latest:
            s_dt = (dt.datetime.strptime(ld, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")
            if s_dt <= latest:
                todo[code] = s_dt
            else:
                fresh += 1
        else:
            fresh += 1
    log = [f"【⑥ 本地个股日线】目录 {config.DAILY_DIR}；最新入库日 {latest}；已是最新 {fresh} 只，需补 {len(todo)} 只"]
    if skipped_idx:
        log.append(f"  已跳过与指数同号的个股 {skipped_idx}（其本地文件可能是指数数据，不追加）")
    if not todo:
        return log

    all_cal = trade_dates(min(todo.values()), latest)
    cal_ymds = [_ymd(d) for d in all_cal]
    first_idx = {code: bisect.bisect_left(cal_ymds, s) for code, s in todo.items()}
    # 过滤掉起点之后已经没有交易日的股票（如周末日期后无交易日）
    todo = {code: s for code, s in todo.items() if first_idx[code] < len(all_cal)}
    if not todo:
        return log

    # 寻找使 (按日批量天数 + 逐只离群股数) 最小的切分点
    sorted_idx = sorted(first_idx[c] for c in todo)
    best_split, best_cost = None, len(todo)
    for idx in range(len(all_cal)):
        n_days = len(all_cal) - idx
        n_indiv = bisect.bisect_left(sorted_idx, idx)
        if n_days + n_indiv < best_cost:
            best_cost = n_days + n_indiv
            best_split = idx

    if best_split is None:
        bulk_days = []
        bulk_todo = {}
        indiv_todo = dict(todo)
        log.append(f"  方式：逐只（{len(indiv_todo)} 次调用，比按日批量更省）")
    else:
        bulk_days = all_cal[best_split:]
        bulk_todo = {c: s for c, s in todo.items() if first_idx[c] >= best_split}
        indiv_todo = {c: s for c, s in todo.items() if first_idx[c] < best_split}
        log.append(f"  方式：按日批量 {len(bulk_days)} 天（{len(bulk_todo)} 只）+ 逐只补齐离群股 {len(indiv_todo)} 只（共约 {best_cost} 次调用）")

    written, bad, discont = 0, 0, []
    total_ops = max(len(bulk_days) + len(indiv_todo), 1)
    done_ops = 0

    for i in range(0, len(bulk_days), chunk_days):
        frames, day_frames = [], []
        for d in bulk_days[i:i + chunk_days]:
            df = call("daily", trade_date=_ymd(d))
            if len(df):
                frames.append(df)
                day_frames.append((_ymd(d), df))
            done_ops += 1
            if progress:
                progress(min(done_ops / total_ops, 1.0), f"本地日线 {d.date()}")
        if day_frames:
            record_avg_from_daily(day_frames)
        if not frames:
            continue
        big = _to_std(pd.concat(frames, ignore_index=True), names)
        for code, g in big.groupby("code"):
            if code not in bulk_todo:
                continue
            g = g[g["date"].str.replace("-", "") >= bulk_todo[code]]
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

    for code, start in indiv_todo.items():
        try:
            df = _windowed("daily", start, latest, ts_code=ts_code(code, "stock"))
        except Exception as e:  # noqa
            log.append(f"  {code}: {e}")
            done_ops += 1
            if progress:
                progress(min(done_ops / total_ops, 1.0), f"离群个股 {code}")
            continue
        if len(df):
            g = _to_std(df, names)
            path = _local_path(code)
            if path.exists() and not _continuous(path, g):
                discont.append(code)
            elif _append_rows(path, g):
                written += 1
            else:
                bad += 1
        done_ops += 1
        if progress:
            progress(min(done_ops / total_ops, 1.0), f"离群个股 {code}")'''


def main():
    if not TARGET.exists():
        sys.exit(f"[中止] 找不到 {TARGET}，请把脚本放在项目根目录后运行。")

    src = TARGET.read_text(encoding="utf-8-sig")
    if "best_split, best_cost" in src:
        print("[跳过] tushare_src.py 已经包含最小调用切分优化。")
        return

    cnt = src.count(OLD_HEAD)
    if cnt != 1:
        sys.exit(f"[中止] tushare_src.py 中期望找到 1 处匹配代码，实际找到 {cnt} 处。未修改文件。")

    new_src = src.replace(OLD_HEAD, NEW_HEAD, 1)
    compile(new_src, str(TARGET), "exec")

    bak = TARGET.with_suffix(".py.bak_before_split_fix")
    if not bak.exists():
        shutil.copy2(TARGET, bak)
    TARGET.write_text(new_src, encoding="utf-8")
    print(f"[更新] {TARGET.name}（备份：{bak.name}）")
    print("完成！请重启 Streamlit（或下次运行 auto_update.py）即可生效。")


if __name__ == "__main__":
    main()