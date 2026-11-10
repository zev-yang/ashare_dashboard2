"""逐个测试外资代理的每个数据来源（不重试、不写文件），看哪个能通、哪个报什么错。
用法（PowerShell）：  python diagnose_proxies.py
把输出整段贴给我即可定位。"""
import time

import fetchers

import akshare as ak

print("akshare 版本:", fetchers.ak_version())
for sym, desc, cands in fetchers.proxy_candidates():
    print(f"\n=== {sym} {desc} ===")
    for src, fn, save_as in cands:
        t0 = time.time()
        try:
            raw = fn(ak)
            df = fetchers.normalize_price(raw)
            print(f"  ✅ {src}: {len(df)} 行，{df['date'].min().date()} ~ {df['date'].max().date()}，原始列={list(raw.columns)}")
        except Exception as e:  # noqa
            print(f"  ❌ {src}: {type(e).__name__}: {str(e)[:200]}  ({time.time() - t0:.1f}s)")
        time.sleep(1.0)
