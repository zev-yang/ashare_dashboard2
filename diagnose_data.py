"""检查本地日线目录能不能被程序识别。用法（PowerShell，和启动网页同一个窗口）：
    $env:ASHARE_DAILY_DIR="D:\\your\\daily"
    python diagnose_data.py
"""
import pandas as pd

import config
import loaders

pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_columns", 20)
print("日线目录:", config.DAILY_DIR, "| 存在:", config.DAILY_DIR.exists())
files = [p for p in config.DAILY_DIR.rglob("*") if p.suffix.lower() in (".csv", ".parquet")] if config.DAILY_DIR.exists() else []
print("csv/parquet 文件数:", len(files), "| 前10个:", [p.name for p in files[:10]])
codes = [v["code"] for v in config.INDEXES.values()]
print("\n=== 四大指数 ===")
print(loaders.diagnose(codes).to_string(index=False))
etf = sorted({c for v in config.ETF_GROUPS.values() for c in v})
sec = sorted({c for s in config.SECTORS.values() for c, _ in s["stocks"]})
miss = [c for c in etf + sec if not loaders.has_local(c)]
print(f"\nETF/板块个股 {len(etf) + len(sec)} 个，目录里缺文件名含代码的有 {len(miss)} 个：", miss)
