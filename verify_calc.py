"""独立复算：不经过网页，直接读你的日线文件，把均线/涨跌的每一步中间值打印出来，方便和行情软件核对。
用法：  python verify_calc.py            （默认 沪深300，最新交易日）
        python verify_calc.py 000905 2026-11-06
"""
import sys

import pandas as pd

import config
import loaders

code = sys.argv[1] if len(sys.argv) > 1 else config.INDEXES[config.BENCH]["code"]
df = loaders.load_price(code)
if df is None or "close" not in df.columns:
    sys.exit(f"{code}: 没有可用日线。{loaders.diagnose([code]).to_string(index=False)}")
if len(sys.argv) > 2:
    df = df[df.index <= pd.Timestamp(sys.argv[2])]
c = df["close"].dropna()
print(f"{code} 共 {len(c)} 根日线，{c.index[0].date()} ~ {c.index[-1].date()}")
print("\n最近 6 个交易日收盘（请和行情软件逐个核对！）：")
print(c.tail(6).to_string())
last = c.iloc[-1]
ma = {n: c.tail(n).mean() for n in (5, 20, 60)}      # 手工算：最近 n 个收盘的算术平均
print("\n手工均线 =", {f"MA{k}": round(v, 2) for k, v in ma.items()})
print(f"最新收盘 {last:.2f}")
bull = last > ma[20] > ma[60] and ma[5] > ma[20]
bear = last < ma[20] < ma[60] and ma[5] < ma[20]
print(f"多头排列条件 收盘>MA20>MA60 且 MA5>MA20 → {bull}")
print(f"空头排列条件 收盘<MA20<MA60 且 MA5<MA20 → {bear}")
print("结论:", "多头排列" if bull else "空头排列" if bear else ("震荡偏强" if last > ma[60] else "震荡偏弱"))
print(f"\n当日涨跌 {last / c.iloc[-2] - 1:+.2%}")
