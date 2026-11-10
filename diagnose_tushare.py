"""逐个测试程序用到的 Tushare 接口（需要已配置 token）。用法： python diagnose_tushare.py
把输出整段贴给我即可（不会输出 token）。"""
import pandas as pd

import tushare_src as tsrc

pd.set_option("display.width", 220, "display.max_colwidth", 120)
print("token 已配置:", tsrc.available())
if tsrc.available():
    print(tsrc.self_check().to_string(index=False))
