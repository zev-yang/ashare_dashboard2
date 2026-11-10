"""命令行自动更新：交易日收盘后补齐最新数据；非交易日/数据已最新时自动跳过。
用法：
    python auto_update.py                # 按设置判断是否需要更新
    python auto_update.py --force        # 不判断，直接更新
    python auto_update.py --print-schtasks   # 打印 Windows 计划任务命令（每个工作日 18:00 运行）
Windows 计划任务读不到你终端里的环境变量，所以 Tushare token 请用网页「数据」页「保存到本地文件」。"""
import sys

import autoupdate

if "--print-schtasks" in sys.argv:
    print(autoupdate.schtasks_command())
    sys.exit(0)

run_it, why = autoupdate.should_run(force="--force" in sys.argv)
print("更新" if run_it else "跳过", "-", why)
if run_it:
    log = autoupdate.run()
    print("\n".join(log))
