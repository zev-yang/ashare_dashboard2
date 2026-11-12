"""自动更新：设置、状态、是否该更新的判断、执行。网页和命令行（auto_update.py）共用。"""
import json
import time
from pathlib import Path

import pandas as pd

import config
import fetchers
import freshness


def _settings_path() -> Path:
    return config.CACHE_DIR / "settings.json"


def _state_path() -> Path:
    return config.CACHE_DIR / "auto_update_state.json"


def _log_path() -> Path:
    p = config.CACHE_DIR / "logs"
    p.mkdir(exist_ok=True)
    return p / "update.log"


def defaults() -> dict:
    return {"auto_update": bool(config.AUTO_UPDATE_DEFAULT), "cutoff": config.READY_CUTOFF, "backfill": 60,
            "include_sector": True, "include_cons": False, "include_local": True, "retry_minutes": 30,
            "morning_start": "07:00", "morning_end": "12:00"}


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa
        return {}


def load_settings() -> dict:
    return {**defaults(), **_read(_settings_path())}


def save_settings(d: dict):
    _settings_path().write_text(json.dumps({**load_settings(), **d}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state() -> dict:
    return _read(_state_path())


def save_state(d: dict):
    _state_path().write_text(json.dumps({**load_state(), **d}, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(title: str, lines: list):
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(f"\n===== {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {title} =====\n" + "\n".join(lines) + "\n")


def read_log_tail(n=60) -> str:
    p = _log_path()
    return "\n".join(p.read_text(encoding="utf-8").splitlines()[-n:]) if p.exists() else ""


def should_run(now=None, force=False):
    """返回 (是否更新, 原因)。
    收盘后（≥截止时间）：开关开启 + 今天是交易日 + 核心数据落后 + 距上次尝试已过重试间隔。
    早间补数据（截止时间之前，且在 morning_start～morning_end 之间，默认 07:00–12:00）：前一交易日的融资融券 / ETF份额
    要到次日早晨才披露，所以这段时间用"严格口径"检查（这两类也必须到前一交易日），落后就更新。"""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    s = load_settings()
    if force:
        return True, "手动强制"
    if not s["auto_update"]:
        return False, "自动更新已关闭"
    if not freshness.is_trading_day(now):
        return False, "今天不是交易日"
    hm = (now.hour, now.minute)
    morning = hm < freshness.parse_cutoff(s["cutoff"])
    if morning:
        if hm < freshness.parse_cutoff(s["morning_start"]):
            return False, f"未到早间补数据时间 {s['morning_start']}"
        if hm >= freshness.parse_cutoff(s["morning_end"]):
            return False, f"已过早间补数据时段（{s['morning_start']}–{s['morning_end']}），等 {s['cutoff']} 收盘后再更新"
    expected = freshness.expected_latest(now, s["cutoff"])
    if not freshness.core_stale(freshness.collect(now, expected, strict=morning)):
        return False, "数据已是最新" + ("（含融资融券 / ETF份额）" if morning else "")
    gap = min(int(s["retry_minutes"]), 25) if morning else int(s["retry_minutes"])   # 早间任务每 30 分钟跑一次，间隔要略小于它
    last_try = load_state().get("last_attempt")
    if last_try and (now - pd.Timestamp(last_try)).total_seconds() < gap * 60:
        return False, f"距上次尝试不足 {gap} 分钟"
    return True, ("早间补数据：" if morning else "") + f"数据落后于 {expected.date()}"


def run(progress=None, now=None, full=False) -> list:
    """执行一次完整更新并记录状态/日志。"""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    s = load_settings()
    save_state({"last_attempt": now.isoformat()})
    morning = (not full) and (now.hour, now.minute) < freshness.parse_cutoff(s["cutoff"])
    if morning:      # 早间只补次日早晨才披露的融资融券 + ETF份额
        log = fetchers.update_morning(int(s["backfill"]), progress)
    else:
        log = fetchers.update_all(int(s["backfill"]), s["include_sector"], s["include_cons"], s["include_local"], progress)
    expected = freshness.expected_latest(now, s["cutoff"])
    ok = not freshness.core_stale(freshness.collect(now, expected, strict=morning))
    save_state({"last_finish": pd.Timestamp.now().isoformat(), "last_ok": ok,
                "last_expected": str(expected.date()) if expected is not None else None,
                **({"last_success": pd.Timestamp.now().isoformat()} if ok else {})})
    append_log("自动更新" + ("（完成，数据已最新）" if ok else "（完成，但仍有数据落后，稍后重试）"), log)
    return log


def schtasks_command(project_dir=None, python_exe=None, at="18:00") -> str:
    import sys
    project_dir = Path(project_dir or config.BASE)
    python_exe = python_exe or sys.executable
    return (f'schtasks /Create /TN "AStockAutoUpdate" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST {at} /F '
            f'/TR "\\"{python_exe}\\" \\"{project_dir / "auto_update.py"}\\""')


def schtasks_morning_command(project_dir=None, python_exe=None, at="07:00", every_min=30, hours=5) -> str:
    """早间补数据任务：每个工作日 at 起，每 every_min 分钟运行一次、持续 hours 小时（默认 07:00 起到 12:00）。
    交易所的融资融券 / ETF份额通常在 08:30–09:00 左右才披露，所以 07:00 起要反复重试；
    auto_update.py 每次先判断，数据已齐就立刻退出，不会重复下载。"""
    cmd = schtasks_command(project_dir, python_exe, at).replace('"AStockAutoUpdate"', '"AStockMorningUpdate"', 1)
    return cmd.replace(" /F ", f" /RI {every_min} /DU {hours:02d}:00 /F ", 1)
