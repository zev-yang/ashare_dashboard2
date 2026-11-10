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
            "include_sector": True, "include_cons": False, "include_local": True, "retry_minutes": 30}


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
    """返回 (是否更新, 原因)。条件：开关开启 + 今天是交易日 + 已过截止时间 + 核心数据落后 + 距上次尝试已过重试间隔。"""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    s = load_settings()
    if force:
        return True, "手动强制"
    if not s["auto_update"]:
        return False, "自动更新已关闭"
    if not freshness.is_trading_day(now):
        return False, "今天不是交易日"
    h, m = freshness.parse_cutoff(s["cutoff"])
    if (now.hour, now.minute) < (h, m):
        return False, f"未到 {s['cutoff']}（收盘后数据才齐）"
    expected = freshness.expected_latest(now, s["cutoff"])
    if not freshness.core_stale(freshness.collect(now, expected)):
        return False, "数据已是最新"
    last_try = load_state().get("last_attempt")
    if last_try and (now - pd.Timestamp(last_try)).total_seconds() < int(s["retry_minutes"]) * 60:
        return False, f"距上次尝试不足 {s['retry_minutes']} 分钟"
    return True, f"数据落后于 {expected.date()}"


def run(progress=None, now=None) -> list:
    """执行一次完整更新并记录状态/日志。"""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    s = load_settings()
    save_state({"last_attempt": now.isoformat()})
    log = fetchers.update_all(int(s["backfill"]), s["include_sector"], s["include_cons"], s["include_local"], progress)
    expected = freshness.expected_latest(now, s["cutoff"])
    ok = not freshness.core_stale(freshness.collect(now, expected))
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
