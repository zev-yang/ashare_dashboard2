# -*- coding: utf-8 -*-
"""给 value_screen_core.py 打补丁：pyarrow 读 parquet 时"非法指令"崩溃（退出码 3221225501 / 0xC000001D）。

用法（在 value_screen_core.py 所在文件夹里）：
    python fix_pyarrow_crash.py

做的事（先备份成 value_screen_core.py.bak，任何一步对不上就中止、不改文件）：
  1. 所有读 parquet 的地方改走 _read_pq()，引擎可用环境变量 VS_PARQUET_ENGINE 切换（pyarrow / fastparquet）。
  2. run_in_subprocess 遇到"底层库崩溃"时自动重试：默认 → 关闭 Arrow SIMD 加速 → 换 fastparquet（装了才试）。
     成功过一次的设置会记住，之后直接用。
  3. 崩溃退出码表补上 0xC000001D（非法指令）等，页面提示更准确。
"""
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("value_screen_core.py")

HELPERS = r'''
# 原生库崩溃的退出码：Windows 0xC0000005 访问冲突 / 0xC000001D 非法指令 / 0xC0000409；POSIX 的 SIGSEGV / SIGILL / SIGABRT
NATIVE_CRASH = {3221225477, 3221225501, 3221226505, -1073741819, -1073741795, -1073740791,
                -11, -4, -6, 139, 132, 134}


def _read_pq(path, columns=None) -> pd.DataFrame:
    """读 parquet。引擎可用环境变量 VS_PARQUET_ENGINE 指定（pyarrow / fastparquet；默认 auto = pyarrow）。"""
    return pd.read_parquet(path, columns=columns, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))
'''

HINT_NEW = r'''if rc in NATIVE_CRASH:
            return ("底层库崩溃（访问冲突 / 非法指令 / 段错误），通常是 pyarrow 与本机 CPU 或安装环境不兼容。"
                    "程序已自动尝试过备用读法（见下方说明）；仍失败的话请看下面的调用栈。")'''

RUN_NEW = r'''def _env_attempts() -> list:
    """pyarrow 读 parquet 在个别机器上会因 CPU 指令集（SIMD）问题"非法指令"崩溃。
    依次尝试：默认 → 关闭 Arrow 的 SIMD 加速 → 再换 fastparquet 引擎（装了才试）。"""
    import importlib.util
    seq = [("默认设置", {}),
           ("关闭 Arrow SIMD 加速 (ARROW_USER_SIMD_LEVEL=NONE)", {"ARROW_USER_SIMD_LEVEL": "NONE"})]
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎",
                    {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    return seq


_GOOD_ENV: Optional[dict] = None    # 上次成功的环境变量组合：之后直接用它，不必每次重试


def run_in_subprocess(p: Optional[Params] = None, timeout: int = 900, cmd_prefix=None) -> Result:
    """在独立 Python 进程里跑选股（读数据、算分位、回归都在子进程）。
    好处：内存吃紧或底层库崩溃时只会让子进程退出，网页服务不会掉线，页面还能显示失败原因。
    子进程若因底层库崩溃（NATIVE_CRASH）退出，会换一组环境变量自动重试（见 _env_attempts）。"""
    global _GOOD_ENV
    p = p or Params()
    here = Path(__file__).resolve()
    tpath = trace_path()
    tpath.parent.mkdir(parents=True, exist_ok=True)
    attempts = [("上次成功的设置", _GOOD_ENV)] if _GOOD_ENV is not None else _env_attempts()
    tried, err = [], None
    for label, extra in attempts:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "result.pkl")
            cmd = list(cmd_prefix or [sys.executable, str(here)]) + [
                "--out", out, "--trace", str(tpath), "--excl", str(p.excl_small_pct), "--cheap", str(p.cheap_pct),
                "--fmin", str(int(p.fscore_min)), "--amt-min", str(p.amt_min_yi)]
            if p.exclude_bj:
                cmd.append("--exclude-bj")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            env.update(extra)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=timeout, cwd=str(here.parent), env=env)
            except subprocess.TimeoutExpired as e:
                raise ScreenFailed("timeout", str(e.stderr or "")[-3000:], _read_trace(tpath)) from e
            if proc.returncode == 2:
                raise DataMissing((proc.stderr or "").replace("DATA_MISSING:", "").strip() or "数据不足")
            if proc.returncode == 0 and os.path.exists(out):
                with open(out, "rb") as f:
                    d = pickle.load(f)
                _GOOD_ENV = dict(extra)
                res = Result(candidates=d["candidates"], asof=d["asof"], log=d["log"], counts=d["counts"],
                             csv_path=Path(d["csv_path"]) if d.get("csv_path") else None)
                if tried:
                    res.log.insert(0, f"⚠ 前 {len(tried)} 次因底层库崩溃失败（{'、'.join(tried)}），已改用：{label}")
                return res
            err = ScreenFailed(proc.returncode, (proc.stderr or "")[-3000:], _read_trace(tpath))
            tried.append(label)
            if proc.returncode not in NATIVE_CRASH:
                break    # 普通脚本报错：换环境变量没用，直接报
    _GOOD_ENV = None
    if len(tried) > 1:
        err.stderr = "已依次尝试：" + " → ".join(tried) + "，都崩溃了。\n" + err.stderr
    raise err
'''


def die(msg):
    sys.exit(f"[中止] {msg}\n文件没有被修改。")


def replace_once(src, old, new, label):
    if src.count(old) != 1:
        die(f"{label}：期望在文件里找到 1 处，实际 {src.count(old)} 处。")
    return src.replace(old, new)


def main():
    if not TARGET.exists():
        die(f"当前文件夹里没有 {TARGET}。请先 cd 到 value_screen_core.py 所在的文件夹。")
    src = TARGET.read_text(encoding="utf-8")
    if "NATIVE_CRASH" in src:
        print("已经打过补丁了，无需重复。")
        return

    # 1) 辅助函数：插在 CRIT 定义之后
    m = re.search(r"^CRIT = .*\n", src, re.M)
    if not m:
        die("找不到 CRIT = ... 这一行。")
    src = src[:m.end()] + HELPERS + src[m.end():]

    # 2) 所有 read_parquet 改走 _read_pq
    src = replace_once(src, "parts = [pd.read_parquet(f, columns=columns) if columns else pd.read_parquet(f) for f in fs]",
                       "parts = [_read_pq(f, columns) for f in fs]", "_read_dir 里的 read_parquet")
    src = replace_once(src, "df = pd.read_parquet(f, columns=cols)",
                       "df = _read_pq(f, cols)", "load_daily_basic 里的 read_parquet")
    src = replace_once(src, "pd.read_parquet(vd.IND_FILE)",
                       "_read_pq(vd.IND_FILE)", "load_industry 里的 read_parquet")

    # 3) 崩溃提示
    pat = re.compile(r'if rc in \(3221225477, -1073741819, -11, 139\):\n\s+return "[^"\n]*"')
    if len(pat.findall(src)) != 1:
        die("ScreenFailed.hint() 里的崩溃判断没找到（文件可能已被改动过）。")
    src = pat.sub(lambda _m: HINT_NEW, src)

    # 4) 整个 run_in_subprocess 换成带自动重试的版本（到下一个 "# ───" 分节注释为止）
    start = src.find("def run_in_subprocess(")
    if start < 0:
        die("找不到 def run_in_subprocess(。")
    nxt = re.compile(r"^# ─", re.M).search(src, start)
    if not nxt:
        die("找不到 run_in_subprocess 之后的分节注释行（# ───…）。")
    src = src[:start] + RUN_NEW + "\n\n" + src[nxt.start():]

    shutil.copy2(TARGET, TARGET.with_suffix(".py.bak"))
    TARGET.write_text(src, encoding="utf-8")
    compile(src, str(TARGET), "exec")      # 语法自检
    print("完成。原文件已备份为 value_screen_core.py.bak。重新点「▶ 运行选股」即可。")


if __name__ == "__main__":
    main()
