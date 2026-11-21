# -*- coding: utf-8 -*-
"""修复「💎 价值质量选股」页点击「🔄 更新全部数据（增量）」导致 Streamlit 服务崩溃退出的问题。

改造内容：
  1. value_screen_data.py：
     - 所有 parquet 读写改走 _read_pq / _to_pq（支持 VS_PARQUET_ENGINE 切换 pyarrow / fastparquet）；
     - update_all() 每一步之间主动 gc.collect() 释放内存；
     - 增加子进程入口与 run_update_in_subprocess()：实时回传进度条，遇到底层库崩溃(0xC000001D等)自动切换
       ARROW_USER_SIMD_LEVEL=NONE 与 fastparquet 重试，且即使极端异常也只影响子进程，不会拖垮 Streamlit。
  2. views/value_screen.py：
     - 「更新全部数据」和 6 个单独更新按钮改走 vd.run_update_in_subprocess(...)。

用法（在项目根目录运行）：
    python fix_value_screen_update_crash.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_HELPERS = r'''
NATIVE_CRASH = {3221225477, 3221225501, 3221226505, -1073741819, -1073741795, -1073740791,
                -11, -4, -6, 139, 132, 134}


def _read_pq(path, columns=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))


def _to_pq(df: pd.DataFrame, path: Path):
    df.to_parquet(path, index=False, engine=os.environ.get("VS_PARQUET_ENGINE", "auto"))
'''

DATA_SUBPROC = r'''

def _env_attempts() -> list:
    import importlib.util
    seq = [("默认设置", {}),
           ("关闭 Arrow SIMD 加速", {"ARROW_USER_SIMD_LEVEL": "NONE"})]
    if importlib.util.find_spec("fastparquet") is not None:
        seq.append(("关闭 SIMD + fastparquet 引擎",
                    {"ARROW_USER_SIMD_LEVEL": "NONE", "VS_PARQUET_ENGINE": "fastparquet"}))
    return seq


_GOOD_ENV: Optional[dict] = None


def run_update_in_subprocess(step: str = "all", progress: Progress = None, timeout: int = 3600) -> list:
    """在独立子进程里执行数据下载/parquet写入，避免 pyarrow 崩溃或内存波动拖垮 Streamlit 服务。"""
    global _GOOD_ENV
    import pickle
    import subprocess
    import sys
    import tempfile

    here = Path(__file__).resolve()
    attempts = [("上次成功的设置", _GOOD_ENV)] if _GOOD_ENV is not None else _env_attempts()
    tried, last_err = [], ""
    for label, extra in attempts:
        with tempfile.TemporaryDirectory() as td:
            out_pkl = Path(td) / "update_log.pkl"
            cmd = [sys.executable, "-u", str(here), "--step", str(step), "--out", str(out_pkl)]
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            env.update(extra)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(here.parent),
                env=env,
            )
            tail = []
            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    line = raw.rstrip("\r\n")
                    if line.startswith("PROGRESS:"):
                        parts = line.split(":", 2)
                        if len(parts) == 3 and progress:
                            try:
                                progress(float(parts[1]), parts[2])
                            except Exception:  # noqa
                                pass
                    elif line:
                        tail.append(line)
                        if len(tail) > 40:
                            tail.pop(0)
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return [f"【更新超时】超过 {timeout} 秒未完成（已下载的分片已保存，再点一次会继续）"]

            if rc == 0 and out_pkl.exists():
                with open(out_pkl, "rb") as f:
                    logs = pickle.load(f)
                _GOOD_ENV = dict(extra)
                if tried:
                    logs.insert(0, f"⚠ 前 {len(tried)} 次因底层库崩溃失败（{'、'.join(tried)}），已改用：{label}（分片已自动续传）")
                return logs

            tried.append(label)
            last_err = f"子进程退出码 {rc}\n" + "\n".join(tail[-20:])
            if rc not in NATIVE_CRASH:
                break
    _GOOD_ENV = None
    return [f"【更新中断】已尝试：{' → '.join(tried)}，已下载的分片保留，再点一次会续传。\n{last_err}"]


def _cli(argv=None) -> int:
    import argparse
    import pickle
    import traceback

    ap = argparse.ArgumentParser(description="value_screen_data 子进程更新入口")
    ap.add_argument("--step", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    def cb(frac, text):
        print(f"PROGRESS:{min(max(float(frac), 0.0), 1.0):.4f}:{text}", flush=True)

    try:
        if a.step == "all":
            logs = update_all(cb)
        else:
            _, fn = STEPS[int(a.step)]
            logs = fn(cb)
    except Exception:  # noqa
        traceback.print_exc()
        return 1
    with open(a.out, "wb") as f:
        pickle.dump(logs, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
'''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    cnt = src.count(old)
    if cnt != 1:
        raise RuntimeError(f"{label}：期望找到 1 处锚点，实际 {cnt} 处。")
    return src.replace(old, new)


def patch_data(src: str):
    if "def run_update_in_subprocess(" in src:
        return None
    src = replace_once(
        src,
        "import datetime as dt\nimport os\nimport time",
        "import datetime as dt\nimport gc\nimport os\nimport time",
        "value_screen_data.py imports",
    )
    src = replace_once(
        src,
        "Progress = Optional[Callable[[float, str], None]]",
        "Progress = Optional[Callable[[float, str], None]]\n" + DATA_HELPERS,
        "value_screen_data.py helpers",
    )
    src = replace_once(
        src,
        '            df.to_parquet(out_dir / f"{d}.parquet", index=False)',
        '            _to_pq(df, out_dir / f"{d}.parquet")',
        "_pull_by_date to_parquet",
    )
    src = replace_once(
        src,
        '    allf[["code", "ts_code", "name", "industry", "market", "list_date"]].to_parquet(IND_FILE, index=False)',
        '    _to_pq(allf[["code", "ts_code", "name", "industry", "market", "list_date"]], IND_FILE)',
        "update_industry to_parquet",
    )
    src = replace_once(
        src,
        '        if p.exists():\n'
        '            g = pd.concat([pd.read_parquet(p), g], ignore_index=True)\n'
        '        g = (g.sort_values(["ts_code", "ann_date"])\n'
        '              .drop_duplicates(["ts_code", "end_date"], keep="first"))\n'
        '        g[[c for c in FUND_KEEP if c in g.columns]].to_parquet(p, index=False)',
        '        if p.exists():\n'
        '            g = pd.concat([_read_pq(p), g], ignore_index=True)\n'
        '        g = (g.sort_values(["ts_code", "ann_date"])\n'
        '              .drop_duplicates(["ts_code", "end_date"], keep="first"))\n'
        '        _to_pq(g[[c for c in FUND_KEEP if c in g.columns]], p)',
        "_save_fund read/write parquet",
    )
    src = replace_once(
        src,
        '        if len(df):\n            df.to_parquet(f, index=False)',
        '        if len(df):\n            _to_pq(df, f)',
        "update_cashflow to_parquet",
    )
    src = replace_once(
        src,
        'def update_all(progress: Progress = None) -> list:\n'
        '    """全部增量更新。某一步失败不影响其它步（分片已在盘上，重跑会续上）。"""\n'
        '    log = []\n'
        '    for label, fn in STEPS:\n'
        '        try:\n'
        '            log += fn(progress)\n'
        '        except Exception as e:  # noqa\n'
        '            log.append(f"【{label}】中止：{e!r}（已下载的分片保留，重跑会续上）")\n'
        '    return log',
        'def update_all(progress: Progress = None) -> list:\n'
        '    """全部增量更新。某一步失败不影响其它步（分片已在盘上，重跑会续上）。"""\n'
        '    log = []\n'
        '    for i, (label, fn) in enumerate(STEPS):\n'
        '        sub = (lambda frac, text, i=i, label=label: progress((i + frac) / len(STEPS), f"{label} · {text}")) if progress else None\n'
        '        try:\n'
        '            log += fn(sub)\n'
        '        except Exception as e:  # noqa\n'
        '            log.append(f"【{label}】中止：{e!r}（已下载的分片保留，重跑会续上）")\n'
        '        gc.collect()\n'
        '    return log',
        "update_all",
    )
    return src.rstrip() + "\n" + DATA_SUBPROC


def patch_view(src: str):
    if "run_update_in_subprocess" in src:
        return None
    return replace_once(
        src,
        'def _run_update(label, fn):\n'
        '    bar = st.progress(0.0, text=f"{label}：准备中…")\n\n'
        '    def cb(frac, text):\n'
        '        bar.progress(min(max(float(frac), 0.0), 1.0), text=text)\n\n'
        '    try:\n'
        '        logs = fn(cb)\n'
        '    except Exception as e:  # noqa\n'
        '        logs = [f"【{label}】失败：{e!r}"]\n'
        '    bar.empty()\n'
        '    st.session_state["vs_update_log"] = logs\n'
        '    st.rerun()\n\n\n'
        'if st.button("🔄 更新全部数据（增量）", type="primary", disabled=not has_token, key="vs_update_all"):\n'
        '    _run_update("全部数据", vd.update_all)\n\n'
        'with st.expander("单独更新某类数据"):\n'
        '    cols = st.columns(3)\n'
        '    for i, (label, fn) in enumerate(vd.STEPS):\n'
        '        if cols[i % 3].button(label, key=f"vs_upd_{i}", disabled=not has_token):\n'
        '            _run_update(label, fn)',
        'def _run_update(label, step_or_fn):\n'
        '    bar = st.progress(0.0, text=f"{label}：准备中…")\n\n'
        '    def cb(frac, text):\n'
        '        bar.progress(min(max(float(frac), 0.0), 1.0), text=text)\n\n'
        '    try:\n'
        '        if isinstance(step_or_fn, str):\n'
        '            logs = vd.run_update_in_subprocess(step_or_fn, cb)\n'
        '        else:\n'
        '            logs = step_or_fn(cb)\n'
        '    except Exception as e:  # noqa\n'
        '        logs = [f"【{label}】失败：{e!r}"]\n'
        '    bar.empty()\n'
        '    st.session_state["vs_update_log"] = logs\n'
        '    st.rerun()\n\n\n'
        'if st.button("🔄 更新全部数据（增量）", type="primary", disabled=not has_token, key="vs_update_all"):\n'
        '    _run_update("全部数据", "all")\n\n'
        'with st.expander("单独更新某类数据"):\n'
        '    cols = st.columns(3)\n'
        '    for i, (label, _fn) in enumerate(vd.STEPS):\n'
        '        if cols[i % 3].button(label, key=f"vs_upd_{i}", disabled=not has_token):\n'
        '            _run_update(label, str(i))',
        "views/value_screen.py _run_update",
    )


def main():
    p_data = ROOT / "value_screen_data.py"
    p_view = ROOT / "views" / "value_screen.py"
    for p in (p_data, p_view):
        if not p.exists():
            sys.exit(f"[中止] 找不到 {p}，请把脚本放在项目根目录后运行。")

    planned = []
    for p, fn in ((p_data, patch_data), (p_view, patch_view)):
        old = p.read_text(encoding="utf-8-sig")
        try:
            new = fn(old)
        except Exception as e:
            sys.exit(f"[中止] {p.name} 失败：{e}\n所有文件均未修改。")
        if new is not None:
            compile(new, str(p), "exec")
            planned.append((p, new))
        else:
            print(f"[跳过] {p.relative_to(ROOT)} 已经包含子进程更新保护")

    for p, new in planned:
        bak = p.with_suffix(p.suffix + ".bak_before_subproc")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(new, encoding="utf-8")
        print(f"[更新] {p.relative_to(ROOT)}（备份：{bak.name}）")

    print("\n完成！请重启 Streamlit：streamlit run app.py")
    print("现在在「💎 价值质量选股」页点击「🔄 更新全部数据（增量）」或各分步按钮，都会在独立子进程中执行：")
    print("  · 遇到 pyarrow 底层崩溃会自动切换关闭 SIMD / fastparquet 重试，并从断点分片续传；")
    print("  · Streamlit 主服务不会再意外退出。")


if __name__ == "__main__":
    main()