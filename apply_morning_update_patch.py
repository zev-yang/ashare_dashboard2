# -*- coding: utf-8 -*-
"""给 ashare_dashboard 打补丁：早间补数据 + 数据"非最新"提示 + 融资/ETF 重抓。

用法（在 app.py 所在文件夹里）：
    python apply_morning_update_patch.py

先在内存里把所有改动算好，任何一处对不上就整体中止、一个文件都不改；
全部通过才写盘，并把每个被改的文件备份成 <文件名>.bak_before_morning。

改动内容：
  fetchers.py        update_range 增加 refresh_last：已入库的最近 3 个交易日也重抓并覆盖（修复"早上抓到不完整的当日数据，
                     之后再点下载却因为日期已存在被跳过，列永远不重算"）；新增 update_morning（只补融资融券 + ETF份额）。
  freshness.py       collect(strict=...)；融资/ETF 晚一天时状态改为「⏳ 非最新」，不再写成"✅ 最新"。
  common.py          顶部数据状态条：有数据不是最新时显示黄色提示并列出是哪几类、到哪一天。
  views/dashboard.py 总览表下方：列出哪些列用的是旧数据；融资买入占比为"—"时写明原因。
  autoupdate.py      should_run 增加早间窗口（默认 07:00–12:00，仍用严格口径检查融资/ETF）；run 在早间只补融资+ETF；
                     新增 schtasks_morning_command（07:00 起每 30 分钟重试到 12:00 的计划任务）。
  auto_update.py     --print-schtasks 同时打印早间任务；--force 仍然是完整更新。
  views/data.py      数据页显示两条计划任务命令。
  README.md          补一条说明（找不到对应位置就跳过）。
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(".")
edits = {}          # 路径 -> 新内容
original = {}       # 路径 -> 原内容


def die(msg):
    sys.exit(f"[中止] {msg}\n没有任何文件被修改。")


def get(path):
    p = ROOT / path
    if path in edits:
        return edits[path]
    if not p.exists():
        die(f"找不到 {path}。请先 cd 到 app.py 所在的文件夹。")
    original[path] = p.read_text(encoding="utf-8")
    return original[path]


def put(path, text):
    edits[path] = text


def rep_once(path, old, new, label):
    t = get(path)
    n = t.count(old)
    if n != 1:
        die(f"{path}：{label} —— 期望找到 1 处，实际 {n} 处（文件可能已被改动过）。")
    put(path, t.replace(old, new))


def slice_replace(path, start_marker, end_marker, new, label, must_contain=()):
    t = get(path)
    a = t.find(start_marker)
    if a < 0:
        die(f"{path}：{label} —— 找不到起点 {start_marker!r}。")
    b = t.find(end_marker, a + len(start_marker))
    if b < 0:
        die(f"{path}：{label} —— 找不到终点 {end_marker!r}。")
    old = t[a:b]
    for s in must_contain:
        if s not in old:
            die(f"{path}：{label} —— 原函数里没有 {s!r}，和预期版本不一致。")
    put(path, t[:a] + new + t[b:])


# ═══════════════════════════ fetchers.py ═══════════════════════════
UPDATE_RANGE_NEW = '''REFRESH_LAST = 3     # 增量更新时，已入库的最近 N 个交易日也重抓覆盖（交易所/数据源可能先放出一部分，之后才补全）


def update_range(name: str, start, end, progress=None, refresh_last: int = 0) -> list:
    """按交易日逐日抓取并入库。已有的日期默认跳过；refresh_last>0 时，已入库的最近 N 个交易日会重新抓取并覆盖
    （避免"早上抓到的是不完整的当日数据，之后再点下载却因日期已存在被跳过"）。重抓不到（尚未发布）不算错误，旧数据保留。
    开头连续 3 天都因同一原因失败时提前停止，不再刷屏。"""
    cal = trading_days(start, end)
    have = set(read_ds(name)["date"].dt.date)
    have_days = [d for d in cal if d.date() in have]
    refresh = {d.date() for d in have_days[-refresh_last:]} if refresh_last else set()
    days = [d for d in cal if d.date() not in have or d.date() in refresh]
    log, ok, fail, refreshed, first_errs, stopped = [], 0, 0, 0, [], False
    for i, d in enumerate(days):
        again = d.date() in refresh
        try:
            n = append_ds(name, FETCHERS[name](d))
            if again:
                refreshed += 1 if n else 0
            else:
                ok += 1 if n else 0
        except Exception as e:  # noqa
            if again:
                pass        # 重抓不到：保留已有数据，不计失败
            else:
                fail += 1
                if fail <= 3:
                    log.append(f"  {d.date()}: {explain(e)}")
                if ok == 0:
                    first_errs.append(str(e))
                    if len(first_errs) >= 3 and len(set(first_errs[-3:])) == 1:
                        stopped = True
                        break
        if progress:
            progress((i + 1) / max(len(days), 1), f"{name} {d.date()}")
    head = f"{name}: 新增 {ok} 个交易日" + (f"，重抓最近 {refreshed} 天" if refresh_last else "") + f"，失败/尚未发布 {fail} 天"
    if stopped:
        head += f"；连续失败原因相同，已停止（共 {len(days)} 天待抓）。可修好后重试，或在侧边栏手动导入官网文件"
    log.insert(0, head)
    return log


'''

UPDATE_MORNING_NEW = '''# ───────────── 早间补数据（只补次日早晨才披露的两类）─────────────
def update_morning(backfill_days=60, progress=None) -> list:
    """融资融券、ETF份额要到次日早晨才披露：早间只补这两类，不重跑日线 / 期指 / 外资代理。"""
    steps = [("① 融资融券", lambda p: download_margin(backfill_days, p)),
             ("ETF份额", lambda p: incremental_update("etf_shares", backfill_days, p))]
    log = []
    for i, (name, fn) in enumerate(steps):
        sub = (lambda frac, text, i=i, name=name: progress((i + frac) / len(steps), f"{name} · {text}")) if progress else None
        try:
            log += fn(sub)
        except Exception as e:  # noqa
            log.append(f"{name}: 整步失败 {explain(e)}")
    return log


'''

slice_replace("fetchers.py", "def update_range(", "def incremental_update(", UPDATE_RANGE_NEW,
              "update_range", must_contain=("have = set(read_ds(name)", "FETCHERS[name](d)"))
rep_once("fetchers.py", "return update_range(name, start, end, progress)",
         "return update_range(name, start, end, progress, refresh_last=REFRESH_LAST)", "_incremental_by_day 调用 update_range")
rep_once("fetchers.py", "# ───────────── 颗粒度对齐检查 ─────────────",
         UPDATE_MORNING_NEW + "# ───────────── 颗粒度对齐检查 ─────────────", "插入 update_morning")

# ═══════════════════════════ freshness.py ═══════════════════════════
rep_once("freshness.py", "def collect(now=None, expected=None) -> pd.DataFrame:",
         "def collect(now=None, expected=None, strict=False) -> pd.DataFrame:\n"
         '    """strict=True（早间补数据用）：融资融券 / ETF份额也必须到 expected，不再允许晚一天。"""',
         "collect 签名")
rep_once("freshness.py", "ALLOWED_LAG.get(name, 0)))", "0 if strict else ALLOWED_LAG.get(name, 0)))", "collect 里的 ALLOWED_LAG")
rep_once("freshness.py", 'st_ = f"✅ 最新（披露延迟 {lag} 日，正常）"',
         'st_ = f"⏳ 非最新：晚 {lag} 个交易日（交易所次日早晨披露）"', "_row 状态文字")

# ═══════════════════════════ common.py ═══════════════════════════
rep_once("common.py", 'bad = df[df["核心"] & df["状态"].str.startswith(("⚠️", "❌"))]',
         'bad = df[df["核心"] & df["状态"].str.startswith(("⚠️", "❌"))]\n'
         '    _lag = pd.to_numeric(df["落后(交易日)"], errors="coerce").fillna(0)\n'
         '    soft = df[(_lag > 0) & ~df.index.isin(bad.index)]      # 晚于应有日期、但属于"次日披露"的正常延迟',
         "render_freshness 计算 soft")
rep_once("common.py", '    else:\n        head = f"数据状态：✅ 日线最新到',
         '    elif len(soft):\n'
         '        items = "；".join(f"{str(r[\'数据\']).split(\'(\')[0]}（只到 {pd.Timestamp(r[\'最新日期\']).date()}）"\n'
         '                          for _, r in soft.iterrows())\n'
         '        head = (f"数据状态：⏳ 日线最新到 **{pd.Timestamp(last).date()}**，但以下数据**不是最新**（应有 "\n'
         '                f"**{pd.Timestamp(exp).date() if exp is not None else \'—\'}**）：{items}。"\n'
         '                "融资融券 / ETF份额通常次日早晨才披露，早间自动更新（默认 07:00 起每 30 分钟重试）会补上；"\n'
         '                "在此之前相关列用的是旧数据。")\n'
         '        box = st.warning\n'
         '    else:\n        head = f"数据状态：✅ 日线最新到',
         "render_freshness 增加 soft 分支")

# ═══════════════════════════ views/dashboard.py ═══════════════════════════
DASH_BLOCK = '''# ── 数据时效 / 缺失原因：哪些列用的是旧数据、为什么是"—" ──
    _asof = pd.Timestamp(asof)
    _stale = []
    for _lab, _df_, _cols in (("融资融券", margin, "融资买入占比、融资占比分位"), ("ETF份额", shares, "ETF近5日净申购"),
                              ("期指持仓排名", rank, "前20空单5日增量、空单性质"), ("期指日行情", fdaily, "年化基差")):
        if len(_df_) and pd.Timestamp(_df_["date"].max()) < _asof:
            _stale.append(f"**{_lab}** 只到 {pd.Timestamp(_df_['date'].max()).date()}（统计截止日 {_asof.date()}）→ 影响列：{_cols}")
    if _stale:
        st.warning("⏳ 以下数据**不是最新**，对应列用的是更早的数据（融资融券 / ETF份额通常次日早晨才披露，之后的更新会补上）：\\n\\n- "
                   + "\\n- ".join(_stale))
    _why = []
    for _nm, _info in IDX.items():
        _m = mg_sum.get(_nm) or {}
        if "ratio" in _m:
            continue
        if not len(margin):
            _why.append("融资融券明细是空的（「数据」页点 ①）")
            break
        if not len(cons):
            _why.append("没有指数成分股名单（「数据」页点「指数成分股名单」）")
            break
        _cc = cons.loc[cons["index_code"] == _info["code"], "code"].tolist()
        if not _cc:
            _why.append(f"{_nm}：没有成分股名单（「数据」页点「指数成分股名单」）")
            continue
        if not margin["code"].isin(_cc).any():
            _why.append(f"{_nm}：融资融券明细里一只成分股都没匹配上（股票代码格式不一致？）")
            continue
        _amt = amount_panel(tuple(_cc), S)
        _na = _amt.shape[1] if len(_amt) else 0
        if _na == 0:
            _why.append(f"{_nm}：成分股 {len(_cc)} 只，本地一只都没有带成交额的日线，算不出占比（「数据」页点 ⑥ 补日线）")
        else:
            _why.append(f"{_nm}：成分股 {len(_cc)} 只里本地有成交额的 {_na} 只，但和融资数据没有重叠的交易日"
                        f"（成交额到 {pd.Timestamp(_amt.index.max()).date()}，融资到 {pd.Timestamp(margin['date'].max()).date()}）")
    if _why:
        st.info("融资买入占比 / 分位显示为「—」的原因：\\n\\n- " + "\\n- ".join(_why))
    '''
anchor = 'st.markdown(f"**避险防御篮子相对沪深300**'
rep_once("views/dashboard.py", anchor, DASH_BLOCK + anchor, "总览表后插入数据时效提示")

# ═══════════════════════════ autoupdate.py ═══════════════════════════
rep_once("autoupdate.py", '"include_sector": True, "include_cons": False, "include_local": True, "retry_minutes": 30}',
         '"include_sector": True, "include_cons": False, "include_local": True, "retry_minutes": 30,\n'
         '            "morning_start": "07:00", "morning_end": "12:00"}', "defaults 增加早间窗口")

SHOULD_RUN_NEW = '''def should_run(now=None, force=False):
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


'''
slice_replace("autoupdate.py", "def should_run(", "def run(", SHOULD_RUN_NEW, "should_run",
              must_contain=("未到", "数据已是最新", "核心数据落后"))
rep_once("autoupdate.py", "def run(progress=None, now=None) -> list:",
         "def run(progress=None, now=None, full=False) -> list:", "run 签名")
rep_once("autoupdate.py",
         '    log = fetchers.update_all(int(s["backfill"]), s["include_sector"], s["include_cons"], s["include_local"], progress)',
         '    morning = (not full) and (now.hour, now.minute) < freshness.parse_cutoff(s["cutoff"])\n'
         '    if morning:      # 早间只补次日早晨才披露的融资融券 + ETF份额\n'
         '        log = fetchers.update_morning(int(s["backfill"]), progress)\n'
         '    else:\n'
         '        log = fetchers.update_all(int(s["backfill"]), s["include_sector"], s["include_cons"], s["include_local"], progress)',
         "run 里的更新调用")
rep_once("autoupdate.py", "    ok = not freshness.core_stale(freshness.collect(now, expected))",
         "    ok = not freshness.core_stale(freshness.collect(now, expected, strict=morning))", "run 里的 ok 判断")

t = get("autoupdate.py")
if "def schtasks_morning_command" in t:
    die("autoupdate.py 里已经有 schtasks_morning_command，看起来已经打过补丁。")
put("autoupdate.py", t.rstrip("\n") + '''


def schtasks_morning_command(project_dir=None, python_exe=None, at="07:00", every_min=30, hours=5) -> str:
    """早间补数据任务：每个工作日 at 起，每 every_min 分钟运行一次、持续 hours 小时（默认 07:00 起到 12:00）。
    交易所的融资融券 / ETF份额通常在 08:30–09:00 左右才披露，所以 07:00 起要反复重试；
    auto_update.py 每次先判断，数据已齐就立刻退出，不会重复下载。"""
    cmd = schtasks_command(project_dir, python_exe, at).replace('"AStockAutoUpdate"', '"AStockMorningUpdate"', 1)
    return cmd.replace(" /F ", f" /RI {every_min} /DU {hours:02d}:00 /F ", 1)
''')

# ═══════════════════════════ auto_update.py ═══════════════════════════
rep_once("auto_update.py", "print(autoupdate.schtasks_command())",
         "print(autoupdate.schtasks_command())\n    print(autoupdate.schtasks_morning_command())", "打印两条计划任务命令")
rep_once("auto_update.py", "log = autoupdate.run()", 'log = autoupdate.run(full="--force" in sys.argv)', "--force 走完整更新")

# ═══════════════════════════ views/data.py ═══════════════════════════
rep_once("views/data.py", 'st.code(autoupdate.schtasks_command(), language="powershell")',
         'st.code(autoupdate.schtasks_command() + "\\n\\n" + autoupdate.schtasks_morning_command(), language="powershell")\n'
         '    st.caption("上面两条：①「AStockAutoUpdate」每个工作日 18:00 完整更新；②「AStockMorningUpdate」每个工作日 07:00 起每 30 分钟检查一次、到 12:00 为止，"\n'
         '               "只补融资融券 + ETF份额（交易所约 08:30–09:00 才披露，所以 07:00 先试、之后重试），数据齐了就自动跳过。"\n'
         '               "电脑需要在这些时间开机且未休眠；要让它能唤醒电脑，到「任务计划程序」里勾选「唤醒计算机以运行此任务」。")',
         "数据页显示两条计划任务")

# ═══════════════════════════ README.md（可选） ═══════════════════════════
readme_anchor = "  3. 所有更新（手动、自动、计划任务）都会记录到"
rt = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else None
if rt is not None and rt.count(readme_anchor) == 1:
    original["README.md"] = rt
    put("README.md", rt.replace(
        readme_anchor,
        "  2b. **早间补数据**：融资融券、ETF份额要到次日早晨（约 08:30–09:00）才披露，所以收盘当晚这两类通常还是前一天的，"
        "页面顶部会用黄色条明确标出「不是最新」。再用一个计划任务 `AStockMorningUpdate`（每个工作日 07:00 起每 30 分钟一次，到 12:00），"
        "只补这两类，数据齐了自动跳过；创建命令同样由 `python auto_update.py --print-schtasks` 打印（第二条）。\n" + readme_anchor))
else:
    print("（README.md 里没找到对应位置，跳过说明文字，不影响功能）")

# ═══════════════════════════ 全部通过 → 语法检查 → 写盘 ═══════════════════════════
for path, text in edits.items():
    if path.endswith(".py"):
        try:
            compile(text, path, "exec")
        except SyntaxError as e:
            die(f"{path} 打补丁后语法检查失败：{e}")

for path, text in edits.items():
    p = ROOT / path
    shutil.copy2(p, str(p) + ".bak_before_morning")
    p.write_text(text, encoding="utf-8")
    print("已修改", path)
print("\n完成。每个被改文件都有 .bak_before_morning 备份。请重启 streamlit，然后按提示创建早间计划任务。")
