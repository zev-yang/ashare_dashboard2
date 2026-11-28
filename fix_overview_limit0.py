# -*- coding: utf-8 -*-
"""修复 analytics.py 中 overview_index_history 的 ffill(limit=0) 报错。"""
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "analytics.py"

OLD_BLOCK = '''    if etf_df is not None and not etf_df.empty:
        e_sig = etf_df["sig"].shift(1) if lag_t1 else etf_df["sig"]
        e_ext = etf_df["extreme"].shift(1) if lag_t1 else etf_df["extreme"]
        f2 = e_sig.reindex(idx).ffill(limit=0 if lag_t1 else 1)
        ext2 = e_ext.reindex(idx).ffill(limit=0 if lag_t1 else 1).fillna(False)
    else:
        f2, ext2 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)

    if mar_df is not None and not mar_df.empty:
        m_sig = mar_df["sig"].shift(1) if lag_t1 else mar_df["sig"]
        m_ovh = mar_df["overheated"].shift(1) if lag_t1 else mar_df["overheated"]
        f4 = m_sig.reindex(idx).ffill(limit=0 if lag_t1 else 1)
        ovh4 = m_ovh.reindex(idx).ffill(limit=0 if lag_t1 else 1).fillna(False)
    else:
        f4, ovh4 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)'''

NEW_BLOCK = '''    if etf_df is not None and not etf_df.empty:
        if lag_t1:
            f2 = etf_df["sig"].reindex(idx).shift(1)
            ext2 = etf_df["extreme"].reindex(idx).shift(1).eq(True)
        else:
            f2 = etf_df["sig"].reindex(idx).ffill(limit=1)
            ext2 = etf_df["extreme"].reindex(idx).ffill(limit=1).eq(True)
    else:
        f2, ext2 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)

    if mar_df is not None and not mar_df.empty:
        if lag_t1:
            f4 = mar_df["sig"].reindex(idx).shift(1)
            ovh4 = mar_df["overheated"].reindex(idx).shift(1).eq(True)
        else:
            f4 = mar_df["sig"].reindex(idx).ffill(limit=1)
            ovh4 = mar_df["overheated"].reindex(idx).ffill(limit=1).eq(True)
    else:
        f4, ovh4 = pd.Series(np.nan, index=idx), pd.Series(False, index=idx)'''


def main():
    if not TARGET.exists():
        sys.exit(f"[中止] 找不到 {TARGET}")
    src = TARGET.read_text(encoding="utf-8-sig")
    if NEW_BLOCK in src:
        print("[跳过] analytics.py 已修复")
        return
    if src.count(OLD_BLOCK) != 1:
        sys.exit(f"[中止] 未找到目标代码块（匹配数：{src.count(OLD_BLOCK)}）")
    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    compile(new_src, str(TARGET), "exec")
    TARGET.write_text(new_src, encoding="utf-8")
    print("[修复] analytics.py（移除 ffill(limit=0)，并在回测模式下按交易日索引准确滞后1日）")
    print("请刷新网页或重启 streamlit run app.py。")


if __name__ == "__main__":
    main()