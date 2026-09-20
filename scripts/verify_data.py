"""数据完整性自检：三个 feather 的大小 / schema / 行数 / 可读性。"""
from __future__ import annotations

import sys
from pathlib import Path

RAW = Path(r"D:\fly-brain\flyscreen\data\raw")
EXPECT = {
    "body-annotations.feather": 14_483_314,
    "connectome-weights.feather": 1_051_241_946,
    "syn-partners.feather": 6_777_179_098,
}


def check_small(p: Path) -> bool:
    import pyarrow.feather as feather

    t = feather.read_table(str(p))
    print(f"  schema: {len(t.column_names)} 列")
    print(f"  columns: {t.column_names}")
    print(f"  rows: {t.num_rows:,}")
    # 真读一遍数据，触发 CRC / 解压校验
    n = 0
    for c in t.column_names:
        n += len(t[c].to_numpy(zero_copy_only=False))
    print(f"  读取全部列成功（累计 {n:,} 个值）")
    return True


def check_stream(p: Path, key_col: str) -> bool:
    """大文件：按批扫描，边读边校验，不整表载入。"""
    import pyarrow.dataset as ds

    d = ds.dataset(str(p), format="feather")
    print(f"  schema: {len(d.schema.names)} 列")
    print(f"  columns: {d.schema.names}")
    total = 0
    mn, mx = None, None
    import numpy as np

    for b in d.scanner(columns=[key_col], batch_size=8_000_000).to_batches():
        a = b.column(key_col).to_numpy(zero_copy_only=False)
        total += len(a)
        lo, hi = int(a.min()), int(a.max())
        mn = lo if mn is None else min(mn, lo)
        mx = hi if mx is None else max(mx, hi)
    print(f"  rows: {total:,}")
    print(f"  {key_col} 范围: {mn:,} .. {mx:,}")
    return True


def main() -> int:
    bad = 0
    for name, expect in EXPECT.items():
        p = RAW / name
        print(f"\n{'='*70}\n{name}")
        if not p.exists():
            print("  !! 不存在")
            bad += 1
            continue
        size = p.stat().st_size
        ok_size = size == expect
        print(f"  大小: {size:,} / {expect:,}  {'OK' if ok_size else '不匹配'}")
        if not ok_size:
            print(f"  完成度 {100.0*size/expect:.1f}% -> 跳过内容校验")
            bad += 1
            continue
        try:
            if name == "body-annotations.feather":
                check_small(p)
            else:
                # 大文件都是 (body_pre, body_post, weight) 三元组
                check_stream(p, "body_pre")
            print("  => 内容校验通过")
        except Exception as e:  # noqa: BLE001
            print(f"  => 内容校验失败: {type(e).__name__}: {e}")
            bad += 1
    print(f"\n{'='*70}\n失败项: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
