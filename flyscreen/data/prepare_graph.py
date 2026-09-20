"""把 connectome-weights 变成 LIF 能用的 CSR 稀疏图。

输入 data/raw/connectome-weights.feather   (body_pre, body_post, weight)

注意：这个文件里是**原始分割 ID**，包含大量未注释的碎片（body_post 有 8700 万个
唯一值）。必须映射到我们注释过的神经元表上，只保留两端都在表里的边。

输出 data/cache/graph.npz
    indptr   int32[N+1]    按突触前神经元分组的 CSR 行指针
    indices  int32[E]      突触后神经元下标
    wsyn     float32[E]    带符号的突触权重 = 突触数 × 符号（突触前决定）
    n, e     int64
    min_weight int64

用法：
    python -m flyscreen.data.prepare_graph
    python -m flyscreen.data.prepare_graph --min-weight 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, RAW_DIR
from . import dataset

# Shiu et al. 2024 的每突触权重（mV）。MaleCNS 比雌脑突触更密，
# 参考实现在 male-cns 上把它标定到 0.65 倍。
MV_PER_SYNAPSE = 0.275
DEFAULT_GAIN = 0.65


def build(min_weight: int = 5, batch_rows: int = 4_000_000) -> Path:
    import pyarrow.dataset as ds

    src = RAW_DIR / "connectome-weights.feather"
    if not src.exists():
        raise FileNotFoundError(f"缺少 {src}\n先跑：python -m flyscreen.data.fetch")

    t = dataset.load()
    print(f"[graph] 神经元表 {t.n:,} 个")
    ids = t.body_id  # 已按升序保存
    if not np.all(np.diff(ids) > 0):
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        raise RuntimeError("神经元表未按 body_id 升序，请重新跑 prepare")

    sign = t.nt_sign if t.nt_sign is not None else np.zeros(t.n, dtype=np.int8)
    sign_f = np.where(sign == 1, -1.0, 1.0).astype(np.float32)
    print(f"[graph] 符号：兴奋 {int((sign == 0).sum()):,} / 抑制 {int((sign == 1).sum()):,}")

    d = ds.dataset(str(src), format="feather")
    have = set(d.schema.names)
    need = {"body_pre", "body_post", "weight"}
    if not need <= have:
        raise RuntimeError(f"connectome-weights 缺少列 {need - have}，实际 {sorted(have)}")

    pre_l: list[np.ndarray] = []
    post_l: list[np.ndarray] = []
    w_l: list[np.ndarray] = []
    rows = 0
    kept = 0
    t0 = time.time()

    for b in d.scanner(columns=["body_pre", "body_post", "weight"], batch_size=batch_rows).to_batches():
        p = b.column("body_pre").to_numpy(zero_copy_only=False).astype(np.int64)
        q = b.column("body_post").to_numpy(zero_copy_only=False).astype(np.int64)
        w = b.column("weight").to_numpy(zero_copy_only=False).astype(np.int32)
        rows += p.size

        pi = np.searchsorted(ids, p)
        np.clip(pi, 0, t.n - 1, out=pi)
        qi = np.searchsorted(ids, q)
        np.clip(qi, 0, t.n - 1, out=qi)

        keep = (ids[pi] == p) & (ids[qi] == q)
        if min_weight > 1:
            keep &= w >= min_weight

        rows_kept = int(keep.sum())
        if rows_kept:
            pre_l.append(pi[keep].astype(np.int32))
            post_l.append(qi[keep].astype(np.int32))
            w_l.append(w[keep].astype(np.float32))
            kept += rows_kept

        el = time.time() - t0
        print(
            f"[graph] 扫过 {rows:>13,} 行  保留 {kept:>11,}  "
            f"({100.0*kept/max(1,rows):5.2f}%)  {el:6.1f}s",
            flush=True,
        )

    if not pre_l:
        raise RuntimeError("没有任何边落在神经元表上")

    pre = np.concatenate(pre_l)
    post = np.concatenate(post_l)
    w = np.concatenate(w_l)
    del pre_l, post_l, w_l

    print(f"[graph] 排序 {pre.size:,} 条边 ...")
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = post[order]
    w = w[order]
    del order

    # 带符号权重：符号由突触前神经元决定（Dale 法则）
    wsyn = (w * sign_f[pre]).astype(np.float32)

    indptr = np.zeros(t.n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    np.cumsum(indptr, out=indptr)
    indptr = indptr.astype(np.int32)

    deg = np.diff(indptr)
    print(f"[graph] 完成：{pre.size:,} 条边，平均出度 {deg.mean():.1f}，最大 {deg.max():,}")
    print(f"[graph] 孤立神经元（出度 0）：{int((deg == 0).sum()):,}")
    print(f"[graph] 入度统计：平均 {pre.size/t.n:.1f}")

    out = CACHE_DIR / "graph.npz"
    np.savez(
        out,
        indptr=indptr,
        indices=post.astype(np.int32),
        wsyn=wsyn,
        n=np.int64(t.n),
        e=np.int64(pre.size),
        min_weight=np.int64(min_weight),
    )
    print(f"[graph] 写入 {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-weight", type=int, default=5,
                    help="只保留突触数 >= 该值的连接（默认 5，与参考实现一致）")
    a = ap.parse_args(argv)
    build(min_weight=a.min_weight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
