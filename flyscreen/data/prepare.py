"""从 MaleCNS v1.0 注释构建神经元表缓存。

坐标直接取自 annotations 里的 `somaLocation`（8nm 体素），**不需要 6.8 GB 的
syn-partners**。syn-partners 只在想要突触级细节时才用。

输出 data/cache/neurons.npz：
    body_id     int64[N]      神经元 ID
    pos_um      float32[N,3]  胞体坐标（微米）
    cls         int16[N]      superclass 索引
    classes     <U32[k]
    side        int8[N]       -1 未知 / 0 左 / 1 右
    groups      bool[N,8]     8 个身体部位（6 腿 + 2 翅）的神经元掩码
    group_names <U16[8]

用法：
    python -m flyscreen.data.prepare
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, NM_PER_UM, RAW_DIR, VOXEL_NM

GROUP_NAMES = [
    "leg_front_left",
    "leg_front_right",
    "leg_mid_left",
    "leg_mid_right",
    "leg_hind_left",
    "leg_hind_right",
    "wing_left",
    "wing_right",
]

# MaleCNS 注释里腿/翅运动神经元池的 subclass 标记
#   fl = front leg, ml = mid leg, hl = hind leg, wm = wing motor neuron
LEG_SUBCLASS = ("leg_front", "fl"), ("leg_mid", "ml"), ("leg_hind", "hl")
WING_SUBCLASS = "wm"

EXCLUDE_STATUS = {"Glia"}

# 神经递质 -> 符号。Dale 法则：整细胞同号。
# 乙酰胆碱/多巴胺/章鱼胺/血清素/unclear 为兴奋；GABA/谷氨酸/组胺为抑制
# （谷氨酸在成体果蝇经 GluCl 起抑制作用）。
EXCITATORY_NT = {"acetylcholine", "dopamine", "octopamine", "serotonin", "unclear", "glutamate?"}
INHIBITORY_NT = {"gaba", "glutamate", "histamine"}


def _nt_sign(name: str) -> int:
    n = (name or "").strip().lower()
    if n in INHIBITORY_NT:
        return 1  # 1 = 抑制
    if n in EXCITATORY_NT:
        return 0  # 0 = 兴奋
    return 0  # 未知按兴奋处理（与参考实现一致）


def _str_col(t, name: str) -> np.ndarray:
    return np.array(
        [("" if v is None else str(v).strip()) for v in t[name].to_numpy(zero_copy_only=False)],
        dtype=object,
    )


def _side_array(t) -> np.ndarray:
    for col in ("somaSide", "rootSide"):
        if col not in t.column_names:
            continue
        v = _str_col(t, col)
        out = np.full(len(v), -1, dtype=np.int8)
        out[np.isin(v, ["L", "l", "left"])] = 0
        out[np.isin(v, ["R", "r", "right"])] = 1
        if (out >= 0).sum() > 0.5 * len(out):
            return out
    return np.full(t.num_rows, -1, dtype=np.int8)


def load_neurotransmitters(path: Path | None = None) -> dict[int, str]:
    """读 body-neurotransmitters feather，返回 {body_id: 递质名}。

    优先 consensus_nt，退回 predicted_nt，再退回 celltype_predicted_nt。
    """
    import pyarrow.feather as feather

    path = path or (RAW_DIR / "body-neurotransmitters.feather")
    if not Path(path).exists():
        print(f"[nt] 没有 {path.name}，全部按兴奋处理")
        return {}
    t = feather.read_table(str(path))
    cols = set(t.column_names)

    def col(name: str):
        if name not in cols:
            return [""] * t.num_rows
        return [("" if v is None else str(v).strip().lower()) for v in t[name].to_numpy(zero_copy_only=False)]

    body = t["body"].to_numpy(zero_copy_only=False).astype(np.int64)
    consensus = col("consensus_nt")
    pred = col("predicted_nt")
    celltype = col("celltype_predicted_nt")

    out: dict[int, str] = {}
    for i in range(t.num_rows):
        v = consensus[i] or pred[i] or celltype[i]
        out[int(body[i])] = v
    print(f"[nt] 载入 {len(out):,} 条神经递质预测 ({Path(path).name})")
    return out


def load_neurons(ann_path: Path | None = None, nt_path: Path | None = None) -> dict:
    import pyarrow.feather as feather

    ann_path = ann_path or (RAW_DIR / "body-annotations.feather")
    if not Path(ann_path).exists():
        raise FileNotFoundError(
            f"缺少 {ann_path}\n先下载：python -m flyscreen.data.fetch"
        )

    t = feather.read_table(str(ann_path))
    n_all = t.num_rows
    print(f"[annotations] {n_all:,} 行 × {len(t.column_names)} 列")

    body_id = t["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64)

    # ---- 胞体坐标
    loc = t["somaLocation"].to_pylist()
    has = np.array([v is not None and len(v) == 3 for v in loc], dtype=bool)
    print(f"[annotations] 含 somaLocation 的神经元: {has.sum():,} ({100.0*has.sum()/n_all:.1f}%)")

    # ---- 去掉胶质细胞
    status = _str_col(t, "status") if "status" in t.column_names else np.array([""] * n_all, dtype=object)
    keep = has & ~np.isin(status, list(EXCLUDE_STATUS))
    n_glia = int((has & np.isin(status, list(EXCLUDE_STATUS))).sum())
    print(f"[annotations] 剔除胶质细胞 {n_glia:,} 个，保留 {keep.sum():,} 个神经元")

    idx = np.flatnonzero(keep)
    coords = np.array([loc[i] for i in idx], dtype=np.float64)
    pos_um = (coords * (VOXEL_NM / NM_PER_UM)).astype(np.float32)

    superclass = _str_col(t, "superclass") if "superclass" in t.column_names else np.array([""] * n_all, dtype=object)
    subclass = _str_col(t, "subclass") if "subclass" in t.column_names else np.array([""] * n_all, dtype=object)
    type_ = _str_col(t, "type") if "type" in t.column_names else np.array([""] * n_all, dtype=object)
    side = _side_array(t)

    # ---- 神经递质 -> 兴奋/抑制
    nt_map = load_neurotransmitters(nt_path)
    nt_sign = np.zeros(len(idx), dtype=np.int8)
    nt_name = np.array([""] * len(idx), dtype=object)
    hit = 0
    for k, i in enumerate(idx):
        name = nt_map.get(int(body_id[i]), "")
        nt_name[k] = name
        nt_sign[k] = _nt_sign(name)
        if name:
            hit += 1
    print(
        f"[nt] 命中 {hit:,}/{len(idx):,}；"
        f"兴奋 {int((nt_sign == 0).sum()):,} / 抑制 {int((nt_sign == 1).sum()):,}"
    )

    return {
        "body_id": body_id[idx],
        "pos_um": pos_um,
        "superclass": superclass[idx],
        "subclass": subclass[idx],
        "type": type_[idx],
        "side": side[idx],
        "nt_sign": nt_sign,
        "nt_name": nt_name,
    }


def encode_classes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names: list[str] = []
    index: dict[str, int] = {}
    codes = np.full(len(values), -1, dtype=np.int16)
    for i, v in enumerate(values):
        if not v:
            continue
        c = index.get(v)
        if c is None:
            c = len(names)
            index[v] = c
            names.append(v)
        codes[i] = c
    return codes, np.asarray(names, dtype="<U32")


def build_groups(subclass: np.ndarray, side: np.ndarray) -> np.ndarray:
    """8 个身体部位的神经元掩码：subclass ∈ {fl,ml,hl,wm} × 左/右。"""
    n = len(subclass)
    groups = np.zeros((n, len(GROUP_NAMES)), dtype=bool)
    sub_l = np.array([s.lower() for s in subclass], dtype=object)
    side = np.asarray(side)

    for k, (_name, code) in enumerate(LEG_SUBCLASS):
        m = sub_l == code
        for s, off in ((0, 0), (1, 1)):
            groups[:, k * 2 + off] = m & (side == s)
    mw = sub_l == WING_SUBCLASS
    groups[:, 6] = mw & (side == 0)
    groups[:, 7] = mw & (side == 1)
    return groups


def main() -> int:
    d = load_neurons()

    cls, classes = encode_classes(d["superclass"])
    groups = build_groups(d["subclass"], d["side"])

    print("\n[groups] 8 个驱动部位：")
    tot = 0
    for i, name in enumerate(GROUP_NAMES):
        c = int(groups[:, i].sum())
        tot += c
        flag = "" if c else "   <-- 空！"
        print(f"    {name:18s} {c:>6,d}{flag}")
    print(f"    {'合计':18s} {tot:>6,d}")

    print("\n[classes] superclass 分布：")
    u, c = np.unique(cls[cls >= 0], return_counts=True)
    for i in np.argsort(-c)[:14]:
        print(f"    {classes[u[i]]:28s} {c[i]:>8,d}")

    ext = d["pos_um"].max(axis=0) - d["pos_um"].min(axis=0)
    print(f"\n[neurons] {len(d['body_id']):,} 个")
    print(f"[neurons] 包围盒 (µm) XYZ = {np.round(ext, 1)}")

    out = CACHE_DIR / "neurons.npz"
    np.savez_compressed(
        out,
        body_id=d["body_id"].astype(np.int64),
        pos_um=d["pos_um"],
        cls=cls,
        classes=classes,
        side=d["side"].astype(np.int8),
        groups=groups,
        group_names=np.asarray(GROUP_NAMES, dtype="<U16"),
        nt_sign=d["nt_sign"].astype(np.int8),
        nt_name=np.asarray(d["nt_name"], dtype="<U16"),
    )
    print(f"\n[neurons] 写入 {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
