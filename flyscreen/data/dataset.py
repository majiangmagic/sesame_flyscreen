"""加载预处理好的神经元表缓存。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR


@dataclass
class NeuronTable:
    body_id: np.ndarray  # int64[N]
    pos_um: np.ndarray  # float32[N,3]
    cls: np.ndarray  # int16[N]
    classes: np.ndarray  # <U32[k]
    side: np.ndarray  # int8[N]
    groups: np.ndarray  # bool[N,8]
    group_names: np.ndarray  # <U16[8]
    nt_sign: np.ndarray | None = None  # int8[N]  0=兴奋 1=抑制
    nt_name: np.ndarray | None = None  # <U16[N]

    @property
    def n(self) -> int:
        return int(self.pos_um.shape[0])

    @property
    def extent(self) -> np.ndarray:
        return self.pos_um.max(axis=0) - self.pos_um.min(axis=0)

    @property
    def center(self) -> np.ndarray:
        lo = self.pos_um.min(axis=0)
        hi = self.pos_um.max(axis=0)
        return (lo + hi) / 2.0

    def group_dict(self) -> dict[str, np.ndarray]:
        return {
            str(name): np.flatnonzero(self.groups[:, i])
            for i, name in enumerate(self.group_names)
        }

    def class_colors(self, palette: np.ndarray) -> np.ndarray:
        """按 superclass 给每个神经元分配 RGB 颜色。"""
        out = np.zeros((self.n, 3), dtype=np.uint8)
        k = len(self.classes)
        for i in range(self.n):
            c = int(self.cls[i])
            if 0 <= c < k:
                out[i] = palette[c % len(palette)]
        return out


def load(path: Path | None = None) -> NeuronTable:
    path = Path(path) if path else (CACHE_DIR / "neurons.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"缺少神经元缓存 {path}\n先生成：python -m flyscreen.data.prepare"
        )
    z = np.load(path, allow_pickle=False)
    return NeuronTable(
        body_id=z["body_id"],
        pos_um=z["pos_um"],
        cls=z["cls"],
        classes=z["classes"],
        side=z["side"],
        groups=z["groups"],
        group_names=z["group_names"],
        nt_sign=z["nt_sign"] if "nt_sign" in z.files else None,
        nt_name=z["nt_name"] if "nt_name" in z.files else None,
    )


def load_graph(path: Path | None = None):
    """加载连接组 CSR 图。

    返回 (indptr int64[N+1], indices int32[E], wsyn float32[E], n)。
    wsyn 已带符号（突触前决定，Dale 法则）。
    """
    path = Path(path) if path else (CACHE_DIR / "graph.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"缺少连接图缓存 {path}\n先生成：python -m flyscreen.data.prepare_graph"
        )
    z = np.load(path, allow_pickle=False)
    return z["indptr"], z["indices"], z["wsyn"], int(z["n"])
