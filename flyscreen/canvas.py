"""视频帧 -> 神经元驱动。

把 176k 个神经元的 3D 质心投影到 2D（背面俯视），视频帧按双线性采样
打到每个神经元上。视叶神经元极密，所以额外做密度均衡，否则画面会
塌成两块白斑。

同时维护一层"荧光屏拖尾"：每帧先衰减，再取 max(衰减后, 当前驱动)。
"""
from __future__ import annotations

import numpy as np

from .config import CanvasConfig


class NeuronCanvas:
    def __init__(self, pos_um: np.ndarray, cfg: CanvasConfig) -> None:
        self.cfg = cfg
        self.pos = np.asarray(pos_um, dtype=np.float32)
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError(f"pos_um 形状应为 (N,3)，实际 {self.pos.shape}")
        self.n = self.pos.shape[0]

        self.u = np.zeros(self.n, dtype=np.float32)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.weight = np.ones(self.n, dtype=np.float32)

        self.activity = np.zeros(self.n, dtype=np.float32)

        self.set_view(cfg.view_axes)

    # ------------------------------------------------------------ 投影
    def set_view(self, axes: tuple[int, int]) -> None:
        """切换投影轴（0=x, 1=y, 2=z）。"""
        ax, ay = int(axes[0]), int(axes[1])
        if ax == ay or not (0 <= ax < 3 and 0 <= ay < 3):
            raise ValueError(f"非法的投影轴 {axes}")
        self.axes = (ax, ay)

        a = self.pos[:, ax].astype(np.float64)
        b = self.pos[:, ay].astype(np.float64)

        p = float(np.clip(self.cfg.clip_percentile, 0.0, 20.0))
        alo, ahi = np.percentile(a, [p, 100.0 - p])
        blo, bhi = np.percentile(b, [p, 100.0 - p])
        if ahi <= alo:
            ahi = alo + 1.0
        if bhi <= blo:
            bhi = blo + 1.0

        self.u = np.clip((a - alo) / (ahi - alo), 0.0, 1.0).astype(np.float32)
        # 图像 y 轴向下，所以翻转，让点云正着显示
        self.v = np.clip(1.0 - (b - blo) / (bhi - blo), 0.0, 1.0).astype(np.float32)

        self._update_weight()

    def _update_weight(self) -> None:
        """按局部神经元密度给每个神经元一个可见度权重。"""
        mode = self.cfg.density_equalize
        if mode == "none":
            self.weight = np.ones(self.n, dtype=np.float32)
            return

        g = max(8, int(self.cfg.density_grid))
        ix = np.clip((self.u * (g - 1)).astype(np.int32), 0, g - 1)
        iy = np.clip((self.v * (g - 1)).astype(np.int32), 0, g - 1)
        counts = np.bincount(ix * g + iy, minlength=g * g).astype(np.float32)
        cell = counts[ix * g + iy]
        cell = np.maximum(cell, 1.0)

        if mode == "full":
            w = 1.0 / cell
        elif mode == "sqrt":
            w = 1.0 / np.sqrt(cell)
        else:
            raise ValueError(f"未知的 density_equalize: {mode!r}")

        m = float(w.mean())
        if m > 0:
            w = w / m
        self.weight = w.astype(np.float32)

    # ------------------------------------------------------------ 驱动
    def drive_from_frame(self, gray: np.ndarray) -> np.ndarray:
        """把一帧灰度图（H,W; 0..1）采样成每个神经元的驱动强度。"""
        g = np.asarray(gray, dtype=np.float32)
        if g.ndim != 2:
            raise ValueError(f"灰度帧应为 (H,W)，实际 {g.shape}")
        h, w = g.shape
        if h < 2 or w < 2:
            return np.zeros(self.n, dtype=np.float32)

        py = self.v * (h - 1)
        px = self.u * (w - 1)

        y0 = np.floor(py).astype(np.int32)
        x0 = np.floor(px).astype(np.int32)
        wy = (py - y0).astype(np.float32)
        wx = (px - x0).astype(np.float32)

        y0c = np.clip(y0, 0, h - 1)
        x0c = np.clip(x0, 0, w - 1)
        y1c = np.clip(y0 + 1, 0, h - 1)
        x1c = np.clip(x0 + 1, 0, w - 1)

        v00 = g[y0c, x0c]
        v01 = g[y0c, x1c]
        v10 = g[y1c, x0c]
        v11 = g[y1c, x1c]

        val = (
            v00 * (1.0 - wy) * (1.0 - wx)
            + v01 * (1.0 - wy) * wx
            + v10 * wy * (1.0 - wx)
            + v11 * wy * wx
        )

        gain = float(self.cfg.gain)
        gamma = float(self.cfg.gamma)
        lo = float(self.cfg.black_point)
        hi = float(self.cfg.white_point)
        if hi <= lo:
            hi = lo + 1e-6
        if lo > 0 or hi < 1.0:
            val = np.clip((np.clip(val, 0.0, 1.0) - lo) / (hi - lo), 0.0, 1.0)
        if gamma > 0 and abs(gamma - 1.0) > 1e-6:
            val = np.power(np.clip(val, 0.0, 1.0), gamma)
        return (val * gain).astype(np.float32)

    def advance(self, drive: np.ndarray) -> np.ndarray:
        """推进一层拖尾并返回当前活动强度。"""
        cfg = self.cfg
        self.activity *= float(cfg.decay)
        np.maximum(self.activity, drive, out=self.activity)
        if cfg.floor > 0:
            self.activity[self.activity < cfg.floor] = 0.0
        return self.activity

    def decay_factor(self, dt_ms: float, frame_ms: float) -> float:
        """把"每帧衰减"换算成"每 tick 衰减"。"""
        if frame_ms <= 0:
            return float(self.cfg.decay)
        return float(self.cfg.decay) ** (dt_ms / frame_ms)

    def advance_trace(self, spikes: np.ndarray, decay_per_tick: float) -> np.ndarray:
        """荧光屏拖尾：先衰减，再取 max(衰减后, 本 tick 的脉冲)。"""
        self.activity *= decay_per_tick
        np.maximum(self.activity, spikes, out=self.activity)
        if self.cfg.floor > 0:
            self.activity[self.activity < self.cfg.floor] = 0.0
        return self.activity

    def set_activity(self, values: np.ndarray) -> None:
        np.copyto(self.activity, values, casting="unsafe")

    def reset(self) -> None:
        self.activity.fill(0.0)

    # ------------------------------------------------------------ 身体驱动
    def rates(self, groups: dict[str, np.ndarray], boost: float = 4.0) -> dict[str, float]:
        """把当前活动按神经元分组求均值，得到每个身体部位的驱动标量 (0..1)。

        groups: {部位名: 该部位的神经元索引数组}
        每个部位只有几十到一百多个运动神经元，所以乘一个增益让信号可用。
        """
        act = self.activity
        out: dict[str, float] = {}
        for name, idx in groups.items():
            if idx.size == 0:
                out[name] = 0.0
                continue
            out[name] = float(np.clip(float(act[idx].mean()) * boost, 0.0, 1.0))
        return out
