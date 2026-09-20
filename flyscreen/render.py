"""点云渲染：加色叠加 + 盒式模糊辉光。

分三层叠色，每层独立着色：

    结构层   未点亮的神经元，暗蓝，保留果蝇脑的形状（**静态，只在 _build 里算一次**）
    驱动层   视频直接点亮的神经元，冷色（青白）—— 这是"你给它的"
    脉冲层   LIF 沿连接组传播出来的活动，暖色（琥珀）—— 这是"它自己的反应"

分色不是为了好看，是为了诚实：一眼能看出哪些是输入、哪些是网络的回应。

性能：这是整条管线里最贵的一步。实测下来（600×490 画布）有三个关键点：

1. **结构层是静态的** —— 放到 `_build()` 里算一次，别每帧重算。
2. **`np.power(v, 1.0)` 等于什么都没做，但 `pow` 很贵** —— fade 为 0/1 时直接跳过。
3. **别每帧新建 `(h,w,3)` 临时数组**。每层新建两个 3.5 MB 数组看着不多，
   但那是全新的内存页，缺页中断把 `_layer` 拖到 12 ms（只做 240 万次浮点运算，
   相当于 2 亿次/秒，比 numpy 正常速度慢一个数量级）。
   改成**三个预分配的 (h,w) 颜色平面 + 原地运算** 之后就正常了。
"""
from __future__ import annotations

import numpy as np


def _box_blur_1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return a
    n = a.shape[axis]
    k = 2 * r + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad)
    c = np.cumsum(p, axis=axis)
    zshape = list(a.shape)
    zshape[axis] = 1
    c = np.concatenate([np.zeros(zshape, dtype=c.dtype), c], axis=axis)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, k + n)
    lo[axis] = slice(0, n)
    return (c[tuple(hi)] - c[tuple(lo)]) / float(k)


def box_blur(a: np.ndarray, r: int, passes: int = 1) -> np.ndarray:
    out = a
    for _ in range(max(0, passes)):
        if r <= 0:
            break
        out = _box_blur_1d(out, r, 0)
        out = _box_blur_1d(out, r, 1)
    return out


class PointCloudRenderer:
    """把神经元活动渲染成一张 RGB 图。"""

    def __init__(self, canvas, cfg) -> None:
        self.cfg = cfg
        self.canvas = canvas
        self.w = int(cfg.canvas_w)
        self.h = int(cfg.canvas_h)
        self._build()

    # ------------------------------------------------------------ 预计算
    def _build(self) -> None:
        c = self.canvas
        sx = np.clip((c.u * (self.w - 1)).round().astype(np.int64), 0, self.w - 1)
        sy = np.clip((c.v * (self.h - 1)).round().astype(np.int64), 0, self.h - 1)
        self.flat = sy * self.w + sx
        self.minlength = self.w * self.h

        hw = (self.h, self.w)
        # 三个颜色平面（替代每层新建 (h,w,3) 临时数组）
        self._plane = [np.zeros(hw, np.float32) for _ in range(3)]
        self._v = np.empty(hw, np.float32)      # v
        self._v2 = np.empty(hw, np.float32)     # v^2
        self._t = np.empty(hw, np.float32)      # 通用暂存
        self._acc = np.empty(hw, np.float32)
        self._out = np.empty((self.h, self.w, 3), np.uint8)
        self._bg = tuple(float(x) for x in self.cfg.background)

        # 结构层静态，算一次并缓存成三个平面
        struct = self._splat(c.weight * float(self.cfg.structure_gain))
        self._struct_plane = self._make_planes(struct, 1.0, (0, 0, 0),
                                               self.cfg.structure_color, fade=0.0)

    def set_size(self, w: int, h: int) -> None:
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = int(w), int(h)
        self._build()

    # ------------------------------------------------------------ 内部
    def _splat(self, intensity: np.ndarray) -> np.ndarray:
        acc = np.bincount(self.flat, weights=intensity, minlength=self.minlength)
        return acc.reshape(self.h, self.w).astype(np.float32)

    def _make_planes(self, acc: np.ndarray, gain: float, lo, hi, fade: float):
        """算出一层的三个颜色平面（新建数组，只有 _build 里用）。"""
        v = acc * np.float32(gain)
        np.clip(v, 0.0, 1.0, out=v)
        if fade == 1.0:
            p = v * v
        elif fade == 0.0:
            p = v
        else:
            p = np.power(v, np.float32(fade))
        return [np.asarray(lo, np.float32)[c] + (np.asarray(hi, np.float32)[c]
                - np.asarray(lo, np.float32)[c]) * p
                for c in range(3)]

    def _add_layer(self, acc: np.ndarray, gain: float, lo, hi, fade: float) -> None:
        """把一层原地累加到 self._plane 上，不新建任何大数组。

            fade=0:  rgb_c += lo_c + (hi_c-lo_c)*v
            fade=1:  rgb_c += lo_c*v + (hi_c-lo_c)*v*v
        """
        v = self._v
        np.multiply(acc, np.float32(gain), out=v)
        np.clip(v, 0.0, 1.0, out=v)
        t = self._t
        if fade == 1.0:
            vv = self._v2
            np.multiply(v, v, out=vv)
            for c in range(3):
                p = self._plane[c]
                lc = np.float32(lo[c])
                dc = np.float32(hi[c] - lo[c])
                if lc != 0.0:
                    np.multiply(v, lc, out=t)
                    np.add(p, t, out=p)
                if dc != 0.0:
                    np.multiply(vv, dc, out=t)
                    np.add(p, t, out=p)
        else:
            for c in range(3):
                p = self._plane[c]
                lc = np.float32(lo[c])
                dc = np.float32(hi[c] - lo[c])
                if lc != 0.0:
                    np.add(p, lc, out=p)
                if dc != 0.0:
                    np.multiply(v, dc, out=t)
                    np.add(p, t, out=p)

    # ------------------------------------------------------------ 渲染
    def render(self, spikes: np.ndarray | None = None,
               drive: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        # 起点 = 缓存好的结构层
        for c in range(3):
            np.copyto(self._plane[c], self._struct_plane[c])

        act_sum = None
        if drive is not None:
            acc = self._splat(np.asarray(drive, dtype=np.float32) * self.canvas.weight)
            act_sum = acc
            self._add_layer(acc, cfg.drive_gain,
                            cfg.drive_color_lo, cfg.drive_color_hi, fade=1.0)

        if spikes is not None:
            acc = self._splat(np.asarray(spikes, dtype=np.float32) * self.canvas.weight)
            if act_sum is None:
                act_sum = acc
            else:
                np.add(act_sum, acc, out=act_sum)
            self._add_layer(acc, cfg.activity_gain,
                            cfg.spike_color_lo, cfg.spike_color_hi, fade=1.0)

        # 辉光只加在活动上（结构层不发光，保持干净）
        if act_sum is not None and cfg.glow_passes > 0 and cfg.glow_radius > 0:
            glow = box_blur(act_sum, int(cfg.glow_radius), int(cfg.glow_passes))
            self._add_layer(glow, cfg.glow_mix, (0, 0, 0), cfg.glow_color, fade=0.0)

        # 合成 uint8
        out = self._out
        for c in range(3):
            p = self._plane[c]
            np.clip(p, self._bg[c], 255.0, out=p)
            out[..., c] = p
        return out
