"""把运动神经元群的活动变成 8 路关节力矩信号。

**为什么不能直接用活动水平**：视频把大半个脑点亮之后，每个运动神经元群的
平均活动是一个又高又稳的数（实测翅膀那两路恒为 2.1，被钳死在最大张开角度）。
用水平当力矩 = 果蝇摆一个固定姿势不动。

**抽搐的本质是变化，不是水平。** 所以这里做两件事：

1. **逐路自适应归一化** —— 每路对自己的慢基线和慢尺度做 z-score。
   翅膀那路基线是 2.1 也没关系，输出照样在 -1..1 之间摆动。
2. **混合两个分量**：偏离慢基线的部分（抽搐） + 变化率（抖动）。

慢基线同时兼任"它现在整体有多活跃"，所以视频节奏快的时候基线抬升、
抽搐幅度自然变大。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import BodyConfig


@dataclass
class ChannelState:
    base: float = 0.0
    scale: float = 0.0
    prev: float = 0.0
    last: float = 0.0
    n: int = 0


class BodyDriver:
    """活动数组 -> {部位名: [-1,1] 的关节驱动}。"""

    def __init__(self, groups: dict[str, np.ndarray], cfg: BodyConfig) -> None:
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        self.cfg = cfg
        self.state = {k: ChannelState() for k in self.groups}
        # 每个部位的总活动量（均值）—— 用归一化前先算好
        self._buf = {k: 0.0 for k in self.groups}

    def reset(self) -> None:
        for st in self.state.values():
            st.base = st.scale = st.prev = st.last = 0.0
            st.n = 0

    def raw_rates(self, activity: np.ndarray) -> dict[str, float]:
        return {k: float(activity[idx].mean()) for k, idx in self.groups.items()}

    def update(self, activity: np.ndarray, dt: float) -> dict[str, float]:
        cfg = self.cfg
        out: dict[str, float] = {}
        tau = max(1e-3, float(cfg.baseline_tau))
        alpha = min(1.0, dt / tau)

        for k, idx in self.groups.items():
            st = self.state[k]
            r = float(activity[idx].mean()) if idx.size else 0.0

            if st.n < 3:
                # 预热：先把基线拉到位，避免开头几下猛抽
                st.base = r if st.n == 0 else st.base + (r - st.base) * 0.5
                st.scale = max(st.scale, 1e-3)
                st.prev = r
                st.last = r
                st.n += 1
                out[k] = 0.0
                continue

            # 慢基线（姿态）与慢尺度（幅度）
            st.base += (r - st.base) * alpha
            st.scale += (abs(r - st.base) - st.scale) * alpha
            scale = max(1e-4, st.scale)

            # 偏离基线 = 抽搐；再除以自身尺度，各路自动等权
            dev = (r - st.base) / scale
            dev = float(np.clip(dev, -3.0, 3.0)) / 3.0

            # 变化率 = 抖动
            d = (r - st.prev) / max(1e-4, dt)
            st.prev = r
            d = float(np.clip(d, -1e3, 1e3)) / 1e3

            st.last = r
            v = float(cfg.twitch_gain) * dev + float(cfg.jitter_gain) * d
            out[k] = float(np.clip(v, -1.0, 1.0))
        return out
