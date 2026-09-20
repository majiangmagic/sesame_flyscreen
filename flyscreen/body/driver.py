"""把运动神经元群的活动变成 8 路关节力矩信号。

**为什么不能直接用活动水平**：视频把大半个脑点亮之后，每个运动神经元群的
平均活动是一个又高又稳的数（实测翅膀那两路恒为 2.1，被钳死在最大张开角度）。
用水平当力矩 = 果蝇摆一个固定姿势不动。

**抽搐的本质是变化，不是水平。** 所以这里做三件事：

1. **逐路减慢基线，再除以固定的参考尺度** —— dev = (r - 慢基线) / dev_ref[部位]。
   翅膀那路基线是 2.1 也没关系，各路自动等权。
   注意：这里**不再**除以"该路自己当前的波动幅度"。浮动尺度会把任何输入都
   归一化到满幅 —— 实测恒定不动的画面也抽到 |v|=0.490，与真实视频几乎一样，
   于是"抽得多猛"不再携带"脑反应有多强"的信息。换成固定标定值后，恒定图
   0.390、真实视频 0.519，幅度重新有意义。
2. **混合两个分量**：偏离慢基线的部分（抽搐） + 变化率（抖动）。
3. **画面变化门控** —— 画面没变时（本帧与上帧驱动数组完全相同）把输出压到 0。
   这一条是必需的：静止画面虽然水平不变，但恒定速率的泊松驱动仍让网络持续
   随机波动（实测波动幅度是真实视频的 50~70%），只靠前两条压不下去。门控后
   静止画面 |v| = 0.000，真实视频 0.491（观感不变）。

慢基线同时兼任"它现在整体有多活跃"，所以视频节奏快的时候基线抬升。
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

    def update(self, activity: np.ndarray, dt: float,
               motion: float | None = None) -> dict[str, float]:
        cfg = self.cfg
        out: dict[str, float] = {}
        tau = max(1e-3, float(cfg.baseline_tau))
        alpha = min(1.0, dt / tau)

        # 画面变化门控：静止画面（motion=0）-> 身体完全不动
        gate = 1.0
        lo, hi = float(cfg.motion_lo), float(cfg.motion_hi)
        if motion is not None and hi > lo:
            gate = float(np.clip((float(motion) - lo) / (hi - lo), 0.0, 1.0))

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

            # 慢基线（姿态），慢尺度只留作观测，不再当除数
            st.base += (r - st.base) * alpha
            st.scale += (abs(r - st.base) - st.scale) * alpha
            ref = float(cfg.dev_ref.get(k) or 0.0)
            if ref <= 0.0:
                ref = max(1e-4, st.scale)      # 未标定的部位退回自适应

            # 偏离基线 = 抽搐；除以固定参考值，幅度才反映脑反应强度
            dev = (r - st.base) / max(1e-4, ref)
            dev = float(np.clip(dev, -3.0, 3.0)) / 3.0

            # 变化率 = 抖动；参考尺度用 jitter_ref（原来那个 1e3 把这一项压没了）
            d = (r - st.prev) / max(1e-4, dt)
            st.prev = r
            jref = max(1e-6, float(cfg.jitter_ref))
            d = float(np.clip(d, -3.0 * jref, 3.0 * jref)) / (3.0 * jref)

            st.last = r
            v = float(cfg.twitch_gain) * dev + float(cfg.jitter_gain) * d
            out[k] = float(np.clip(v * gate, -1.0, 1.0))
        return out
