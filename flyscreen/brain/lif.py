"""全脑 LIF 脉冲网络。

模型照 Shiu et al. 2024（*Nature* 634:210）的参考实现：

    tau_m dv/dt = (v_rest - v) + g          tau_m = 20 ms
    tau_s dg/dt = -g                        tau_s = 5 ms
    每个脉冲经过 1.8 ms 延迟后，给每个突触后神经元注入
        g_i += sign_j * N_ji * 0.275 mV * gain
    v > -45 mV  ->  发放；v = -52 mV，g = 0，不应期 2.2 ms

积分用精确线性解：

    v(t+dt) = v_rest + (v-v_rest)*exp(-dt/tm)
              + g0*(ts/(ts-tm))*(exp(-dt/ts)-exp(-dt/tm))
    g(t+dt) = g0*exp(-dt/ts)

参考实现在 male-cns 上把全局增益标定到 0.65（每个突触 0.179 mV）——雌脑的
0.275 mV 会把这个更密的图直接推成癫痫。

**视频驱动用泊松脉冲发生器**，不用注入电流。原因：注入电流会撞上阈值悬崖——
稳态 g 一越过 7 mV，被驱动的神经元就以 133 Hz 连续发放，几千个神经元一起灌，
整个网络瞬间饱和，视频图案完全消失。泊松率是平滑可控的，与参考实现一致。

关键性质：**没有自发活动**。静止的脑永远静止，每个脉冲都能追溯到外部输入。

性能：装了 numba 走 JIT 路径（整个 step 一次循环，无临时数组）。
本项目实测 141,781 神经元 / 5.5M 突触约 1.3 ms/步；纯 numpy 回退约 8.4 ms/步。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # numba 是可选加速
    from numba import njit as _njit

    HAS_NUMBA = True
except Exception:  # noqa: BLE001
    HAS_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]):
            return args[0]

        def _deco(f):
            return f

        return _deco


@dataclass
class LifConfig:
    dt_ms: float = 1.0
    tau_m: float = 20.0
    tau_s: float = 5.0
    v_rest: float = -52.0
    v_reset: float = -52.0
    v_thresh: float = -45.0
    t_refract: float = 2.2
    delay_ms: float = 1.8
    mv_per_synapse: float = 0.275
    gain: float = 0.65

    # E/I 平衡校正。MaleCNS 的原始总突触权重里，兴奋比抑制强 1.53 倍，
    # 网络天生超临界 —— 任何一点输入都会自持地烧下去，视频图案被完全冲散。
    # 打开后把抑制权重整体放大，使总兴奋权重 = 总抑制权重。
    # 这是本次实现唯一在参考模型之外加的修正，会打印实际倍率。
    ei_balance: bool = True

    # 视频驱动：泊松脉冲发生器。drive∈[0,1] 映射成 max_rate_hz 以内的放电率。
    max_rate_hz: float = 80.0
    drive_threshold: float = 0.01

    # 可选：所有神经元的高斯噪声（mV/步）。默认 0，保持确定性。
    noise_mv: float = 0.0


@_njit(cache=True, fastmath=True)
def _step_numba(
    v, g, refr, delay_buf, dp, indptr, indices, wsyn,
    drv_idx, drv_p, drv_rand, use_drive,
    em, es, k_coef, v_rest, v_reset, v_thresh, refr_steps,
    spikes, spk, in_spk,
):
    n = v.shape[0]
    dsteps = delay_buf.shape[0]

    if refr_steps > 0:
        for i in range(n):
            r = refr[i]
            if r > 0:
                refr[i] = r - 1

    # 1) 积分，并把本 tick 的输入消费掉
    for i in range(n):
        a = delay_buf[dp, i]
        delay_buf[dp, i] = 0.0
        g0 = g[i] + a
        if refr[i] == 0:
            v[i] = v_rest + (v[i] - v_rest) * em + g0 * k_coef
        else:
            v[i] = v_reset
        g[i] = g0 * es

    # 2) 网络自身的阈值发放
    nsp = 0
    for i in range(n):
        s = v[i] > v_thresh
        spikes[i] = s
        if s:
            spk[nsp] = i
            nsp += 1
            v[i] = v_reset
            g[i] = 0.0
            if refr_steps > 0:
                refr[i] = refr_steps

    # 3) 被视频驱动的神经元按泊松率发放（旁路 LIF）
    n_in = 0
    if use_drive:
        for k in range(drv_idx.shape[0]):
            if drv_rand[k] < drv_p[k]:
                in_spk[n_in] = drv_idx[k]
                n_in += 1

    # 4) 所有脉冲散射回延迟环的同一格，delay_steps 步后重新被读到
    for k in range(nsp):
        j = spk[k]
        for e in range(indptr[j], indptr[j + 1]):
            delay_buf[dp, indices[e]] += wsyn[e]
    for k in range(n_in):
        j = in_spk[k]
        for e in range(indptr[j], indptr[j + 1]):
            delay_buf[dp, indices[e]] += wsyn[e]

    dnext = dp + 1
    if dnext >= dsteps:
        dnext = 0
    return nsp, n_in, dnext


def _gather(indptr: np.ndarray, indices: np.ndarray, wsyn: np.ndarray, rows: np.ndarray):
    """把若干"突触前神经元"的所有出边摊平取出来（ragged gather）。"""
    if rows.size == 0:
        return indices[:0], wsyn[:0]
    counts = indptr[rows + 1] - indptr[rows]
    total = int(counts.sum())
    if total == 0:
        return indices[:0], wsyn[:0]
    starts = indptr[rows]
    cum = np.cumsum(counts)
    inner = np.arange(total, dtype=np.int64) - np.repeat(cum - counts, counts)
    pos = np.repeat(starts, counts) + inner
    return indices[pos], wsyn[pos]


class LifNetwork:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        wsyn: np.ndarray,
        n: int,
        cfg: LifConfig,
        seed: int = 1234,
    ) -> None:
        self.cfg = cfg
        self.n = int(n)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        w = np.ascontiguousarray(wsyn, dtype=np.float32)

        # E/I 平衡：把抑制侧整体放大到与兴奋侧总权重相等
        self.ei_scale = 1.0
        if cfg.ei_balance and w.size:
            pos = float(w[w > 0].sum())
            neg = float(-w[w < 0].sum())
            if neg > 0 and pos > 0:
                self.ei_scale = pos / neg
                w = np.where(w < 0, w * np.float32(self.ei_scale), w).astype(np.float32)

        self.wsyn = w * np.float32(cfg.gain)
        self.rng = np.random.default_rng(seed)
        self.backend = "numba" if HAS_NUMBA else "numpy"

        dt, tm, ts = cfg.dt_ms, cfg.tau_m, cfg.tau_s
        if abs(tm - ts) < 1e-9:
            raise ValueError("tau_m 与 tau_s 不能相等")
        self.em = float(np.exp(-dt / tm))
        self.es = float(np.exp(-dt / ts))
        self.k_coef = float((ts / (ts - tm)) * (self.es - self.em))
        self.refr_steps = int(max(0, round(cfg.t_refract / dt)))
        self.delay_steps = int(max(1, round(cfg.delay_ms / dt)))
        self.rate_scale = float(cfg.max_rate_hz * dt / 1000.0)

        self.v = np.full(self.n, cfg.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.int16)
        self.delay_buf = np.zeros((self.delay_steps, self.n), dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self._spk = np.zeros(self.n, dtype=np.int64)
        self._in_spk = np.zeros(self.n, dtype=np.int64)

        # 驱动集合（每帧用 set_drive 重建一次）
        self._drv_idx = np.zeros(0, dtype=np.int64)
        self._drv_p = np.zeros(0, dtype=np.float32)
        self._drv_rand = np.zeros(0, dtype=np.float32)
        self._drive_ready = False

        # numpy 回退路径的暂存
        self._tmp = np.zeros(self.n, dtype=np.float32)
        self.delay_pos = 0
        self.t = 0
        self.last_input_spikes = 0

    # ------------------------------------------------------------ 驱动
    def set_drive(self, drive: np.ndarray | None) -> None:
        """每个视频帧调用一次：预计算被驱动的神经元及其每 tick 发放概率。"""
        if drive is None:
            self._drive_ready = False
            self._drv_idx = np.zeros(0, dtype=np.int64)
            self._drv_p = np.zeros(0, dtype=np.float32)
            return
        d = np.asarray(drive, dtype=np.float32).ravel()
        if d.size != self.n:
            raise ValueError(f"drive 长度 {d.size} 与神经元数 {self.n} 不符")
        idx = np.flatnonzero(d > self.cfg.drive_threshold)
        self._drv_idx = idx.astype(np.int64)
        self._drv_p = np.clip(d[idx] * self.rate_scale, 0.0, 0.9).astype(np.float32)
        self._drv_rand = np.zeros(idx.size, dtype=np.float32)
        self._drive_ready = idx.size > 0

    # ------------------------------------------------------------ 推进
    def step(self, drive: np.ndarray | None = None) -> np.ndarray:
        """推进一个 dt，返回本步**网络自身**的发放掩码。

        被视频驱动的泊松脉冲不计入返回值（它们由渲染的冷色层单独表示），
        其数量记在 self.last_input_spikes。
        """
        drive_given = drive is not None
        if drive_given:
            self.set_drive(drive)

        if self.backend == "numba":
            return self._step_jit()

        # numpy 回退：drive 直接给出时也走泊松（与 JIT 路径一致）
        if drive_given and not self._drive_ready:
            pass
        return self._step_numpy()

    def _step_jit(self) -> np.ndarray:
        use = self._drive_ready
        if use:
            self._drv_rand = self.rng.random(
                self._drv_idx.size, dtype=np.float32
            ).astype(np.float32)
        nsp, n_in, self.delay_pos = _step_numba(
            self.v, self.g, self.refr, self.delay_buf, self.delay_pos,
            self.indptr, self.indices, self.wsyn,
            self._drv_idx, self._drv_p, self._drv_rand, use,
            self.em, self.es, self.k_coef,
            self.cfg.v_rest, self.cfg.v_reset, self.cfg.v_thresh,
            self.refr_steps,
            self.spikes, self._spk, self._in_spk,
        )
        self.last_input_spikes = int(n_in)
        if self.cfg.noise_mv > 0:
            self.v += self.rng.normal(0.0, self.cfg.noise_mv, size=self.n).astype(np.float32)
        self.t += 1
        return self.spikes

    def _step_numpy(self) -> np.ndarray:
        """纯 numpy 回退（无 numba 时使用）。动力学与 JIT 路径一致。"""
        cfg = self.cfg
        n = self.n
        dp = self.delay_pos

        if self.refr_steps > 0:
            self.refr -= 1
            np.maximum(self.refr, 0, out=self.refr)

        arriving = self.delay_buf[dp]
        tmp = self._tmp
        np.copyto(tmp, arriving)
        self.delay_buf[dp] = 0.0
        if cfg.noise_mv > 0:
            tmp += self.rng.normal(0.0, cfg.noise_mv, size=n).astype(np.float32)

        v = self.v
        np.add(self.g, tmp, out=tmp)           # tmp = g0
        np.subtract(v, cfg.v_rest, out=v)
        np.multiply(v, self.em, out=v)
        np.multiply(tmp, self.k_coef, out=self.g)  # 借用 g 当暂存
        np.add(v, self.g, out=v)
        np.add(v, cfg.v_rest, out=v)
        np.multiply(tmp, self.es, out=self.g)  # g = g0*es

        if self.refr_steps > 0:
            np.copyto(v, cfg.v_reset, where=self.refr > 0)

        spiked = v > cfg.v_thresh
        idx = np.flatnonzero(spiked)
        if idx.size:
            v[idx] = cfg.v_reset
            self.g[idx] = 0.0
            if self.refr_steps > 0:
                self.refr[idx] = self.refr_steps

        n_in = 0
        inp = idx[:0]
        if self._drive_ready and self._drv_idx.size:
            r = self.rng.random(self._drv_idx.size)
            mask = r < self._drv_p
            inp = self._drv_idx[mask]
            n_in = int(inp.size)

        rows = np.concatenate([idx, inp]) if inp.size else idx
        if rows.size:
            cols, ws = _gather(self.indptr, self.indices, self.wsyn, rows)
            if cols.size:
                contrib = np.bincount(cols, weights=ws, minlength=n)
                self.delay_buf[dp] += contrib.astype(np.float32)

        self.delay_pos = (dp + 1) % self.delay_steps
        self.last_input_spikes = n_in
        self.t += 1
        self.spikes = spiked
        return spiked

    # ------------------------------------------------------------ 辅助
    def reset(self) -> None:
        self.v.fill(self.cfg.v_rest)
        self.g.fill(0.0)
        self.refr.fill(0)
        self.delay_buf.fill(0.0)
        self.delay_pos = 0
        self.spikes.fill(False)
        self.t = 0
        self.last_input_spikes = 0

    def stats(self) -> dict:
        return {
            "t": self.t,
            "spikes": int(self.spikes.sum()),
            "input_spikes": self.last_input_spikes,
            "v_mean": float(self.v.mean()),
            "g_mean": float(self.g.mean()),
        }
