"""MuJoCo 真果蝇身体。

模型：flybody（Google DeepMind + HHMI Janelia 的解剖级果蝇身体，Apache-2.0），
网格命名用 T1/T2/T3 表示前/中/后腿，和 MaleCNS 连接组的神经毡命名一致。

两种模式
--------
* **地面模式（默认）** —— 真重力（模型自带 -981，单位是厘米）、地面接触、摩擦。
  果蝇站在地上，被神经驱动推着走/挣扎/翻倒，每一步都由 MuJoCo 真算。
* **悬空模式**（`free_floating=True`）—— 关掉重力、冻结根部自由度，
  身体浮在空中只有腿翅在动。想避开地面物理时用。

模型本来是按「站在地面上」建的：原版 scene.xml 的地面在 z = -0.132，
而实测六只跗爪（脚）的最低点在 z = -0.1319，正好吻合。

关节名不做硬编码，运行时按关键字匹配，并把匹配结果打印出来。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import BodyConfig

# 每个部位的关键字：必须同时命中的 "T1..T3 / wing" 与 "left|right"
PART_SPEC: dict[str, tuple[tuple[str, ...], str]] = {
    "leg_front_left": (("t1",), "left"),
    "leg_front_right": (("t1",), "right"),
    "leg_mid_left": (("t2",), "left"),
    "leg_mid_right": (("t2",), "right"),
    "leg_hind_left": (("t3",), "left"),
    "leg_hind_right": (("t3",), "right"),
    "wing_left": (("wing",), "left"),
    "wing_right": (("wing",), "right"),
}


def _side_of(name: str) -> str | None:
    n = name.lower()
    if re.search(r"(^|[_\-.])l($|[_\-.])", n) or "left" in n:
        return "left"
    if re.search(r"(^|[_\-.])r($|[_\-.])", n) or "right" in n:
        return "right"
    return None


class FlyBody:
    def __init__(self, xml_path: str | Path, cfg: BodyConfig, quiet: bool = False) -> None:
        import mujoco

        self.mj = mujoco
        self.cfg = cfg
        self.quiet = quiet
        self.xml_path = str(xml_path)
        if not Path(self.xml_path).exists():
            raise FileNotFoundError(f"找不到 flybody MJCF: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.nu = int(self.model.nu)
        self.act_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
            for i in range(self.nu)
        ]
        self.jnt_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"jnt{i}"
            for i in range(self.model.njnt)
        ]

        if cfg.free_floating:
            self.model.opt.gravity[:] = 0.0
        if cfg.physics_timestep > 0:
            self.model.opt.timestep = float(cfg.physics_timestep)
        mode = "悬空" if cfg.free_floating else "地面（真重力 + 接触）"
        self._log(
            f"[body] 物理步长 {self.model.opt.timestep*1000:.3f} ms  模式：{mode}"
            f"  重力 {self.model.opt.gravity[2]:.0f}"
        )

        self._find_root()
        self._index_parts()

        # 记下静止姿态（姿态辅助的目标），必须在落地之前抓
        mujoco.mj_forward(self.model, self.data)
        self._qpos_rest = self.data.qpos.copy()
        self._index_posture()
        self._index_lights()

        # 落地：让物理自己把果蝇放到地面上
        if not cfg.free_floating and cfg.settle_steps > 0:
            for _ in range(int(cfg.settle_steps)):
                mujoco.mj_step(self.model, self.data)
            z = self.root_height()
            self._log(
                f"[body] 落地 {cfg.settle_steps} 步后 根部 z = {z:.5f}  "
                f"接触点 {int(self.data.ncon)} 个"
            )
        else:
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)

        self._setup_camera()

        self._ctrl_range = (
            np.array(self.model.actuator_ctrlrange, dtype=np.float32)
            if self.model.nu
            else np.zeros((0, 2), dtype=np.float32)
        )
        # 正/负方向各自还有多少行程（0 = 静止位，所以方向不同余量也不同）
        lo, hi = self._ctrl_range[:, 0], self._ctrl_range[:, 1]
        finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        self._amp_pos = np.maximum(np.where(finite, hi, 1.0), 0.0).astype(np.float32)
        self._amp_neg = np.maximum(np.where(finite, -lo, 1.0), 0.0).astype(np.float32)

        self._renderer = None
        self._rw = self._rh = 0
        self._smooth = np.zeros(self.nu, dtype=np.float32)
        self._last_drive = np.zeros(self.nu, dtype=np.float32)

    # ------------------------------------------------------------ 日志
    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    # ------------------------------------------------------------ 内部
    def _find_root(self) -> None:
        mj = self.mj
        self.root_qpos_adr = -1
        self.root_qvel_adr = -1
        self.root_qpos0 = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                self.root_qpos_adr = int(self.model.jnt_qposadr[j])
                self.root_qvel_adr = int(self.model.jnt_dofadr[j])
                self.root_qpos0 = self.data.qpos[
                    self.root_qpos_adr : self.root_qpos_adr + 7
                ].copy()
                break
        if self.cfg.free_floating and self.root_qpos_adr >= 0:
            self._log("[body] 根部自由关节已冻结（悬空模式）")

    def _index_posture(self) -> None:
        """把 actuator 映射到关节，供姿态辅助使用。"""
        mj = self.mj
        self._act_qadr = np.full(self.nu, -1, dtype=np.int32)
        self._act_dof = np.full(self.nu, -1, dtype=np.int32)
        self._qpos_target = np.zeros(self.nu, dtype=np.float32)
        trn_joint = int(mj.mjtTrn.mjTRN_JOINT)
        for i in range(self.nu):
            if int(self.model.actuator_trntype[i]) != trn_joint:
                continue
            jid = int(self.model.actuator_trnid[i, 0])
            qa = int(self.model.jnt_qposadr[jid])
            self._act_qadr[i] = qa
            self._act_dof[i] = int(self.model.jnt_dofadr[jid])
            self._qpos_target[i] = float(self._qpos_rest[qa])

    def _index_lights(self) -> None:
        """记下自己加的那几盏灯相对果蝇根部的偏移。

        灯如果固定在世界坐标，果蝇被抽得满地打滚跑远之后，脚下那片地板就
        照不到了 —— 实测跑出 6 cm 之后地面几乎全黑。所以每帧把灯跟着搬过去。
        """
        self._light_idx: list[int] = []
        self._light_off: list[np.ndarray] = []
        root = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] \
            if self.root_qpos_adr >= 0 else np.zeros(3)
        for i in range(self.model.nlight):
            nm = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_LIGHT, i)
            if nm in ("key", "fill", "rim"):
                self._light_idx.append(i)
                self._light_off.append(
                    (self.model.light_pos[i] - root).astype(np.float64))
        self._lights_follow = bool(self.cfg.lights_follow and self._light_idx)
        if self._light_idx:
            self._log(f"[body] 跟随灯光 {len(self._light_idx)} 盏 "
                      f"({'开' if self._lights_follow else '关'}）")

    def _move_lights(self) -> None:
        if not self._lights_follow or self.root_qpos_adr < 0:
            return
        root = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3]
        for i, off in zip(self._light_idx, self._light_off):
            self.model.light_pos[i] = root + off

    def _index_parts(self) -> None:
        """把执行器分到 8 个部位。先拆左右，再按 T1/T2/T3 / wing 归类。"""
        buckets: dict[str, list[int]] = {k: [] for k in PART_SPEC}
        leftover: list[str] = []

        for i, name in enumerate(self.act_names):
            low = name.lower()
            side = _side_of(low)
            if side is None:
                leftover.append(name)
                continue
            matched = False
            for part, (keys, want_side) in PART_SPEC.items():
                if want_side != side:
                    continue
                if any(k in low for k in keys):
                    if keys == ("wing",) and "wing" not in low:
                        continue
                    buckets[part].append(i)
                    matched = True
                    break
            if not matched:
                leftover.append(name)

        self.parts = {k: np.asarray(v, dtype=np.int32) for k, v in buckets.items()}
        self._log(f"[body] MJCF: {Path(self.xml_path).name}")
        self._log(
            f"[body] {self.model.nbody} bodies / {self.model.njnt} joints / {self.nu} actuators"
        )
        for part, idx in self.parts.items():
            ex = ", ".join(self.act_names[i] for i in idx[:3])
            more = f" ...(+{len(idx)-3})" if len(idx) > 3 else ""
            self._log(f"[body]   {part:17s} {len(idx):>4d} actuators   {ex}{more}")
        if leftover:
            self._log(f"[body]   未归类 {len(leftover)} 个：{leftover[:6]}")

    def _setup_camera(self) -> None:
        mj = self.mj
        mj.mj_forward(self.model, self.data)
        self.cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.cam)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == 0:  # world / floor
                continue
            center = self.data.geom_xpos[g]
            r = float(self.model.geom_rbound[g])
            lo = np.minimum(lo, center - r)
            hi = np.maximum(hi, center + r)
        if not np.isfinite(lo).all():
            lo, hi = np.array([-1.0] * 3), np.array([1.0] * 3)
        self.bbox_lo, self.bbox_hi = lo, hi
        self.cam.lookat[:] = (lo + hi) / 2.0
        self.cam.distance = float(np.linalg.norm(hi - lo)) * float(
            self.cfg.camera_distance_scale
        )
        if self.root_qpos_adr >= 0:
            root = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3]
            self._cam_offset = (lo + hi) / 2.0 - root
        else:
            self._cam_offset = np.zeros(3)
        self._cam_z0 = float(self.cam.lookat[2])
        if self.cfg.free_floating:
            self.cam.azimuth = 128.0
            self.cam.elevation = -14.0
        else:
            self.cam.azimuth = 132.0
            self.cam.elevation = float(self.cfg.camera_elevation)
        self._log(
            f"[body] 相机 lookat={np.round(self.cam.lookat,3)} "
            f"dist={self.cam.distance:.3f}  包围盒={np.round(hi-lo,3)}"
        )

    def _track_camera(self) -> None:
        """让相机跟着果蝇走，否则它翻滚出去就跑出画面了。

        **竖直方向不跟**。果蝇被抽起来时 root z 能窜到 0.38（站立时 −0.006），
        相机要是一起抬上去，脚下那片地板就整个掉出视野 —— 表现为"场景里没有地板"。
        所以只跟水平方向和四周吸附，高度钉死在初始水平。
        """
        if self.root_qpos_adr < 0 or not self.cfg.camera_track:
            return
        root = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3]
        want = root + self._cam_offset
        if not self.cfg.camera_track_z:
            want[2] = self._cam_z0
        d = float(np.linalg.norm(want - self.cam.lookat))
        if d > float(self.cfg.camera_snap_dist):
            self.cam.lookat[:] = want          # 太远了，直接跟过去
            return
        a = float(np.clip(self.cfg.camera_track_smooth, 0.0, 1.0))
        self.cam.lookat[:] = self.cam.lookat * (1.0 - a) + want * a

    # ------------------------------------------------------------ 驱动
    def step(self, drive: dict[str, float], duration: float) -> None:
        """按部位驱动，推进 duration 秒的物理。drive 取值 [-1,1]。"""
        self._apply_ctrl(drive, duration)
        mj = self.mj
        ts = float(self.model.opt.timestep)
        n = int(max(1, round(float(duration) / ts)))
        n = min(n, int(self.cfg.max_substeps))
        for _ in range(n):
            mj.mj_step(self.model, self.data)
            if self.cfg.free_floating:
                self._freeze_root()

    def _apply_ctrl(self, drive: dict[str, float], dt: float) -> None:
        """drive 取值 [-1,1]，0 = 静止位。按各执行器正负方向的余量缩放。"""
        target = np.zeros(self.nu, dtype=np.float32)
        travel = float(self.cfg.travel)
        for part in PART_SPEC:
            idx = self.parts.get(part)
            if idx is None or idx.size == 0:
                continue
            v = float(np.clip(drive.get(part, 0.0), -1.0, 1.0))
            if v == 0.0:
                continue
            amp = np.where(v > 0.0, self._amp_pos[idx], self._amp_neg[idx])
            target[idx] = v * amp * travel * float(self.cfg.part_gain)

        # 姿态辅助：把关节往静止姿态拉，相当于给果蝇一点肌肉张力。
        # 0 = 关（纯物理）。注意调大会把仿真推爆，实测 0.3 就不稳。
        kp = float(self.cfg.posture_gain)
        if kp > 0.0 and self.nu:
            kd = float(self.cfg.posture_kd)
            valid = self._act_qadr >= 0
            if valid.any():
                q = self.data.qpos[self._act_qadr[valid]]
                v_ = self.data.qvel[self._act_dof[valid]]
                target[valid] += kp * (self._qpos_target[valid] - q) - kd * v_

        tau = max(1e-4, float(self.cfg.smoothing_tau))
        alpha = 1.0 - float(np.exp(-max(1e-6, dt) / tau))
        self._smooth += alpha * (target - self._smooth)
        self._last_drive = self._smooth.copy()

        if self.nu:
            lo = self._ctrl_range[:, 0]
            hi = self._ctrl_range[:, 1]
            finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo - 1e-9)
            self.data.ctrl[:] = np.where(finite, np.clip(self._smooth, lo, hi), self._smooth)

    def _freeze_root(self) -> None:
        if self.root_qpos_adr >= 0 and self.root_qpos0 is not None:
            self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 7] = self.root_qpos0
        if self.root_qvel_adr >= 0:
            self.data.qvel[self.root_qvel_adr : self.root_qvel_adr + 6] = 0.0

    # ------------------------------------------------------------ 状态查询
    def contacts(self) -> tuple[int, float]:
        """当前接触点数量，以及它们沿接触法向的力总和。"""
        mj = self.mj
        n = int(self.data.ncon)
        if n == 0:
            return 0, 0.0
        f = np.zeros(6, dtype=np.float64)
        total = 0.0
        for i in range(n):
            mj.mj_contactForce(self.model, self.data, i, f)
            total += abs(float(f[0]))
        return n, total

    def root_height(self) -> float:
        if self.root_qpos_adr < 0:
            return 0.0
        return float(self.data.qpos[self.root_qpos_adr + 2])

    def uprightness(self) -> float:
        """+1 = 正立，0 = 侧躺，-1 = 四脚朝天。"""
        if self.root_qpos_adr < 0:
            return 1.0
        q = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 7]
        return float(1.0 - 2.0 * (float(q[4]) ** 2 + float(q[5]) ** 2))

    def root_position(self) -> np.ndarray:
        if self.root_qpos_adr < 0:
            return np.zeros(3)
        return self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3].copy()

    def drive_snapshot(self) -> np.ndarray:
        return self._last_drive.copy()

    # ------------------------------------------------------------ 渲染
    def render(self, width: int, height: int, camera: str = "free") -> np.ndarray:
        mj = self.mj
        self._move_lights()
        self._track_camera()
        if self._renderer is None or (self._rw, self._rh) != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mj.Renderer(self.model, height=int(height), width=int(width))
            self._rw, self._rh = int(width), int(height)
        mj.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self.cam)
        return self._renderer.render().copy()

    # ------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        mj = self.mj
        mj.mj_resetData(self.model, self.data)
        self._smooth[:] = 0.0
        if self.cfg.free_floating:
            self._freeze_root()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def find_model_xml(root: str | Path) -> Path | None:
    """在目录里找 flybody 的主 MJCF。优先我们自己套的场景文件。"""
    root = Path(root)
    if not root.exists():
        return None
    for name in ("flyscene.xml", "scene.xml", "flybody.xml", "fruitfly.xml"):
        p = root / name
        if p.exists():
            return p
    cands = sorted(root.glob("*.xml"))
    return cands[0] if cands else None
