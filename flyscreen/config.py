"""全局配置：路径、数据源、画布/渲染/身体参数。

数据源固定为 MaleCNS v1.0（HHMI Janelia FlyEM + University of Cambridge +
MRC LMB + Google Research），许可 CC BY 4.0。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # D:\fly-brain\flyscreen
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = PROJECT_ROOT / "out"

for _d in (RAW_DIR, CACHE_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据源
MALECNS_BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
MALECNS_FILES = {
    # key: (远端文件名, 本地文件名)
    "annotations": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "body-annotations.feather",
    ),
    "connectome_weights": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "connectome-weights.feather",
    ),
    "syn_partners": (
        "syn-partners-male-cns-v1.0-minconf-0.5.feather",
        "syn-partners.feather",
    ),
}

# 体素边长（纳米）。MaleCNS 为 8nm 各向同性。
VOXEL_NM = 8.0
NM_PER_UM = 1000.0


# ---------------------------------------------------------------- 画布
@dataclass
class CanvasConfig:
    """视频帧 -> 神经元 的映射参数。"""

    # 投影到 2D 时使用的两个解剖轴（0=x, 1=y, 2=z）。
    # view_name 只是给界面用的名字，"xy" 等价于 view_axes=(0,1)。
    view_name: str = "xy"
    view_axes: tuple[int, int] = (0, 1)

    # 切掉两端各 1% 的离群点，让点云铺满画布
    clip_percentile: float = 1.0

    # 每个神经元在画布上的"可见权重"。
    # 视叶神经元极密（约 10 万个挤在两坨里），不做均衡的话画面会变成两块白斑。
    #   "none"  -> 不均衡（原始解剖密度）
    #   "sqrt"  -> 按 1/sqrt(密度) 缩放（推荐，保留一些体积感）
    #   "full"  -> 按 1/密度 缩放（最均匀）
    density_equalize: str = "sqrt"
    density_grid: int = 220  # 统计密度用的网格分辨率

    # 荧光屏拖尾：每帧先衰减，再取 max(衰减后, 当前驱动)
    decay: float = 0.72
    # 低于该强度的活动直接抹掉，避免长期残留噪声
    floor: float = 0.004

    # 视频亮度 -> 驱动强度的映射
    #   black_point  低于该亮度的直接算黑（0 = 不裁）
    #   white_point  高于该亮度的算全亮（1 = 不裁）
    #   gamma        <1 提亮暗部；>1 压暗部（实拍视频建议 2 左右，否则整个脑一起亮）
    black_point: float = 0.0
    white_point: float = 1.0
    gamma: float = 0.85
    gain: float = 1.0


# ---------------------------------------------------------------- 渲染
@dataclass
class RenderConfig:
    width: int = 1280
    height: int = 720

    # 画布区域（点云）在窗口里占的像素尺寸；其余留给身体画面
    # 放大后的默认值。关键指标是**点云密度**（141,781 个点 / 画布像素数）：
    #   600x490  -> 0.482 点/像素   点严重重叠，糊成一团
    #   1000x820 -> 0.173 点/像素   点基本分离，清晰
    # 顺带把输出从 1102x518 抬到 1742x848，全屏播放不用上采样 1.74 倍。
    # 代价：渲染像素约 2.8 倍，fps 会掉一档。
    canvas_w: int = 1000
    canvas_h: int = 820

    # 点半径（像素）与亮度增益
    # 注意：加了 fade 修正后，驱动层/脉冲层在 v=0 处归于全黑（之前会被 lo 染色），
    # 所以这两个增益比修正前调高了约 1.6 倍。
    point_size: int = 1
    structure_gain: float = 0.45   # 未点亮神经元的基础亮度（保留脑结构）
    activity_gain: float = 3.4     # 脉冲层（LIF 传播出来的活动）

    # 辉光：分离式盒式模糊次数，每次等效半径 glow_radius
    glow_radius: int = 4
    glow_passes: int = 2
    glow_mix: float = 0.22  # 模糊层叠加比例

    background: tuple[int, int, int] = (4, 6, 12)

    # 三层配色
    #   结构层（未点亮）—— 暗蓝，负责"看出这是果蝇脑"
    #   驱动层（视频直接点亮）—— 冷色，"你给它的"
    #   脉冲层（连接组传播出来的）—— 暖色，"它自己的反应"
    structure_color: tuple[int, int, int] = (26, 54, 92)
    drive_color_lo: tuple[int, int, int] = (8, 44, 70)
    drive_color_hi: tuple[int, int, int] = (170, 250, 255)
    drive_gain: float = 1.75
    spike_color_lo: tuple[int, int, int] = (60, 18, 0)
    spike_color_hi: tuple[int, int, int] = (245, 140, 40)
    glow_color: tuple[int, int, int] = (50, 110, 165)


# ---------------------------------------------------------------- 身体
@dataclass
class BodyConfig:
    """MuJoCo 真果蝇身体（NeuroMechFly v2 / flybody）。"""

    enabled: bool = True
    width: int = 700
    height: int = 820

    # 8 个驱动部位：6 条腿 + 2 只翅膀
    parts: tuple[str, ...] = (
        "leg_front_left",
        "leg_front_right",
        "leg_mid_left",
        "leg_mid_right",
        "leg_hind_left",
        "leg_hind_right",
        "wing_left",
        "wing_right",
    )

    # 每个部位的驱动增益与相位（避免 8 个部位整齐划一地抽）
    part_gain: float = 1.0
    # 关节力矩上限
    torque_limit: float = 1.0

    # ---- 驱动信号整形（见 body/driver.py 的说明）----
    # 慢基线时间常数（秒）：决定"姿势"跟随视频多快，以及抽搐围绕什么中心
    baseline_tau: float = 0.5
    # 抽搐分量：偏离自身慢基线的部分（这是"抽"的主要来源）
    twitch_gain: float = 1.4
    # 抖动分量：活动变化率
    jitter_gain: float = 0.9
    # 驱动信号的平滑时间常数（秒）。太大就变"摆姿势"，太小就像触电
    smoothing_tau: float = 0.008
    # 把 [-1,1] 驱动映射到关节可用行程的比例（1.0 = 用满 ctrlrange）
    travel: float = 1.0

    # 悬空模式：关掉重力并冻结根部，果蝇浮在空中只有腿翅在动。
    # 关掉（默认）就是**地面模式** —— 真重力、真接触、真摩擦，
    # MuJoCo 每一步都算支撑和落地。
    free_floating: bool = False
    # 落地前的稳定步数（模型 1 步 = 0.1ms，8000 步 = 0.8 秒）
    settle_steps: int = 8000
    # 姿态辅助强度。0 = 纯物理（果蝇会自己倒）；调大像"有肌肉张力"
    posture_gain: float = 0.0
    posture_kd: float = 0.02
    # 相机跟踪：果蝇被驱动时会翻滚移动，不跟就会跑出画面
    camera_track: bool = True
    # 竖直方向不跟：跟上去会把它脚下的地板挤出视野（表现为"场景里没地板"）
    camera_track_z: bool = False
    camera_track_smooth: float = 0.45
    camera_snap_dist: float = 0.04   # 偏离超过这个距离（cm）就直接吸附跟过去
    # 取景倍数。要够宽才能同时容下地板和"飞起来"的过程：
    # 果蝇站立时 root z ≈ -0.006，被抽起来能窜到 +0.4 以上，
    # 地板在 z = -0.134。1.9 倍时视野竖直跨度约 1.1 cm，两头都装得下。
    camera_distance_scale: float = 1.90
    # 视角俯仰。太平会看不见地面，太陡又看不出"飞起来"的高度差
    camera_elevation: float = -19.0
    # 灯光跟着果蝇走。灯固定在世界坐标的话，果蝇滚远了脚下地板就全黑
    lights_follow: bool = True

    # 物理步长与每帧最大子步数。1ms 步长 + 33 子步 ≈ 实时。
    physics_timestep: float = 1e-3
    max_substeps: int = 80


# ---------------------------------------------------------------- 神经动力学
@dataclass
class BrainConfig:
    """全脑 LIF 脉冲网络（Shiu et al. 2024）。"""

    enabled: bool = True

    dt_ms: float = 1.0
    # 全局增益。参考实现在 male-cns 上标定的是 0.65，但那是在"只驱动少量特定
    # 感觉神经元"的前提下。本项目用视频帧广域驱动几千个散布全脑的神经元，
    # 0.65 会让网络自持成一片（实测 25,000 神经元参与、空间相关 0）。
    # 0.30 配合下面的 E/I 平衡，活动降到 ~7,500 且保留空间结构。
    gain: float = 0.30
    # 视频驱动：drive=1 的神经元按该频率做泊松发放（Hz）。
    max_rate_hz: float = 80.0
    # E/I 平衡校正（见 brain/lif.py）。原始连接组兴奋比抑制强 1.53 倍。
    ei_balance: bool = True
    noise_mv: float = 0.0

    # 每个视频帧推进多少神经时间。0 = 自动按 dt 跑满一帧（真实时间）。
    ticks_per_frame: int = 0
    # 每个 tick 的上限，防止 dt 太小时爆掉
    max_ticks_per_frame: int = 120

    # 用真实的运动神经元脉冲驱动身体（否则用画布活动直接驱动）
    body_from_spikes: bool = True


# ---------------------------------------------------------------- 总配置
@dataclass
class Config:
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)

    # 视频解码时缩到该宽度（保持长宽比），0 = 用原始尺寸。
    # 480 太低了 —— 信息在进神经元之前就只剩源的 6.2%，之后画布开多大都补不回来。
    # 960 覆盖到 960x540，实拍视频的细节才留得住。
    video_scale_width: int = 960
    # 预览时的目标帧率
    fps: int = 30
    # 一个视频帧对应多少个仿真 tick（1 = 不做神经仿真，直接点亮）
    ticks_per_frame: int = 1
    # 网页上「开始录制」出片的 x264 CRF，越小越好
    record_crf: int = 18


DEFAULT = Config()
