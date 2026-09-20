"""flyscreen 主程序。

    预览：  python run.py --video D:\\clips\\bad_apple.mp4
    出片：  python run.py --video D:\\clips\\bad_apple.mp4 --out out\\badapple.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from flyscreen import compose
from flyscreen.canvas import NeuronCanvas
from flyscreen.config import DEFAULT, PROJECT_ROOT, Config
from flyscreen.data import dataset
from flyscreen.render import PointCloudRenderer
from flyscreen.video import VideoSource, probe
from flyscreen.writer import VideoWriter

VIEWS = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
    "yx": (1, 0),
    "zx": (2, 0),
    "zy": (2, 1),
}

# 供截图动作使用
_LAST_FRAME: np.ndarray | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flyscreen",
        description="把任意视频播进一只果蝇的脑子里（MaleCNS v1.0 + MuJoCo 果蝇身体）",
    )
    p.add_argument("--video", required=True, help="输入视频（任意 ffmpeg 支持的格式）")
    p.add_argument("--out", default=None, help="输出 mp4；给了就离屏出片")
    p.add_argument("--preview", action="store_true", help="显示预览窗（默认无 --out 时开启）")
    p.add_argument("--seconds", type=float, default=0.0, help="只处理前 N 秒")
    p.add_argument("--fps", type=int, default=None, help="输出/预览帧率")
    p.add_argument("--loop", action="store_true", help="循环播放（预览用）")
    p.add_argument("--view", default="xy", choices=sorted(VIEWS), help="投影轴")
    p.add_argument("--density", default=None, choices=["none", "sqrt", "full"])
    p.add_argument("--decay", type=float, default=None, help="拖尾衰减 0..1")
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None,
                   help="亮度曲线。<1 提亮暗部，>1 压暗部（实拍视频建议 1.8~2.5）")
    p.add_argument("--black-point", type=float, default=None,
                   help="低于该亮度的当黑场（0~1）。实拍视频用它压掉中间调")
    p.add_argument("--white-point", type=float, default=None,
                   help="高于该亮度的当全亮（0~1）")
    p.add_argument("--no-body", action="store_true", help="不加载 MuJoCo 身体")
    p.add_argument("--no-brain", action="store_true",
                   help="关掉 LIF 神经动力学，改成按视频帧直接点亮")
    p.add_argument("--lif-gain", type=float, default=None,
                   help="LIF 全局增益（默认 0.30；参考实现在窄域输入下标定到 0.65，"
                        "本项目的广域视频驱动用 0.65 会自持）")
    p.add_argument("--no-ei-balance", action="store_true",
                   help="关掉 E/I 平衡校正（默认开启；原始图的兴奋比抑制强 1.53 倍）")
    p.add_argument("--brain-dt", type=float, default=None, help="LIF 步长 ms（默认 1.0）")
    p.add_argument("--max-rate", type=float, default=None,
                   help="视频驱动：drive=1 时的泊松发放频率 Hz（默认 80）")
    p.add_argument("--ticks", type=int, default=None, help="每个视频帧推进多少 LIF tick")
    p.add_argument("--brain-noise", type=float, default=None, help="LIF 高斯噪声 mV/步")
    p.add_argument("--glow", type=int, default=None, help="辉光半径，0 = 关闭")
    p.add_argument("--hud", action="store_true", help="在出片里保留文字信息栏")
    p.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True,
                   help="把源视频音轨混进出片（默认开；源视频没音轨会自动跳过）。"
                        "用 --no-audio 关掉")
    p.add_argument("--canvas-size", default=None,
                   help="脑点云面板 WxH（默认用 config.py 的 440x380）")
    p.add_argument("--body-size", default=None,
                   help="果蝇面板 WxH（默认用 config.py 的 620x480）")
    p.add_argument("--body-twitch", type=float, default=None,
                   help="抽搐分量：偏离慢基线的幅度（默认 1.4，调大抽得更凶）")
    p.add_argument("--body-jitter", type=float, default=None,
                   help="抖动量：活动变化率的权重（默认 0.9）")
    p.add_argument("--body-tau", type=float, default=None,
                   help="驱动平滑时间常数 秒（默认 0.008；调到 0.05 就变成摆姿势不动）")
    p.add_argument("--body-travel", type=float, default=None,
                   help="关节行程使用比例（默认 1.0 = 用满；0.3 左右是站着踉跄，"
                        "0.6 以上会翻车）")
    p.add_argument("--body-baseline", type=float, default=None,
                   help="慢基线时间常数 秒（默认 0.5）")
    p.add_argument("--quality", type=int, default=18, help="x264 CRF，越小越好")
    return p.parse_args(argv)


def make_config(a: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.canvas.view_axes = VIEWS[a.view]
    if a.density:
        cfg.canvas.density_equalize = a.density
    if a.decay is not None:
        cfg.canvas.decay = float(a.decay)
    if a.gain is not None:
        cfg.canvas.gain = float(a.gain)
    if a.gamma is not None:
        cfg.canvas.gamma = float(a.gamma)
    if a.black_point is not None:
        cfg.canvas.black_point = float(a.black_point)
    if a.white_point is not None:
        cfg.canvas.white_point = float(a.white_point)
    if a.glow is not None:
        cfg.render.glow_radius = int(a.glow)
    if a.fps:
        cfg.fps = int(a.fps)
    if a.no_body:
        cfg.body.enabled = False
    if a.no_brain:
        cfg.brain.enabled = False
    if a.lif_gain is not None:
        cfg.brain.gain = float(a.lif_gain)
    if a.brain_dt is not None:
        cfg.brain.dt_ms = float(a.brain_dt)
    if a.max_rate is not None:
        cfg.brain.max_rate_hz = float(a.max_rate)
    if a.ticks is not None:
        cfg.brain.ticks_per_frame = int(a.ticks)
    if a.brain_noise is not None:
        cfg.brain.noise_mv = float(a.brain_noise)
    if a.no_ei_balance:
        cfg.brain.ei_balance = False
    if a.canvas_size:
        w, h = a.canvas_size.lower().split("x")
        cfg.render.canvas_w, cfg.render.canvas_h = int(w), int(h)
    if a.body_size:
        w, h = a.body_size.lower().split("x")
        cfg.body.width, cfg.body.height = int(w), int(h)
    if a.body_twitch is not None:
        cfg.body.twitch_gain = float(a.body_twitch)
    if a.body_jitter is not None:
        cfg.body.jitter_gain = float(a.body_jitter)
    if a.body_tau is not None:
        cfg.body.smoothing_tau = float(a.body_tau)
    if a.body_travel is not None:
        cfg.body.travel = float(a.body_travel)
    if a.body_baseline is not None:
        cfg.body.baseline_tau = float(a.body_baseline)
    return cfg


def run(a: argparse.Namespace) -> int:
    cfg = make_config(a)
    video = Path(a.video)
    if not video.exists():
        print(f"!! 找不到视频：{video}", file=sys.stderr)
        return 2

    preview = a.preview or not a.out
    if not preview and not a.out:
        print("!! 既没有 --preview 也没有 --out，没事可做", file=sys.stderr)
        return 2

    # ---------------- 数据 ----------------
    table = dataset.load()
    print(f"[flyscreen] MaleCNS v1.0: {table.n:,} 个神经元")
    ext = table.extent
    print(f"[flyscreen] 包围盒 (µm) = {np.round(ext,1)}")

    canvas = NeuronCanvas(table.pos_um, cfg.canvas)
    renderer = PointCloudRenderer(canvas, cfg.render)
    groups = table.group_dict()
    print(
        f"[flyscreen] 投影轴 {cfg.canvas.view_axes}  密度均衡 {cfg.canvas.density_equalize}"
        f"  拖尾 {cfg.canvas.decay}"
    )

    # ---------------- 神经动力学 ----------------
    brain = None
    if cfg.brain.enabled:
        from flyscreen.brain import LifConfig, LifNetwork

        try:
            indptr, indices, wsyn, n = dataset.load_graph()
            if n != table.n:
                raise RuntimeError(f"图里有 {n} 个神经元，神经元表有 {table.n} 个，请重建缓存")
            lcfg = LifConfig(
                dt_ms=cfg.brain.dt_ms,
                gain=cfg.brain.gain,
                max_rate_hz=cfg.brain.max_rate_hz,
                noise_mv=cfg.brain.noise_mv,
                ei_balance=cfg.brain.ei_balance,
            )
            brain = LifNetwork(indptr, indices, wsyn, n, lcfg)
            print(
                f"[brain] LIF: {n:,} 神经元 / {indices.size:,} 条突触  "
                f"dt={lcfg.dt_ms}ms 增益={lcfg.gain}  E/I 抑制倍率={brain.ei_scale:.3f}  "
                f"延迟={brain.delay_steps} 步  不应期={brain.refr_steps} 步"
                f"  后端={brain.backend}"
            )
        except FileNotFoundError as e:
            print(f"[brain] 没有连接图，退回「直接点亮」模式\n        {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[brain] LIF 初始化失败，退回「直接点亮」模式：{type(e).__name__}: {e}")

    frame_ms = 1000.0 / cfg.fps
    if brain is not None:
        n_ticks = cfg.brain.ticks_per_frame or int(round(frame_ms / cfg.brain.dt_ms))
        n_ticks = max(1, min(n_ticks, cfg.brain.max_ticks_per_frame))
        dpt = canvas.decay_factor(cfg.brain.dt_ms, frame_ms)
        print(f"[brain] 每视频帧 {n_ticks} 个 tick（{n_ticks*cfg.brain.dt_ms:.1f} ms 神经时间）")
    else:
        n_ticks = 0
        dpt = 1.0

    # ---------------- 身体 ----------------
    body = None
    body_driver = None
    prev_drive = None            # 上一帧驱动数组，用于画面变化门控
    if cfg.body.enabled:
        from flyscreen.body import BodyDriver, FlyBody, find_model_xml

        xml = find_model_xml(PROJECT_ROOT / "data" / "assets" / "flybody")
        if xml is None:
            print("[body] 没找到 flybody MJCF，跳过身体。"
                  "先跑：python scripts/fetch_flybody.py")
        else:
            try:
                body = FlyBody(xml, cfg.body)
                body_driver = BodyDriver(groups, cfg.body)
            except Exception as e:  # noqa: BLE001
                print(f"[body] 加载失败，跳过：{type(e).__name__}: {e}")

    # ---------------- 视频 ----------------
    info = probe(video)
    print(
        f"[video] {video.name}  {info.width}x{info.height}  {info.fps:.2f}fps  "
        f"{info.duration:.1f}s"
    )
    src = VideoSource(video, cfg.video_scale_width, loops=a.loop)
    print(f"[video] 解码为 {src.out_width}x{src.out_height} 灰度")

    max_frames = 0
    if a.seconds > 0:
        max_frames = int(round(a.seconds * cfg.fps))

    # ---------------- 输出 ----------------
    w, h = compose.layout_size(cfg)
    if a.hud:
        h += 30
    writer = None
    ui = None
    if a.out:
        writer = VideoWriter(a.out, w, h, cfg.fps, crf=a.quality)
        print(f"[out] 出片 {a.out}  {w}x{h} @ {cfg.fps}fps")
    if preview:
        from flyscreen.ui import Preview

        ui = Preview(cfg)

    # ---------------- 主循环 ----------------
    dt = 1.0 / cfg.fps
    t0 = time.time()
    n = 0
    _last_body = None
    try:
        for frame in src.frames(max_frames=max_frames):
            if ui is not None:
                st = ui.poll()
                if st.quit:
                    break
                for action in ui.take_actions():
                    apply_action(action, cfg, renderer, canvas, table)
                if st.paused and not st.step_once:
                    ui.draw_surface(compose.compose(
                        renderer.render(spikes=canvas.activity), _last_body, cfg))
                    time.sleep(0.03)
                    continue
                st.step_once = False

            drive = canvas.drive_from_frame(frame)
            # 画面变化量：与上一帧驱动数组的平均绝对差（静止画面精确为 0）
            motion = (0.0 if prev_drive is None
                      else float(np.abs(drive - prev_drive).mean()))
            prev_drive = drive
            if brain is not None:
                brain.set_drive(drive)
                total_spikes = 0
                total_input = 0
                for _ in range(n_ticks):
                    sp = brain.step()
                    total_spikes += int(sp.sum())
                    total_input += brain.last_input_spikes
                    canvas.advance_trace(sp, dpt)
                act = canvas.activity
                spike_count = total_spikes + total_input
                canvas_img = renderer.render(spikes=act, drive=drive)
            else:
                act = canvas.advance(drive)
                spike_count = int((act > 0).sum())
                canvas_img = renderer.render(spikes=None, drive=act)

            body_img = None
            if body is not None:
                part_drive = body_driver.update(act, dt, motion)
                body.step(part_drive, dt)
                body_img = body.render(cfg.body.width, cfg.body.height)
                _last_body = body_img

            hud_strip = None
            if ui is not None or a.hud:
                hud_lines = hud_text(n, canvas, cfg, spike_count, brain)
                hud_strip = compose.make_hud_strip(hud_lines, w)

            comp = compose.compose(canvas_img, body_img, cfg, hud_strip)
            globals()["_LAST_FRAME"] = comp
            if writer is not None:
                writer.write(comp)
            if ui is not None:
                ui.draw_surface(comp)

            n += 1
            if n % 30 == 0:
                el = time.time() - t0
                print(
                    f"  帧 {n:>6d}  {n/el:6.1f} fps  {el:7.1f}s"
                    f"  脉冲/帧 {spike_count:>7,}  活动 {int((act>0).sum()):>7,}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[flyscreen] 中断")
    finally:
        src.close()
        if writer is not None:
            writer.close()
            print(f"[out] 已写出 {a.out} （{n} 帧）")
            if a.audio:
                try:
                    # 注意：probe 在模块顶部已经导入过了。这里千万别再写
                    # `from flyscreen.video import probe` —— 那会把它变成整个
                    # 函数的局部变量，遮蔽模块级的导入，函数开头调用就报
                    # UnboundLocalError。
                    from flyscreen.writer import mux_audio

                    if not probe(video).has_audio:
                        print("[out] 源视频没有音轨，跳过混音")
                    else:
                        av = mux_audio(a.out, video, start=0.0)
                        av.replace(a.out)      # 就地替换，别留个 _av 副本
                        print(f"[out] 已混入源视频音轨")
                except Exception as e:  # noqa: BLE001
                    print(f"[out] 混音失败（成片仍然可用）：{type(e).__name__}: {e}")
        if body is not None:
            body.close()
        if ui is not None:
            ui.close()

    print(f"[flyscreen] 完成：{n} 帧，用时 {time.time()-t0:.1f}s")
    return 0


def apply_action(
    action: str,
    cfg: Config,
    renderer: PointCloudRenderer,
    canvas: NeuronCanvas,
    table=None,
) -> None:
    if action == "cycle_view":
        order = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
        try:
            i = order.index(canvas.axes)
        except ValueError:
            i = -1
        canvas.set_view(order[(i + 1) % len(order)])
        renderer._build()
        print(f"[ui] 投影轴 -> {canvas.axes}")
    elif action == "toggle_glow":
        cfg.render.glow_radius = 0 if cfg.render.glow_radius else 4
        print(f"[ui] 辉光半径 -> {cfg.render.glow_radius}")
    elif action == "gain_up":
        cfg.canvas.gain = min(4.0, cfg.canvas.gain * 1.15)
        print(f"[ui] 视频增益 -> {cfg.canvas.gain:.2f}")
    elif action == "gain_down":
        cfg.canvas.gain = max(0.05, cfg.canvas.gain / 1.15)
        print(f"[ui] 视频增益 -> {cfg.canvas.gain:.2f}")
    elif action == "decay_up":
        cfg.canvas.decay = min(0.98, cfg.canvas.decay + 0.02)
        print(f"[ui] 拖尾 -> {cfg.canvas.decay:.2f}")
    elif action == "decay_down":
        cfg.canvas.decay = max(0.0, cfg.canvas.decay - 0.02)
        print(f"[ui] 拖尾 -> {cfg.canvas.decay:.2f}")
    elif action == "screenshot":
        global _LAST_FRAME
        if _LAST_FRAME is not None:
            try:
                from PIL import Image

                p = PROJECT_ROOT / "out" / f"shot_{int(time.time())}.png"
                p.parent.mkdir(exist_ok=True, parents=True)
                Image.fromarray(_LAST_FRAME).save(p)
                print(f"[ui] 截图 -> {p}")
            except Exception as e:  # noqa: BLE001
                print(f"[ui] 截图失败：{e}")


def hud_text(n: int, canvas: NeuronCanvas, cfg: Config, spike_count: int, brain) -> list[str]:
    act = canvas.activity
    lit = int((act > 0).sum())
    mode = "LIF" if brain is not None else "direct"
    return [
        f"frame {n}",
        f"{mode}",
        f"spk/f {spike_count:,}",
        f"lit {lit:,}",
        f"view {canvas.axes}",
        f"decay {cfg.canvas.decay:.2f}",
        "ESC quit SPACE pause V view G glow B body +/- gain <-/-> decay",
    ]


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
