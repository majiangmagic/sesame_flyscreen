"""flyscreen 的 Web 界面。

    python webui.py
    python webui.py --port 8760 --canvas-size 640x520 --body-size 500x520

浏览器打开 http://127.0.0.1:8760/ ，**把视频文件拖进页面**就开始渲染，
左边实时预览，右边调参数（立刻生效），想留档就点「开始录制」，
成片落到 out/ 目录。

原理：浏览器只是显示终端 + 遥控器，所有渲染都在 Python 这边，
通过 MJPEG（multipart/x-mixed-replace）把帧推给 `<img>` 标签。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from flyscreen import compose
from flyscreen.canvas import NeuronCanvas
from flyscreen.config import OUT_DIR, PROJECT_ROOT, Config
from flyscreen.data import dataset
from flyscreen.render import PointCloudRenderer
from flyscreen.video import VideoSource
from flyscreen.web.server import WebUI

VIEWS = {
    "xy": (0, 1), "xz": (0, 2), "yz": (1, 2),
    "yx": (1, 0), "zx": (2, 0), "zy": (2, 1),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="webui")
    # 中文/含空格路径经 Start-Process -ArgumentList 会被拆坏，
    # 所以额外支持环境变量入口：set FLYSCREEN_VIDEO=D:\...\坏苹果.mp4
    p.add_argument("--video", default=os.environ.get("FLYSCREEN_VIDEO") or None,
                   help="启动时就加载的视频（也读环境变量 FLYSCREEN_VIDEO）；"
                        "不给就等你在页面上拖")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8760)
    p.add_argument("--quality", type=int, default=80, help="预览 JPEG 质量 1-95")
    p.add_argument("--stream-scale", type=float, default=1.0, help="预览缩放，0.75 省带宽")
    p.add_argument("--canvas-size", default=None,
                   help="脑点云面板 WxH（默认用 config.py 的 440x380）")
    p.add_argument("--body-size", default=None,
                   help="果蝇面板 WxH（默认用 config.py 的 620x480）")
    p.add_argument("--crf", type=int, default=18, help="录制成片的 x264 CRF，越小越好")
    p.add_argument("--no-body", action="store_true")
    p.add_argument("--no-brain", action="store_true")
    p.add_argument("--float", action="store_true", help="悬空模式（关掉地面）")
    return p.parse_args(argv)


class Session:
    """整条管线 + 录制，支持运行中换片源/换参数。"""

    def __init__(self, cfg: Config, video: str | None) -> None:
        self.cfg = cfg
        self.table = dataset.load()
        self.groups = self.table.group_dict()
        print(f"[web] MaleCNS v1.0: {self.table.n:,} 个神经元")

        self._graph = None
        self.canvas = NeuronCanvas(self.table.pos_um, cfg.canvas)
        self.renderer = PointCloudRenderer(self.canvas, cfg.render)

        self.brain = None
        self.body = None
        self.body_driver = None
        self.build_brain()
        self.build_body()

        self.src: VideoSource | None = None
        self._frames = None
        self.video_path = ""

        # 录制（全自动：开片就录，放到片尾自动收尾）
        self.writer = None
        self.rec_frames = 0
        self.rec_t0 = 0.0
        self.rec_audio = True
        self.rec_last = ""

        self.paused = False
        self.frame_i = 0
        self.dpt = 1.0
        self.n_ticks = 0
        self._recompute_ticks()
        self._t_last = time.time()
        self._fps = 0.0
        # 上一帧的神经元驱动数组，用来算"画面变化量"（门控用）
        self._prev_drive: np.ndarray | None = None

        if video:
            self.open_video(video)

    # ------------------------------------------------------------ 构建
    def build_brain(self) -> None:
        self.brain = None
        if not self.cfg.brain.enabled:
            print("[web] LIF 已关闭（直接点亮模式）")
            return
        from flyscreen.brain import LifConfig, LifNetwork

        try:
            if self._graph is None:
                self._graph = dataset.load_graph()
            indptr, indices, wsyn, n = self._graph
            lcfg = LifConfig(
                dt_ms=self.cfg.brain.dt_ms,
                gain=self.cfg.brain.gain,
                max_rate_hz=self.cfg.brain.max_rate_hz,
                ei_balance=self.cfg.brain.ei_balance,
            )
            self.brain = LifNetwork(indptr, indices, wsyn, n, lcfg)
            print(f"[web] LIF: {n:,} 神经元 / {indices.size:,} 突触  "
                  f"增益 {lcfg.gain}  E/I {self.brain.ei_scale:.3f}  "
                  f"后端 {self.brain.backend}")
        except Exception as e:  # noqa: BLE001
            print(f"[web] LIF 初始化失败，退回直接点亮：{type(e).__name__}: {e}")

    def build_body(self) -> None:
        if self.body is not None:
            self.body.close()
            self.body = None
            self.body_driver = None
        if not self.cfg.body.enabled:
            print("[web] 身体已关闭")
            return
        from flyscreen.body import BodyDriver, FlyBody, find_model_xml

        xml = find_model_xml(PROJECT_ROOT / "data" / "assets" / "flybody")
        if xml is None:
            print("[web] 没找到 flybody，跳过身体")
            return
        try:
            self.body = FlyBody(xml, self.cfg.body)
            self.body_driver = BodyDriver(self.groups, self.cfg.body)
        except Exception as e:  # noqa: BLE001
            print(f"[web] 身体加载失败：{type(e).__name__}: {e}")

    def _recompute_ticks(self) -> None:
        frame_ms = 1000.0 / self.cfg.fps
        if self.brain is not None:
            self.n_ticks = self.cfg.brain.ticks_per_frame or int(
                round(frame_ms / self.cfg.brain.dt_ms))
            self.n_ticks = max(1, min(self.n_ticks, int(self.cfg.brain.max_ticks_per_frame)))
            self.dpt = self.canvas.decay_factor(self.cfg.brain.dt_ms, frame_ms)
        else:
            self.n_ticks = 0
            self.dpt = 1.0

    # ------------------------------------------------------------ 视频
    def open_video(self, path: str | None) -> None:
        if self.src is not None:
            self.src.close()
            self.src = None
        self._frames = None
        if not path:
            print("[web] 没有片源，等你在页面上拖一个进来")
            self.video_path = ""
            return
        if not Path(path).exists():
            print(f"[web] 视频不存在：{path}")
            return
        try:
            # loops=False：让生成器在片尾真的结束，one_frame 的 StopIteration
            # 分支才能收尾出片 + 重新循环。用 loops=True 的话它内部自己转圈，
            # 永远不结束，就会一直录下去。
            src = VideoSource(path, self.cfg.video_scale_width, loops=False)
        except Exception as e:  # noqa: BLE001
            print(f"[web] 打开失败：{type(e).__name__}: {e}")
            return
        self.src = src
        self.video_path = str(path)
        # 生成器必须跟着重建 —— 否则它还会去读已经被 close() 的旧进程
        self._frames = src.frames()
        print(f"[web] 打开视频：{Path(path).name}  "
              f"{src.info.width}x{src.info.height}  {src.info.duration:.1f}s")
        self.canvas.reset()
        self._prev_drive = None          # 换片源，门控从"没有上一帧"重新开始
        if self.brain is not None:
            self.brain.reset()
        self.frame_i = 0
        # 自动开始出片（和实时预览同时进行，走的就是 run.py --out 那条路）
        self.stop_record()
        self.start_record("", audio=True)

    # ------------------------------------------------------------ 录制
    def _safe_stem(self, name: str) -> str:
        s = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", (name or "").strip())
        s = s.strip("._")
        return (s or "flyscreen")[:60]

    def start_record(self, name: str, audio: bool) -> None:
        if self.writer is not None:
            return
        from flyscreen.writer import VideoWriter

        stem = self._safe_stem(name or (Path(self.video_path).stem if self.video_path else ""))
        ts = time.strftime("%m%d_%H%M%S")
        out = OUT_DIR / f"{stem}_flyscreen_{ts}.mp4"
        w, h = compose.layout_size(self.cfg)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.writer = VideoWriter(out, w, h, self.cfg.fps, crf=self.cfg.record_crf)
        self.rec_frames = 0
        self.rec_t0 = time.time()
        self.rec_audio = bool(audio)
        print(f"[web] 开始录制 -> {out.name}  {w}x{h} @{self.cfg.fps}fps")

    def stop_record(self) -> None:
        if self.writer is None:
            return
        path = Path(self.writer.path)
        frames = self.rec_frames
        try:
            self.writer.close()
        finally:
            self.writer = None
            self.rec_last = path.name
        print(f"[web] 出片完成 {path.name}  {frames} 帧  "
              f"{path.stat().st_size/1e6:.1f} MB")
        if self.rec_audio and self.video_path and Path(self.video_path).exists():
            from flyscreen.video import probe
            from flyscreen.writer import mux_audio

            try:
                if not probe(self.video_path).has_audio:
                    print("[web] 源视频没有音轨，跳过混音")
                else:
                    av = mux_audio(path, self.video_path, start=0.0)
                    av.replace(path)
                    print(f"[web] 已混入音轨 -> {path.name}")
            except Exception as e:  # noqa: BLE001
                print(f"[web] 混音失败（成片仍然可用）：{e}")

    # ------------------------------------------------------------ 指令
    def handle(self, cmds, ui: WebUI) -> None:
        for c in cmds:
            if c.kind == "open":
                self.open_video(c.value)      # open_video 内部会自动开始出片
            elif c.kind == "pause":
                self.paused = bool(c.value)
            elif c.kind == "restart":
                # 重播 = 重新打开 + 重新出一遍片
                if self.video_path:
                    self.open_video(self.video_path)
            elif c.kind == "record":
                # 保留 API 兼容；正常流程用不到（开片就自动录）
                if c.key == "start":
                    self.start_record(c.extra.get("name", ""), c.extra.get("audio", True))
                else:
                    self.stop_record()
            elif c.kind == "apply":
                if c.key in ("view", "density"):
                    self.canvas.cfg = self.cfg.canvas
                    if c.key == "view":
                        self.canvas.set_view(self.cfg.canvas.view_axes)
                    else:
                        self.canvas._update_weight()
                    self.renderer._build()
                elif c.key == "brain_enabled":
                    self.build_brain()
                    self._recompute_ticks()
                elif c.key in ("body_enabled", "free_floating"):
                    self.build_body()
                elif c.key == "ticks":
                    self._recompute_ticks()
        ui.set_record(on=self.writer is not None,
                      frames=self.rec_frames,
                      seconds=(time.time() - self.rec_t0) if self.writer else 0.0,
                      path=(Path(self.writer.path).name if self.writer else self.rec_last))
        ui.set_video(self.video_path)      # 切了片源要同步给前端

    # ------------------------------------------------------------ 帧
    def _idle_frame(self):
        """没有片源时也推点东西，让页面不黑屏。"""
        cimg = self.renderer.render(spikes=None, drive=None)
        bimg = None
        body_txt = ""
        if self.body is not None:
            self.body.step({}, 1.0 / self.cfg.fps)
            bimg = self.body.render(self.cfg.body.width, self.cfg.body.height)
            up = self.body.uprightness()
            body_txt = ("正立" if up > 0.7 else "倾斜" if up > 0.2 else "翻倒") + f" {up:+.2f}"
        comp = compose.compose(cimg, bimg, self.cfg, None)
        return comp, {"frame": 0, "fps": 0.0, "spikes": 0,
                      "lit": 0, "body": body_txt, "video": ""}

    def one_frame(self):
        if self.src is None or self._frames is None:
            time.sleep(0.15)
            return self._idle_frame()
        try:
            frame = next(self._frames)
        except StopIteration:
            # 片源放完：把这次出片收尾，然后循环预览（不再重复出片）
            if self.writer is not None:
                self.stop_record()
                print("[web] 片源放完，已自动出片。"
                      "点「重播并重新出片」可以再来一遍。")
            self._frames = self.src.frames()
            try:
                frame = next(self._frames)
            except StopIteration:
                time.sleep(0.05)
                return None, None
        self.frame_i += 1

        drive = self.canvas.drive_from_frame(frame)
        # 画面变化量 = 与上一帧驱动数组的平均绝对差。画面没变（静止视频、重复帧）
        # 精确得到 0，门控会把身体整个冻住 —— 这是"静止图不抽"的实现点。
        motion = (0.0 if self._prev_drive is None
                  else float(np.abs(drive - self._prev_drive).mean()))
        self._prev_drive = drive
        if self.brain is not None:
            self.brain.set_drive(drive)
            spk = 0
            for _ in range(self.n_ticks):
                s = self.brain.step()
                spk += int(s.sum())
                self.canvas.advance_trace(s, self.dpt)
            act = self.canvas.activity
            spikes = spk + self.brain.last_input_spikes * self.n_ticks
            cimg = self.renderer.render(spikes=act, drive=drive)
        else:
            act = self.canvas.advance(drive)
            spikes = int((act > 0).sum())
            cimg = self.renderer.render(spikes=None, drive=act)

        bimg = None
        body_txt = ""
        if self.body is not None:
            part = self.body_driver.update(act, 1.0 / self.cfg.fps, motion)
            self.body.step(part, 1.0 / self.cfg.fps)
            bimg = self.body.render(self.cfg.body.width, self.cfg.body.height)
            up = self.body.uprightness()
            body_txt = ("正立" if up > 0.7 else "倾斜" if up > 0.2 else "翻倒") + f" {up:+.2f}"

        comp = compose.compose(cimg, bimg, self.cfg, None)

        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt if dt > 0 else 0)
        status = {
            "frame": self.frame_i,
            "fps": round(self._fps, 1),
            "spikes": spikes,
            "lit": int((act > 0).sum()),
            "body": body_txt,
            "video": Path(self.video_path).name if self.video_path else "",
        }
        return comp, status

    # ------------------------------------------------------------ 主循环
    def run(self, ui: WebUI) -> None:
        print("[web] 开始推流（Ctrl-C 停止）")
        errs = 0
        try:
            while True:
                self.handle(ui.take_commands(), ui)
                ui.set_paused(self.paused)
                if self.paused:
                    time.sleep(0.05)
                    continue
                self.canvas.cfg = self.cfg.canvas
                self.renderer.cfg = self.cfg.render
                try:
                    comp, status = self.one_frame()
                except Exception as e:  # noqa: BLE001
                    errs += 1
                    print(f"[web] 帧处理出错（第 {errs} 次，已跳过）："
                          f"{type(e).__name__}: {e}")
                    if errs > 200:
                        raise
                    time.sleep(0.1)
                    continue
                if comp is None:
                    continue
                if self.writer is not None:
                    try:
                        self.writer.write(comp)
                        self.rec_frames += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"[web] 写帧失败，停止录制：{e}")
                        self.stop_record()
                ui.publish(comp, status)
        except KeyboardInterrupt:
            print("\n[web] 停止")
        finally:
            self.stop_record()
            if self.src is not None:
                self.src.close()
            if self.body is not None:
                self.body.close()


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = Config()
    if a.canvas_size:
        w, h = a.canvas_size.lower().split("x")
        cfg.render.canvas_w, cfg.render.canvas_h = int(w), int(h)
    if a.body_size:
        w, h = a.body_size.lower().split("x")
        cfg.body.width, cfg.body.height = int(w), int(h)
    if a.no_body:
        cfg.body.enabled = False
    if a.no_brain:
        cfg.brain.enabled = False
    if a.float:
        cfg.body.free_floating = True
    cfg.canvas.view_name = "xy"
    cfg.canvas.view_axes = VIEWS["xy"]
    cfg.record_crf = int(a.crf)

    ui = WebUI(cfg, a.host, a.port, jpeg_quality=a.quality, stream_scale=a.stream_scale)
    ui.start()
    sess = Session(cfg, a.video)
    sess.run(ui)
    ui.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
