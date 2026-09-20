"""通用视频读取层：任意视频文件 -> 灰度帧（float32, 0..1）。

用 ffmpeg 子进程解 rawvideo，因此支持任何 ffmpeg 认识的容器/编码，
不依赖 opencv 的编解码能力。

对外接口：
    probe(path)                  -> VideoInfo
    VideoSource(path, ...)       -> 可迭代 / 可 seek / 可循环
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# ffmpeg 定位：优先 PATH，其次 imageio_ffmpeg 自带的二进制
_FFMPEG: str | None = None
_FFPROBE: str | None = None


def _find_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg  # type: ignore

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if not exe:
        raise RuntimeError(
            "找不到 ffmpeg。请安装 ffmpeg 并加入 PATH，或 pip install imageio-ffmpeg。"
        )
    _FFMPEG = exe
    return exe


def _find_ffprobe() -> str | None:
    global _FFPROBE
    if _FFPROBE:
        return _FFPROBE
    exe = shutil.which("ffprobe")
    if exe:
        _FFPROBE = exe
    return exe


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int  # 0 表示未知
    duration: float
    has_audio: bool = False

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)


def probe(path: str | Path) -> VideoInfo:
    """读取视频元信息。优先 ffprobe，缺失时用 ffmpeg 解析 stderr。"""
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)

    ffprobe = _find_ffprobe()
    if ffprobe:
        cmd = [
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", path,
        ]
        # 必须显式指定 utf-8：ffprobe 吐的是 UTF-8 JSON，而 text=True 不写编码时
        # Python 会按系统 locale 解码，中文 Windows 上是 GBK —— 只要片源路径或
        # 元数据里带中文，就会在读取线程里抛 UnicodeDecodeError，stdout 变成
        # None，紧接着 json.loads(None) 抛 TypeError，整个打开流程静默失败。
        out = subprocess.run(cmd, capture_output=True, text=True, check=True,
                             encoding="utf-8", errors="replace").stdout
        meta = json.loads(out)
        v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
        if v is None:
            raise RuntimeError(f"文件里没有视频流：{path}")
        num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den) if float(den or 1) else 0.0
        if fps <= 0:
            num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den) if float(den or 1) else 25.0
        n = int(v.get("nb_frames") or 0)
        dur = float(meta.get("format", {}).get("duration") or 0.0)
        if n <= 0 and dur > 0 and fps > 0:
            n = int(round(dur * fps))
        return VideoInfo(
            path=path,
            width=int(v["width"]),
            height=int(v["height"]),
            fps=fps or 25.0,
            n_frames=n,
            duration=dur,
            has_audio=any(s.get("codec_type") == "audio" for s in meta.get("streams", [])),
        )

    # 退化路径：直接用 ffmpeg 打开，从 banner 里抓尺寸
    ffmpeg = _find_ffmpeg()
    p = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    text = p.stderr
    import re

    m = re.search(r"Video: .*?, (\d+)x(\d+)", text)
    if not m:
        raise RuntimeError(f"无法解析视频信息：{path}")
    fps_m = re.search(r"(\d+(?:\.\d+)?) fps", text)
    dur_m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    dur = 0.0
    if dur_m:
        dur = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
    fps = float(fps_m.group(1)) if fps_m else 25.0
    return VideoInfo(
        path=path,
        width=int(m.group(1)),
        height=int(m.group(2)),
        fps=fps,
        n_frames=int(dur * fps) if dur else 0,
        duration=dur,
        has_audio="Audio:" in text,
    )


class VideoSource:
    """把视频解码成灰度帧。

    参数
    ----
    out_width
        输出宽度（等比缩放到该宽度），0 = 保持原尺寸。
    loops
        读到结尾后是否回绕到开头继续（预览用）。
    start
        起始秒数。
    """

    def __init__(
        self,
        path: str | Path,
        out_width: int = 480,
        loops: bool = True,
        start: float = 0.0,
        extra_input_args: list[str] | None = None,
    ) -> None:
        self.path = str(path)
        self.info = probe(self.path)
        self.loops = loops
        self.start = float(start)

        if out_width and out_width > 0:
            self.out_width = int(out_width)
            self.out_height = max(2, int(round(out_width / self.info.aspect)))
        else:
            self.out_width = self.info.width
            self.out_height = self.info.height
        # rawvideo 需要偶数宽高（yuv/灰度虽然不强制，但保持偶数最稳）
        self.out_width -= self.out_width % 2
        self.out_height -= self.out_height % 2

        self._extra = extra_input_args or []
        self._proc: subprocess.Popen | None = None
        self.frame_index = 0

    # ------------------------------------------------------------ 内部
    def _spawn(self) -> subprocess.Popen:
        ffmpeg = _find_ffmpeg()
        vf = f"scale={self.out_width}:{self.out_height},format=gray"
        cmd = [ffmpeg, "-v", "error", "-nostdin"]
        if self.start > 0:
            cmd += ["-ss", f"{self.start:.3f}"]
        cmd += self._extra
        cmd += [
            "-i", self.path,
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-",
        ]
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )

    def _read_one(self) -> np.ndarray | None:
        # 进程可能已被 close() 关掉（例如运行中切换片源），这时安静地结束，
        # 让上层把生成器重建掉，而不是抛断言。
        if self._proc is None or self._proc.stdout is None:
            return None
        nbytes = self.out_width * self.out_height
        try:
            buf = self._proc.stdout.read(nbytes)
        except (ValueError, OSError):
            return None
        if not buf or len(buf) < nbytes:
            return None
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.out_height, self.out_width)
        return frame

    # ------------------------------------------------------------ 公开
    def frames(self, max_frames: int = 0) -> Iterator[np.ndarray]:
        """生成 float32 灰度帧，取值 0..1，形状 (out_height, out_width)。"""
        produced = 0
        while True:
            self._proc = self._spawn()
            got_any = False
            while True:
                frame = self._read_one()
                if frame is None:
                    break
                got_any = True
                produced += 1
                self.frame_index += 1
                if max_frames and produced >= max_frames:
                    self.close()
                    yield frame.astype(np.float32) / 255.0
                    return
                yield frame.astype(np.float32) / 255.0
            self.close()
            if not got_any or not self.loops:
                return

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdout:
                    self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def ensure_even(n: int) -> int:
    return n - (n % 2)
