"""ffmpeg 视频写出：把 RGB 帧管道给 ffmpeg 编码成 mp4。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .video import _find_ffmpeg


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: int = 30,
        crf: int = 18,
        preset: str = "medium",
        pix_fmt_in: str = "rgb24",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width) - int(width) % 2
        self.height = int(height) - int(height) % 2
        self.fps = int(fps)
        self.pix_fmt_in = pix_fmt_in
        self.n = 0

        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-y", "-v", "error", "-nostdin",
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt_in,
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        f = np.ascontiguousarray(frame[: self.height, : self.width])
        self._proc.stdin.write(f.tobytes())
        self.n += 1

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            err = self._proc.stderr.read().decode("utf-8", "replace") if self._proc.stderr else ""
            rc = self._proc.wait(timeout=120)
            if rc != 0:
                raise RuntimeError(f"ffmpeg 编码失败 (rc={rc}): {err[:800]}")
        finally:
            self._proc = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def mux_audio(
    video_path: str | Path,
    audio_from: str | Path,
    out_path: str | Path | None = None,
    start: float = 0.0,
) -> Path:
    """把源视频的音轨混进渲染结果（画面直接 copy，不重编码）。

    源视频没有音轨时直接抛错，让调用方决定怎么办 —— 注意 `-map 1:a:0?`
    那个问号：没有它 ffmpeg 会因为找不到音频流而整体失败。
    """
    video_path = Path(video_path)
    out_path = Path(out_path) if out_path else video_path.with_name(
        video_path.stem + "_av" + video_path.suffix
    )
    ffmpeg = _find_ffmpeg()
    cmd = [ffmpeg, "-y", "-v", "error", "-nostdin", "-i", str(video_path)]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += [
        "-i", str(audio_from),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"混音失败: {p.stderr.decode('utf-8', 'replace')[:500]}")
    return out_path
