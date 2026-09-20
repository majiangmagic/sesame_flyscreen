"""生成一段测试视频（不需要外部素材）。

    python scripts/make_test_video.py out/_test.mp4
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyscreen.video import _find_ffmpeg  # noqa: E402

W, H, FPS, SECONDS = 480, 360, 30, 12


def frame(i: int, n: int) -> np.ndarray:
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    t = i / n

    # 李萨如轨迹的大圆
    cx = W / 2 + math.sin(t * math.tau * 2) * (W * 0.30)
    cy = H / 2 + math.sin(t * math.tau * 3 + 1.0) * (H * 0.26)
    r = 26 + 14 * math.sin(t * math.tau * 5)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    # 旋转的方框
    a = t * math.tau
    s = 70
    pts = [
        (W / 2 + math.cos(a + k * math.pi / 2) * s, H / 2 + math.sin(a + k * math.pi / 2) * s)
        for k in range(4)
    ]
    d.polygon(pts, outline=200, width=5)

    # 扫过的竖条（测试横向分辨率）
    for k in range(9):
        x = ((k / 9.0 + t * 1.5) % 1.0) * W
        d.rectangle([x - 4, 12, x + 4, 52], fill=230)

    # 底部进度条
    d.rectangle([12, H - 26, 12 + (W - 24) * t, H - 14], fill=180)

    # 文字
    d.text((14, 14), f"FLYSCREEN  frame {i:04d}", fill=255)
    return np.asarray(img)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/_test.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = FPS * SECONDS
    cmd = [
        _find_ffmpeg(), "-y", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(out),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    for i in range(n):
        p.stdin.write(frame(i, n).tobytes())
    p.stdin.close()
    rc = p.wait()
    print(f"{'OK' if rc == 0 else 'FAIL'}  {out}  {n} 帧 {W}x{H}@{FPS}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
