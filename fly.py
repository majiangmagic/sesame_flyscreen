"""flyscreen 统一入口。

    python fly.py                        # 起 Web 服务（默认），日志自动写 data/logs/
    python fly.py render 视频.mp4         # 只出视频，输出自动命名到 out/
    python fly.py preview 视频.mp4        # 开 pygame 预览窗
    python fly.py check                  # 自检：数据齐不齐、依赖装没装

就是把原来 webui.py / run.py 两个入口收成一个。根目录那两个脚本还在，
需要细调参数时可以直接用；日常用这个就够了。

面板尺寸用三档代替原来那串 --canvas-size/--body-size：

    python fly.py --size small           # 600x490 + 460x490   快，预览流畅
    python fly.py --size mid             # 800x660 + 580x660
    python fly.py --size big             # 1000x820 + 700x820  默认，点云分离最清晰
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 面板三档：(画布宽, 画布高, 身体宽, 身体高)
SIZES = {
    "small": (600, 490, 460, 490),
    "mid": (800, 660, 580, 660),
    "big": (1000, 820, 700, 820),
}


# ---------------------------------------------------------------- 日志
class _Tee(io.TextIOBase):
    """同时往控制台和文件写。服务模式的日志靠它，不用外部重定向。"""

    def __init__(self, path: Path) -> None:
        self.f = path.open("w", encoding="utf-8", buffering=1)  # 行缓冲，实时落盘
        self.console = sys.__stdout__

    def write(self, s: str) -> int:
        try:
            self.console.write(s)
            self.console.flush()
        except Exception:  # noqa: BLE001
            pass
        self.f.write(s)
        return len(s)

    def flush(self) -> None:
        try:
            self.console.flush()
        except Exception:  # noqa: BLE001
            pass
        self.f.flush()


def _setup_log(name: str = "webui.log") -> Path:
    logdir = ROOT / "data" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / name
    tee = _Tee(path)
    sys.stdout = tee          # type: ignore[assignment]
    sys.stderr = tee          # type: ignore[assignment]
    return path


# ---------------------------------------------------------------- 公共参数
def _apply_common(cfg, a) -> None:
    w, h, bw, bh = SIZES[a.size]
    cfg.render.canvas_w, cfg.render.canvas_h = w, h
    cfg.body.width, cfg.body.height = bw, bh
    if a.no_body:
        cfg.body.enabled = False
    if a.no_brain:
        cfg.brain.enabled = False
    if a.floating:
        cfg.body.free_floating = True


def _common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--size", choices=sorted(SIZES), default="big",
                     help="面板档位（默认 big）。small 最快，big 点云最清晰")
    sub.add_argument("--no-body", action="store_true", help="只要脑点云，不要果蝇")
    sub.add_argument("--no-brain", action="store_true", help="关掉 LIF 神经仿真，直接点亮")
    sub.add_argument("--float", action="store_true", dest="floating",
                     help="悬空模式（关掉地面物理）")


# ---------------------------------------------------------------- 各子命令
def cmd_web(a) -> int:
    log = _setup_log("webui.log" if not a.log else a.log)
    print(f"[fly] 日志 -> {log}")
    argv = ["--port", str(a.port), "--quality", str(a.quality)]
    if a.video:
        argv += ["--video", a.video]
    w, h, bw, bh = SIZES[a.size]
    argv += ["--canvas-size", f"{w}x{h}", "--body-size", f"{bw}x{bh}"]
    if a.no_body:
        argv += ["--no-body"]
    if a.no_brain:
        argv += ["--no-brain"]
    if a.floating:
        argv += ["--float"]
    import webui

    return webui.main(argv)


def cmd_render(a) -> int:
    src = Path(a.video)
    if not src.exists():
        print(f"!! 找不到视频：{src}", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else (
        ROOT / "out" / f"{src.stem}_flyscreen_{time.strftime('%m%d_%H%M%S')}.mp4"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fly] {src.name}  ->  {out}")
    argv = ["--video", str(src), "--out", str(out)]
    w, h, bw, bh = SIZES[a.size]
    argv += ["--canvas-size", f"{w}x{h}", "--body-size", f"{bw}x{bh}"]
    if a.seconds:
        argv += ["--seconds", str(a.seconds)]
    if a.no_body:
        argv += ["--no-body"]
    if a.no_brain:
        argv += ["--no-brain"]
    if a.floating:
        argv += ["--float"]
    if a.no_audio:
        argv += ["--no-audio"]
    if a.crf is not None:
        argv += ["--quality", str(a.crf)]
    import run

    return run.main(argv)


def cmd_preview(a) -> int:
    src = Path(a.video)
    if not src.exists():
        print(f"!! 找不到视频：{src}", file=sys.stderr)
        return 2
    argv = ["--video", str(src), "--preview"]
    w, h, bw, bh = SIZES[a.size]
    argv += ["--canvas-size", f"{w}x{h}", "--body-size", f"{bw}x{bh}"]
    if a.no_body:
        argv += ["--no-body"]
    if a.no_brain:
        argv += ["--no-brain"]
    if a.floating:
        argv += ["--float"]
    import run

    return run.main(argv)


def cmd_check(_a) -> int:
    import importlib.util
    import shutil

    print("=" * 62)
    print("依赖")
    print("=" * 62)
    ok = True
    for m in ("numpy", "pyarrow", "pygame", "mujoco", "PIL"):
        have = importlib.util.find_spec(m) is not None
        ok &= have
        print(f"  {m:<10} {'OK' if have else '缺失 -> pip install -r requirements.txt'}")
    for m in ("numba",):
        have = importlib.util.find_spec(m) is not None
        print(f"  {m:<10} {'OK' if have else '缺失（可选，装上 LIF 快 6.7 倍）'}")
    ff = shutil.which("ffmpeg")
    ok &= bool(ff)
    print(f"  {'ffmpeg':<10} {ff or '缺失（必须）'}")

    print()
    print("=" * 62)
    print("数据")
    print("=" * 62)
    need = [
        ("data/raw/body-annotations.feather", "python -m flyscreen.data.fetch"),
        ("data/raw/body-neurotransmitters.feather", "python -m flyscreen.data.fetch"),
        ("data/raw/connectome-weights.feather", "python -m flyscreen.data.fetch"),
        ("data/cache/neurons.npz", "python -m flyscreen.data.prepare"),
        ("data/cache/graph.npz", "python -m flyscreen.data.prepare_graph"),
    ]
    for rel, how in need:
        p = ROOT / rel
        if p.exists():
            print(f"  OK   {rel:<44} {p.stat().st_size/1e6:9.1f} MB")
        else:
            ok = False
            print(f"  缺   {rel:<44} 跑：{how}")

    fb = ROOT / "data" / "assets" / "flybody" / "fruitfly.xml"
    if fb.exists():
        n = len(list((fb.parent / "assets").glob("*.obj")))
        print(f"  OK   data/assets/flybody/{'':<27} {n} 个网格")
    else:
        ok = False
        print(f"  缺   data/assets/flybody/{'':<27} 跑：python scripts/fetch_flybody.py")

    print()
    if ok:
        print("一切就绪。  python fly.py            # 起 Web 服务")
        print("            python fly.py render 视频.mp4")
    else:
        print("上面标了缺什么，按提示补。")
    return 0 if ok else 1


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fly",
        description="把任意视频播进一只果蝇的脑子里（MaleCNS v1.0 + MuJoCo 果蝇）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "面板档位 --size：\n"
            "  small   600x490 + 460x490    快，预览流畅，出片快\n"
            "  mid     800x660 + 580x660\n"
            "  big     1000x820 + 700x820   默认，点云分离最清晰（渲染慢 2.8 倍）\n"
            "\n"
            "例：\n"
            "  python fly.py                       起 Web 服务，浏览器开 127.0.0.1:8760\n"
            "  python fly.py render bad.mp4        出片到 out/bad_flyscreen_<时间>.mp4\n"
            "  python fly.py render bad.mp4 a.mp4  指定输出路径\n"
            "  python fly.py --size small          起服务，用小画布\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd")

    w = sub.add_parser("web", help="起 Web 服务（默认命令）")
    _common(w)
    w.add_argument("--port", type=int, default=8760)
    w.add_argument("--quality", type=int, default=82, help="预览 JPEG 质量 1-95")
    w.add_argument("--video", default=None, help="启动就加载的视频；不给就等拖拽")
    w.add_argument("--log", default=None, help="日志文件名，默认 data/logs/webui.log")

    r = sub.add_parser("render", help="只出视频")
    _common(r)
    r.add_argument("video", help="输入视频")
    r.add_argument("out", nargs="?", default=None, help="输出 mp4，省略就自动命名到 out/")
    r.add_argument("--seconds", type=float, default=0.0, help="只处理前 N 秒（试片用）")
    r.add_argument("--crf", type=int, default=None, help="x264 CRF，越小越好，默认 18")
    r.add_argument("--no-audio", action="store_true", help="不混入源视频音轨")

    v = sub.add_parser("preview", help="开 pygame 预览窗")
    _common(v)
    v.add_argument("video", help="输入视频")

    sub.add_parser("check", help="自检环境与数据")
    return p


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # 不带子命令（或直接给选项）时默认走 web：`python fly.py --size mid` 就等于
    # `python fly.py web --size mid`。但 -h/--help 要留给顶层，否则看不到子命令列表。
    if not raw or (raw[0].startswith("-") and raw[0] not in ("-h", "--help")):
        raw = ["web"] + raw

    a = build_parser().parse_args(raw)
    return {
        "web": cmd_web,
        "render": cmd_render,
        "preview": cmd_preview,
        "check": cmd_check,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
