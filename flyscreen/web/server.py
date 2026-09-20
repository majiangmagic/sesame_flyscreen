"""WebUI：纯标准库的 HTTP 服务 + MJPEG 视频流 + 参数面板 + 拖拽上传 + 录制。

不依赖 Flask/FastAPI/websockets —— 只用 http.server + threading。

原理
----
浏览器**不参与渲染**，只是个「显示终端 + 遥控器」：

    渲染线程（主线程）
      视频帧 → 画布 → LIF → 渲染 → 合成 → RGB 数组
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
              JPEG 编码 → 最新帧槽                  写进 ffmpeg 管道 → out/*.mp4
                        │
    HTTP 服务线程       ▼
      GET /stream  ──► 每个连接独立等新帧（Condition 唤醒）──► 浏览器 <img>
      GET /api/set ──► 直接改内存里的 Config 对象 ──► 下一帧生效

路由
----
    GET  /                    面板（拖拽区 + 录制 + 20 个参数控件）
    GET  /stream              MJPEG 流
    POST /api/upload?name=    原始文件体直接落盘，然后切到它
    GET  /api/record?on=      开始/停止录制
    GET  /api/state           状态 JSON
    GET  /api/set?k=&v=       改参数
    GET  /api/pause /api/restart
"""
from __future__ import annotations

import io
import json
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import Config, PROJECT_ROOT

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUT_DIR = PROJECT_ROOT / "out"

# ---------------------------------------------------------------- 参数表
ParamT = dict[str, Any]


def _p(label: str, kind: str, lo=None, hi=None, step=None,
       get: Callable[[Config], Any] = None, set: Callable[[Config, Any], None] = None,
       on_change: str | None = None, options=None, group: str = "") -> ParamT:
    return dict(label=label, kind=kind, lo=lo, hi=hi, step=step,
                get=get, set=set, on_change=on_change, options=options, group=group)


VIEW_OPTIONS = [
    ("xy", (0, 1), "背面俯视（最好看）"),
    ("xz", (0, 2), "侧视"),
    ("yz", (1, 2), "正视"),
    ("yx", (1, 0), "背面俯视·翻转"),
    ("zx", (2, 0), "侧视·翻转"),
    ("zy", (2, 1), "正视·翻转"),
]

PARAMS: dict[str, ParamT] = {
    "view": _p("投影轴", "select", get=lambda c: c.canvas.view_name,
               set=lambda c, v: setattr(c.canvas, "view_name", v),
               on_change="view", group="画面",
               options=[(k, lbl) for k, _, lbl in VIEW_OPTIONS]),
    "density": _p("密度均衡", "select", get=lambda c: c.canvas.density_equalize,
                  set=lambda c, v: setattr(c.canvas, "density_equalize", v),
                  on_change="density", group="画面",
                  options=[("sqrt", "sqrt（推荐）"), ("none", "关"), ("full", "full")]),
    "decay": _p("荧光拖尾", "float", 0.0, 0.98, 0.01,
                get=lambda c: c.canvas.decay, set=lambda c, v: setattr(c.canvas, "decay", v),
                group="画面"),
    "gain": _p("视频亮度增益", "float", 0.05, 4.0, 0.05,
               get=lambda c: c.canvas.gain, set=lambda c, v: setattr(c.canvas, "gain", v),
               group="画面"),
    "gamma": _p("亮度曲线 γ", "float", 0.2, 4.0, 0.05,
                get=lambda c: c.canvas.gamma, set=lambda c, v: setattr(c.canvas, "gamma", v),
                group="画面"),
    "black_point": _p("黑场（压中间调）", "float", 0.0, 0.9, 0.01,
                      get=lambda c: c.canvas.black_point,
                      set=lambda c, v: setattr(c.canvas, "black_point", v), group="画面"),

    "structure_gain": _p("脑结构亮度", "float", 0.0, 2.0, 0.02,
                         get=lambda c: c.render.structure_gain,
                         set=lambda c, v: setattr(c.render, "structure_gain", v), group="配色"),
    "drive_gain": _p("视频层亮度（冷色）", "float", 0.0, 4.0, 0.05,
                     get=lambda c: c.render.drive_gain,
                     set=lambda c, v: setattr(c.render, "drive_gain", v), group="配色"),
    "activity_gain": _p("脉冲层亮度（暖色）", "float", 0.0, 6.0, 0.05,
                        get=lambda c: c.render.activity_gain,
                        set=lambda c, v: setattr(c.render, "activity_gain", v), group="配色"),
    "glow_radius": _p("辉光半径", "int", 0, 16, 1,
                      get=lambda c: c.render.glow_radius,
                      set=lambda c, v: setattr(c.render, "glow_radius", int(v)), group="配色"),

    "brain_enabled": _p("LIF 神经仿真", "bool", get=lambda c: c.brain.enabled,
                        set=lambda c, v: setattr(c.brain, "enabled", bool(v)),
                        on_change="brain", group="神经"),
    "lif_gain": _p("LIF 增益", "float", 0.0, 1.5, 0.01,
                   get=lambda c: c.brain.gain, set=lambda c, v: setattr(c.brain, "gain", v),
                   group="神经"),
    "max_rate": _p("驱动泊松率 Hz", "float", 5.0, 300.0, 5.0,
                   get=lambda c: c.brain.max_rate_hz,
                   set=lambda c, v: setattr(c.brain, "max_rate_hz", v), group="神经"),
    "ticks": _p("每帧 tick 数", "int", 1, 120, 1,
                get=lambda c: c.brain.ticks_per_frame,
                set=lambda c, v: setattr(c.brain, "ticks_per_frame", int(v)),
                on_change="ticks", group="神经"),

    "body_enabled": _p("MuJoCo 身体", "bool", get=lambda c: c.body.enabled,
                       set=lambda c, v: setattr(c.body, "enabled", bool(v)),
                       on_change="body", group="身体"),
    "free_floating": _p("悬空（关掉地面）", "bool", get=lambda c: c.body.free_floating,
                        set=lambda c, v: setattr(c.body, "free_floating", bool(v)),
                        on_change="body", group="身体"),
    "travel": _p("驱动幅度 travel", "float", 0.0, 1.5, 0.01,
                 get=lambda c: c.body.travel, set=lambda c, v: setattr(c.body, "travel", v),
                 group="身体"),
    "twitch_gain": _p("抽搐强度", "float", 0.0, 3.0, 0.05,
                      get=lambda c: c.body.twitch_gain,
                      set=lambda c, v: setattr(c.body, "twitch_gain", v), group="身体"),
    "jitter_gain": _p("抖动强度", "float", 0.0, 3.0, 0.05,
                      get=lambda c: c.body.jitter_gain,
                      set=lambda c, v: setattr(c.body, "jitter_gain", v), group="身体"),
    "smoothing_tau": _p("平滑时间常数 ms", "float", 2.0, 60.0, 1.0,
                        get=lambda c: c.body.smoothing_tau * 1000.0,
                        set=lambda c, v: setattr(c.body, "smoothing_tau", v / 1000.0),
                        group="身体"),
}


@dataclass
class Command:
    kind: str                       # set / apply / open / pause / restart / record
    key: str = ""
    value: Any = None
    extra: dict = field(default_factory=dict)


class WebUI:
    def __init__(self, cfg: Config, host: str = "127.0.0.1", port: int = 8760,
                 jpeg_quality: int = 82, stream_scale: float = 1.0) -> None:
        self.cfg = cfg
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.stream_scale = float(stream_scale)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._commands: list[Command] = []
        self._status: dict[str, Any] = {}
        self._current_video: str = ""
        self._paused = False
        self._record: dict[str, Any] = {"on": False, "frames": 0, "seconds": 0.0, "path": ""}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._t0 = time.time()
        self._last_upload = ""
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self._httpd is not None:
            return
        self._httpd = ThreadingHTTPServer((self.host, self.port), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="webui-http", daemon=True)
        self._thread.start()
        print(f"[web] 面板地址 http://{self.host}:{self.port}/")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ------------------------------------------------------------ 渲染线程侧
    def publish(self, rgb: np.ndarray, status: dict[str, Any] | None = None) -> None:
        jpg = _encode(rgb, self.jpeg_quality, self.stream_scale)
        with self._cond:
            self._jpeg = jpg
            self._frame_id += 1
            if status:
                self._status.update(status)
            self._status["record"] = dict(self._record)
            self._cond.notify_all()

    def take_commands(self) -> list[Command]:
        with self._lock:
            cmds = self._commands
            self._commands = []
            return cmds

    def _push(self, cmd: Command) -> None:
        with self._lock:
            self._commands.append(cmd)

    def set_video(self, path: str) -> None:
        with self._lock:
            self._current_video = path

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def set_record(self, **kw) -> None:
        with self._lock:
            self._record.update(kw)

    def _wait_frame(self, last_id: int, timeout: float = 5.0):
        with self._cond:
            if self._frame_id == last_id:
                self._cond.wait(timeout)
            return self._frame_id, self._jpeg

    def _snapshot(self) -> dict:
        with self._lock:
            cfg = self.cfg
            vals = {}
            for k, p in PARAMS.items():
                try:
                    vals[k] = p["get"](cfg)
                except Exception:  # noqa: BLE001
                    vals[k] = None
            return {
                "params": vals,
                "status": dict(self._status),
                "current": self._current_video,
                "paused": self._paused,
                "record": dict(self._record),
                "uptime": round(time.time() - self._t0, 1),
            }

    # ------------------------------------------------------------ 上传
    def _receive_upload(self, name: str, body, length: int) -> tuple[bool, str]:
        safe = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", Path(name).name) or "upload.mp4"
        dest = UPLOAD_DIR / safe
        k = 1
        while dest.exists():
            dest = UPLOAD_DIR / f"{Path(safe).stem}_{k}{Path(safe).suffix}"
            k += 1
        got = 0
        try:
            with dest.open("wb") as f:
                remain = length
                while remain > 0:
                    chunk = body.read(min(1 << 20, remain))
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    remain -= len(chunk)
        except Exception as e:  # noqa: BLE001
            return False, f"写入失败: {e}"
        if length and got != length:
            return False, f"上传不完整 {got}/{length}"
        print(f"[web] 收到上传 {dest.name}  {got/1e6:.1f} MB")
        with self._lock:
            self._last_upload = str(dest)
        self._push(Command("open", "video", str(dest)))
        return True, dest.name

    def _apply(self, key: str, raw: str) -> str:
        p = PARAMS.get(key)
        if p is None:
            return f"未知参数 {key}"
        try:
            if p["kind"] == "bool":
                val = str(raw).lower() in ("1", "true", "on", "yes")
            elif p["kind"] == "int":
                val = int(float(raw))
            elif p["kind"] == "float":
                val = float(raw)
            else:
                val = str(raw)
        except Exception as e:  # noqa: BLE001
            return f"值解析失败: {e}"
        with self._lock:
            p["set"](self.cfg, val)
            if key == "view":
                for k, axes, _ in VIEW_OPTIONS:
                    if k == str(val):
                        self.cfg.canvas.view_axes = axes
                        break
        hook = p.get("on_change")
        if hook:
            self._push(Command("apply", key, val))
        return "ok"


def param_schema() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, p in PARAMS.items():
        out[key] = {"label": p["label"], "kind": p["kind"], "lo": p["lo"], "hi": p["hi"],
                    "step": p["step"], "options": p["options"], "group": p["group"] or "其他"}
    return out


def _encode(rgb: np.ndarray, quality: int, scale: float) -> bytes:
    """编码成 JPEG。

    **必须用 subsampling=0（4:4:4）**。点云本质上是高频噪点，默认的 4:2:0
    色度二次采样会在高频区域把色度整个丢掉，画面变成一片灰。
    """
    from PIL import Image

    im = Image.fromarray(rgb)
    if abs(scale - 1.0) > 1e-3:
        w = max(2, int(im.width * scale))
        h = max(2, int(im.height * scale))
        im = im.resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=False, subsampling=0)
    return buf.getvalue()


def _make_handler(ui: WebUI):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "flyscreen-web"

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            """客户端直接断开（关标签页 / curl 超时）时不打 traceback。"""
            try:
                super().handle_one_request()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                self.close_connection = True

        # ---------------------------------------------------- 工具
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        # ---------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            path = u.path

            if path in ("/", "/index.html"):
                html = PAGE.replace("__PARAMS__",
                                    json.dumps(param_schema(), ensure_ascii=False))
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._json(ui._snapshot())
                return
            if path == "/api/set":
                msg = ui._apply((q.get("k") or [""])[0], (q.get("v") or [""])[0])
                self._json({"ok": msg == "ok", "msg": msg})
                return
            if path == "/api/pause":
                paused = (q.get("v") or ["1"])[0] not in ("0", "false")
                ui._push(Command("pause", "", paused))
                self._json({"ok": True, "paused": paused})
                return
            if path == "/api/restart":
                ui._push(Command("restart"))
                self._json({"ok": True})
                return
            if path == "/api/record":
                on = (q.get("on") or ["1"])[0] not in ("0", "false")
                name = (q.get("name") or [""])[0]
                audio = (q.get("audio") or ["1"])[0] not in ("0", "false")
                ui._push(Command("record", "start" if on else "stop", None,
                                 {"name": name, "audio": audio}))
                self._json({"ok": True, "on": on})
                return
            if path == "/stream":
                self._stream()
                return
            self._send(404, b"not found", "text/plain")

        # ---------------------------------------------------- POST
        def do_POST(self) -> None:  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path != "/api/upload":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                self._json({"ok": False, "msg": "空文件"})
                return
            name = (q.get("name") or ["upload.mp4"])[0]
            ok, msg = ui._receive_upload(name, self.rfile, length)
            self._json({"ok": ok, "msg": msg, "name": msg if ok else ""})

        # ---------------------------------------------------- MJPEG
        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=flyframe")
            self.end_headers()
            last = -1
            try:
                while True:
                    fid, jpg = ui._wait_frame(last)
                    if jpg is None or fid == last:
                        continue
                    last = fid
                    self.wfile.write(b"--flyframe\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

    return Handler


# ---------------------------------------------------------------- 页面
PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>flyscreen</title>
<style>
  :root{--bg:#0a0c11;--panel:#12161f;--line:#222a38;--fg:#c8d6e5;
        --dim:#6b7d94;--accent:#4ec3e0;--warm:#e08a3c;--rec:#e0483c}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:13px/1.5 ui-monospace,Consolas,"Courier New",monospace;
       display:flex;height:100vh;overflow:hidden}
  #left{flex:1;display:flex;flex-direction:column;min-width:0;padding:10px}
  #stage{flex:1;display:flex;align-items:center;justify-content:center;
         background:#05070a;border:1px solid var(--line);border-radius:6px;
         overflow:hidden;position:relative}
  #stage img{max-width:100%;max-height:100%;display:block}
  #bar{display:flex;gap:14px;align-items:center;padding:8px 4px 0;
       color:var(--dim);font-size:12px;flex-wrap:wrap}
  #bar b{color:var(--accent);font-weight:600}
  #bar .warm{color:var(--warm)}
  #right{width:330px;flex:none;background:var(--panel);border-left:1px solid var(--line);
         overflow-y:auto;padding:12px}
  h1{font-size:14px;margin:0 0 4px;color:var(--accent);letter-spacing:.5px}
  .sub{color:var(--dim);font-size:11px;margin-bottom:12px}
  .grp{margin:14px 0 6px;color:var(--dim);font-size:11px;text-transform:uppercase;
       letter-spacing:1px;border-bottom:1px solid var(--line);padding-bottom:3px}
  .row{margin:7px 0}
  .row label{display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px}
  .row .val{color:var(--accent)}
  input[type=range]{width:100%;accent-color:var(--accent);height:16px}
  select,button,input[type=text]{width:100%;background:#1a2130;color:var(--fg);
      border:1px solid var(--line);border-radius:4px;padding:6px;font:inherit;font-size:12px}
  button{cursor:pointer;margin-top:6px}
  button:hover{background:#222c3f;border-color:var(--accent)}
  button.rec{background:#2a1416;border-color:#5a2a28;color:#ff8a80}
  button.rec:hover{background:#3a1a1c;border-color:var(--rec)}
  button.live{background:#3a1a1c;border-color:var(--rec);color:#fff;animation:pulse 1.2s infinite}
  @keyframes pulse{50%{background:#5a2020}}
  .btns{display:flex;gap:6px}
  .btns button{flex:1}
  .sw{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
  .sw input{accent-color:var(--accent);width:15px;height:15px}
  #drop{border:2px dashed #2c3a4e;border-radius:8px;
        padding:22px 12px;text-align:center;color:var(--dim);cursor:pointer;
        transition:.15s;background:#0e1219}
  #drop:hover{border-color:var(--accent);color:var(--fg)}
  #drop.over{border-color:var(--warm);background:#1a1410;color:var(--warm);
             transform:scale(1.02)}
  #drop .big{font-size:22px;margin-bottom:6px;color:var(--accent)}
  #uprog{font-size:11px;color:var(--warm);margin-top:6px;word-break:break-all;min-height:16px}
  #recstat{font-size:11px;color:var(--rec);margin-top:6px;min-height:16px}
  .hint{font-size:11px;color:var(--dim)}
</style>
</head>
<body>
<div id="left">
  <div id="stage"><img id="view" src="/stream" alt="stream"></div>
  <div id="bar">
    <span>帧 <b id="s_frame">—</b></span>
    <span>fps <b id="s_fps">—</b></span>
    <span>脉冲/帧 <b id="s_spk">—</b></span>
    <span>点亮 <b id="s_lit">—</b></span>
    <span class="warm" id="s_body">—</span>
    <span style="margin-left:auto" id="s_video"></span>
  </div>
</div>
<div id="right">
  <h1>flyscreen</h1>
  <div class="sub">MaleCNS v1.0 脑 · MuJoCo 真果蝇身体</div>

  <div class="grp">片源</div>
  <div id="drop">
    <div class="big">⬇</div>
    <div>把视频文件拖进来</div>
    <div class="hint">或点击选择 · 任意格式</div>
  </div>
  <input type="file" id="file" accept="video/*,.mkv,.ts" hidden>
  <div id="uprog"></div>
  <div class="btns">
    <button id="b_pause">暂停</button>
    <button id="b_restart">重播并重新出片</button>
  </div>
  <div id="recstat"></div>

  <div id="controls"></div>
</div>

<script>
const PARAMS = __PARAMS__;
let state = {}, paused = false, recording = false;

const $ = id => document.getElementById(id);

/* ---------------- 拖拽上传 ---------------- */
const drop = $('drop');
['dragenter','dragover'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); ev.stopPropagation(); drop.classList.add('over');
}));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); ev.stopPropagation(); drop.classList.remove('over');
}));
drop.addEventListener('drop', ev => {
  const f = ev.dataTransfer.files;
  if (f && f.length) upload(f[0]);
});
drop.addEventListener('click', () => $('file').click());
$('file').addEventListener('change', e => {
  if (e.target.files.length) upload(e.target.files[0]);
});
// 整页也接受拖放，避免拖到面板外被浏览器直接打开
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files && e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
});

function upload(f) {
  const bar = $('uprog');
  const mb = (f.size / 1048576).toFixed(1);
  bar.textContent = `上传中 ${f.name}（${mb} MB）0%`;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload?name=' + encodeURIComponent(f.name));
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) bar.textContent =
      `上传中 ${f.name}（${mb} MB）${(100*e.loaded/e.total).toFixed(0)}%`;
  };
  xhr.onload = () => {
    try {
      const j = JSON.parse(xhr.responseText);
      bar.textContent = j.ok ? `✓ 已加载 ${j.name}` : `✗ ${j.msg}`;
    } catch(_) { bar.textContent = '✗ 返回异常'; }
  };
  xhr.onerror = () => { bar.textContent = '✗ 上传失败'; };
  xhr.send(f);
}

/* ---------------- 录制状态（全自动：拖进来就开始，放到片尾自动停）---------------- */
$('b_pause').onclick = async () => { await fetch('/api/pause'); };
$('b_restart').onclick = async () => { await fetch('/api/restart'); };

/* ---------------- 参数控件 ---------------- */
function el(tag, cls, txt){const e=document.createElement(tag);
  if(cls)e.className=cls; if(txt!==undefined)e.textContent=txt; return e;}

function buildControls(spec){
  const root = $('controls'); const groups = {};
  for (const [key,p] of Object.entries(spec)){
    const g = p.group || '其他';
    if(!groups[g]){ groups[g]=el('div'); root.appendChild(el('div','grp',g));
      root.appendChild(groups[g]); }
    const row = el('div','row');
    if(p.kind==='bool'){
      const lab=el('label','sw');
      const cb=document.createElement('input'); cb.type='checkbox'; cb.id='p_'+key;
      cb.onchange=()=>set(key, cb.checked?1:0);
      lab.appendChild(cb); lab.appendChild(el('span',null,p.label));
      row.appendChild(lab);
    } else if(p.kind==='select'){
      row.appendChild(el('label',null,p.label));
      const s=document.createElement('select'); s.id='p_'+key;
      for(const [v,t] of p.options){const o=document.createElement('option');
        o.value=v; o.textContent=t; s.appendChild(o);}
      s.onchange=()=>set(key, s.value);
      row.appendChild(s);
    } else {
      const lab=el('label'); lab.appendChild(el('span',null,p.label));
      const val=el('span','val',''); val.id='v_'+key; lab.appendChild(val);
      row.appendChild(lab);
      const r=document.createElement('input'); r.type='range'; r.id='p_'+key;
      r.min=p.lo; r.max=p.hi; r.step=p.step;
      r.oninput=()=>{val.textContent=(+r.value).toFixed(p.kind==='int'?0:2);};
      r.onchange=()=>set(key, r.value);
      row.appendChild(r);
    }
    groups[g].appendChild(row);
  }
}

function fill(spec, vals){
  for(const [key,p] of Object.entries(spec)){
    const v = vals[key];
    if(p.kind==='bool'){ const e=$('p_'+key); if(e)e.checked=!!v; }
    else if(p.kind==='select'){ const e=$('p_'+key); if(e)e.value=v; }
    else { const e=$('p_'+key); const t=$('v_'+key);
      if(e && document.activeElement!==e){ e.value=v; }
      if(t) t.textContent=(+v).toFixed(p.kind==='int'?0:2); }
  }
}

async function set(k,v){ await fetch(`/api/set?k=${encodeURIComponent(k)}&v=${encodeURIComponent(v)}`); }

/* ---------------- 轮询状态 ---------------- */
function fmtSec(s){ s=Math.max(0,Math.round(s)); return `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`; }

async function poll(){
  try{
    const s = await (await fetch('/api/state')).json();
    if(!state.params) buildControls(PARAMS);
    fill(PARAMS, s.params);
    const st = s.status || {};
    $('s_frame').textContent = st.frame ?? '—';
    $('s_fps').textContent   = st.fps ?? '—';
    $('s_spk').textContent   = (st.spikes??0).toLocaleString();
    $('s_lit').textContent   = (st.lit??0).toLocaleString();
    $('s_body').textContent  = st.body || '';
    $('s_video').textContent = st.video || '（拖个视频进来）';
    const r = s.record || {};
    recording = !!r.on;
    $('recstat').textContent = r.on
      ? `● 正在出片 ${fmtSec(r.seconds)} · ${(r.frames??0).toLocaleString()} 帧 → ${r.path||''}`
      : (r.path ? `✓ 已出片：${r.path}` : '拖入视频即自动预览 + 出片');
    state = s;
  }catch(e){ /* 服务未就绪 */ }
  setTimeout(poll, 600);
}
poll();
</script>
</body>
</html>
"""
