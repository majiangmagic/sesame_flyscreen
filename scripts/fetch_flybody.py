"""下载 MuJoCo 官方的 flybody 果蝇身体模型（mujoco_menagerie/flybody）。

这是 DeepMind + Janelia 那篇 Nature 2025 全身果蝇仿真用的同一套 MJCF 资产。
只需要 mujoco 就能加载，不依赖 flygym / flybody 这两个 python 包。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

OWNER = "google-deepmind"
REPO = "mujoco_menagerie"
SUBDIR = "flybody"
DEST = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\fly-brain\flyscreen\data\assets\flybody")

API = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/main?recursive=1"
RAW = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/main/"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "flyscreen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    print("listing repo tree ...")
    tree = get_json(API).get("tree", [])
    files = [t for t in tree if t["type"] == "blob" and t["path"].startswith(SUBDIR + "/")]
    if not files:
        print("!! 没有找到 flybody 目录")
        return 1
    total = sum(f.get("size", 0) for f in files)
    print(f"found {len(files)} files, {total/1e6:.1f} MB")

    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, f in enumerate(files, 1):
        rel = f["path"][len(SUBDIR) + 1 :]
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size == f.get("size", -1):
            ok += 1
            continue
        try:
            req = urllib.request.Request(RAW + f["path"], headers={"User-Agent": "flyscreen/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out.write_bytes(r.read())
            ok += 1
            print(f"  [{i}/{len(files)}] {rel}")
        except Exception as e:
            print(f"  [{i}/{len(files)}] FAIL {rel}: {e}")
    print(f"done: {ok}/{len(files)} files -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
