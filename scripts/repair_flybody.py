"""补齐 flybody 缺失/损坏的 mesh（按 GitHub 上的大小校验）。"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

OWNER, REPO, SUB = "google-deepmind", "mujoco_menagerie", "flybody"
DEST = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\fly-brain\flyscreen\data\assets\flybody")
API = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/main?recursive=1"
RAW = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/main/"


def get(url: str, tries: int = 6) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flyscreen/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def main() -> int:
    req = urllib.request.Request(API, headers={"User-Agent": "flyscreen/0.1"})
    tree = json.loads(urllib.request.urlopen(req, timeout=60).read())["tree"]
    files = [t for t in tree if t["type"] == "blob" and t["path"].startswith(SUB + "/")]
    print(f"{len(files)} 个文件待核对")

    bad: list[str] = []
    for f in files:
        rel = f["path"][len(SUB) + 1 :]
        out = DEST / rel
        want = f.get("size", -1)
        if not out.exists() or out.stat().st_size != want:
            bad.append(rel)
    print(f"缺失/不完整: {len(bad)} -> {bad}")

    for rel in bad:
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = get(RAW + SUB + "/" + rel)
            out.write_bytes(data)
            print(f"  OK   {rel}  {len(data):,} bytes")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {rel}: {e}")

    still = [
        f["path"][len(SUB) + 1 :]
        for f in files
        if not (DEST / f["path"][len(SUB) + 1 :]).exists()
        or (DEST / f["path"][len(SUB) + 1 :]).stat().st_size != f.get("size", -1)
    ]
    print(f"仍缺: {len(still)} {still}")
    return 1 if still else 0


if __name__ == "__main__":
    raise SystemExit(main())
