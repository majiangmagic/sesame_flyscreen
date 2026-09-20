"""下载 MaleCNS v1.0 所需的原始数据。

    python -m flyscreen.data.fetch            # 只下必需的两个（约 1.06 GB）
    python -m flyscreen.data.fetch --all      # 连 syn-partners 一起下（+6.8 GB，可选）

必需：
    body-annotations.feather    14 MB   胞体坐标 / 类型 / 左右
    connectome-weights.feather  1.0 GB  突触连接（body_pre, body_post, weight）

可选：
    syn-partners.feather        6.8 GB  每个突触的前后位点坐标（做突触级渲染才需要）
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from ..config import MALECNS_BUCKET, MALECNS_FILES, RAW_DIR


def download(remote: str, dest: Path, chunk: int = 1 << 22) -> None:
    url = f"{MALECNS_BUCKET}/{remote}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "flyscreen/0.1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        done = have
        with tmp.open(mode) as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                pct = 100.0 * done / total if total else 0
                print(
                    f"\r  {dest.name}  {done/1e6:8.1f} / {total/1e6:8.1f} MB  {pct:5.1f}%",
                    end="",
                    flush=True,
                )
    print()
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="连可选的 syn-partners 也下")
    ap.add_argument("--only", default=None, choices=sorted(MALECNS_FILES))
    a = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    keys = [a.only] if a.only else (
        list(MALECNS_FILES) if a.all else ["annotations", "connectome_weights"]
    )
    print(f"目标目录 {RAW_DIR}")
    for k in keys:
        remote, local = MALECNS_FILES[k]
        dest = RAW_DIR / local
        if dest.exists():
            print(f"  已存在 {local} ({dest.stat().st_size/1e6:.1f} MB)，跳过")
            continue
        print(f"  下载 {k}: {remote}")
        try:
            download(remote, dest)
        except Exception as e:  # noqa: BLE001
            print(f"  !! 失败 {k}: {type(e).__name__}: {e}")
            return 1
    print("完成。接下来：python -m flyscreen.data.prepare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
