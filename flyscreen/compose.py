"""画面布局：把脑点云 + 果蝇身体拼成一帧 RGB。预览和离线出片走同一条路径。

性能注意：面板尺寸**故意和渲染输出的尺寸完全一致**。原来为了画边框把内区
缩了 2 像素，结果每帧都走「先横向重采样再纵向重采样」的双重 gather 分支，
白白多花 ~20 ms。现在边框是 blit 完之后直接覆盖最外圈像素，尺寸对得上，
走的是一次 memcpy。
"""
from __future__ import annotations

import numpy as np

from .config import Config

# 和 RenderConfig.background 保持一致，省掉一次全图 maximum
BG = (4, 6, 12)
BORDER = (30, 42, 60)
MARGIN = 14


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """把图放进 w x h 的框里（等比缩放 + 居中留边），返回新数组。"""
    ih, iw = img.shape[:2]
    if ih == h and iw == w:
        return img                                    # 尺寸刚好，直接用
    s = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * s))), max(1, int(round(iw * s)))
    yi = np.clip((np.arange(nh) / s).astype(np.int32), 0, ih - 1)
    xi = np.clip((np.arange(nw) / s).astype(np.int32), 0, iw - 1)
    small = img[yi][:, xi]
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:, :] = BG
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = small
    return out


def _panel(img: np.ndarray, w: int, h: int, key_black: bool) -> np.ndarray:
    """画一个带边框的面板。"""
    if img.shape[0] == h and img.shape[1] == w:
        panel = np.array(img, dtype=np.uint8, copy=True)
        if key_black:
            np.maximum(panel, np.asarray(BG, dtype=np.uint8), out=panel)
    else:
        panel = _fit(img, w, h)
    panel[0, :] = BORDER
    panel[-1, :] = BORDER
    panel[:, 0] = BORDER
    panel[:, -1] = BORDER
    return panel


def layout_size(cfg: Config) -> tuple[int, int]:
    r, b = cfg.render, cfg.body
    body_w = b.width if b.enabled else 0
    w = r.canvas_w + body_w + (3 * MARGIN if body_w else 2 * MARGIN)
    h = max(r.canvas_h, b.height if b.enabled else 0) + 2 * MARGIN
    return int(w), int(h)


def compose(
    canvas_img: np.ndarray,
    body_img: np.ndarray | None,
    cfg: Config,
    hud_strip: np.ndarray | None = None,
) -> np.ndarray:
    """拼帧。hud_strip 是额外的底部信息条（H,W,3），可省。"""
    r, b = cfg.render, cfg.body
    body_w = b.width if (b.enabled and body_img is not None) else 0

    total_w = r.canvas_w + body_w + (3 * MARGIN if body_w else 2 * MARGIN)
    body_h = b.height if body_w else 0
    row_h = max(r.canvas_h, body_h)
    total_h = row_h + 2 * MARGIN

    out = np.empty((total_h, total_w, 3), dtype=np.uint8)
    out[:, :] = BG

    # 两块面板在行内**竖直居中**。点云面板通常比身体面板矮（点云的解剖包围盒
    # 是横的），不居中就会顶在上面、下面留一大块黑。
    x = MARGIN
    y = MARGIN + (row_h - r.canvas_h) // 2
    out[y : y + r.canvas_h, x : x + r.canvas_w] = _panel(
        canvas_img, r.canvas_w, r.canvas_h, key_black=False
    )
    x += r.canvas_w + MARGIN

    if body_w:
        y = MARGIN + (row_h - b.height) // 2
        out[y : y + b.height, x : x + body_w] = _panel(
            body_img, body_w, b.height, key_black=True  # type: ignore[arg-type]
        )

    if hud_strip is not None and hud_strip.size:
        out = np.concatenate([out, hud_strip], axis=0)
    return np.ascontiguousarray(out)


def make_hud_strip(lines: list[str], width: int, height: int = 30) -> np.ndarray:
    """用 pygame 画一条文字信息栏（headless 下也能用）。"""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.font.init()
    strip = pygame.Surface((width, height))
    strip.fill(BG)
    font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
    x = MARGIN
    for line in lines:
        s = font.render(line, True, (150, 200, 225))
        strip.blit(s, (x, (height - s.get_height()) // 2))
        x += s.get_width() + 22
    arr = pygame.surfarray.array3d(strip)  # (W,H,3)
    return np.ascontiguousarray(arr.swapaxes(0, 1))
