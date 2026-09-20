"""pygame 预览窗：左边是果蝇脑点云（视频在脑上播放），右边是 MuJoCo 果蝇身体。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

MARGIN = 14
HUD_H = 28


@dataclass
class UiState:
    paused: bool = False
    show_body: bool = True
    show_hud: bool = True
    quit: bool = False
    step_once: bool = False
    # 交互产生的待处理动作
    actions: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.actions is None:
            self.actions = []


class Preview:
    def __init__(self, cfg: Config, title: str = "flyscreen") -> None:
        import pygame

        self.pygame = pygame
        self.cfg = cfg
        pygame.init()
        pygame.display.set_caption(title)

        r = cfg.render
        b = cfg.body
        body_w = b.width if b.enabled else 0
        self.win_w = r.canvas_w + body_w + (3 * MARGIN if body_w else 2 * MARGIN)
        self.win_h = max(r.canvas_h, b.height if b.enabled else 0) + 2 * MARGIN + HUD_H
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))

        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
        self.font_big = pygame.font.SysFont("consolas,couriernew,monospace", 17, bold=True)
        self.state = UiState()

        self._canvas_rect = pygame.Rect(MARGIN, MARGIN, r.canvas_w, r.canvas_h)
        self._body_rect = pygame.Rect(
            MARGIN * 2 + r.canvas_w, MARGIN, b.width, b.height
        )

    # ------------------------------------------------------------ 绘制
    def _blit(self, img: np.ndarray, rect) -> None:
        pg = self.pygame
        surf = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        if surf.get_size() != (rect.width, rect.height):
            surf = pg.transform.smoothscale(surf, (rect.width, rect.height))
        self.screen.blit(surf, rect.topleft)

    def draw_surface(self, comp: np.ndarray) -> None:
        """显示一帧已经拼好的 RGB 图（和离线出片走同一条路径）。"""
        pg = self.pygame
        surf = pg.surfarray.make_surface(np.ascontiguousarray(comp.swapaxes(0, 1)))
        self.screen.blit(surf, (0, 0))
        pg.display.flip()

    # ------------------------------------------------------------ 事件
    def poll(self) -> UiState:
        pg = self.pygame
        st = self.state
        for e in pg.event.get():
            if e.type == pg.QUIT:
                st.quit = True
            elif e.type == pg.KEYDOWN:
                k = e.key
                if k in (pg.K_ESCAPE, pg.K_q):
                    st.quit = True
                elif k == pg.K_SPACE:
                    st.paused = not st.paused
                elif k == pg.K_PERIOD:
                    st.step_once = True
                elif k == pg.K_h:
                    st.show_hud = not st.show_hud
                elif k == pg.K_b:
                    st.show_body = not st.show_body
                elif k == pg.K_v:
                    st.actions.append("cycle_view")
                elif k == pg.K_g:
                    st.actions.append("toggle_glow")
                elif k in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    st.actions.append("gain_up")
                elif k in (pg.K_MINUS, pg.K_KP_MINUS):
                    st.actions.append("gain_down")
                elif k == pg.K_LEFT:
                    st.actions.append("decay_down")
                elif k == pg.K_RIGHT:
                    st.actions.append("decay_up")
                elif k in (pg.K_s, pg.K_F12):
                    st.actions.append("screenshot")
        return st

    def take_actions(self) -> list[str]:
        a = self.state.actions
        self.state.actions = []
        return a

    def close(self) -> None:
        self.pygame.quit()
