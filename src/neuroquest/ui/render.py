"""Drawing only: reads a GameState (plus optional UI-side effects) and paints it. Never changes the state.

Look ("cartoon world", entry 13): a garden maze of rounded hedges on a sandy path under a pastel
sky, a smiling yellow explorer, pink spiky hunters, blue gems and a wooden gate; white rounded
cards for the HUD, the panel and the screens. Everything is drawn with primitives.

Layout: top HUD · board (centred in the remaining area, optionally sharing it with the
intelligence panel on the right) · bottom bar. Bar heights shrink on small windows. Colours and
helpers come from ``ui/theme.py``; the base colours are kept at cell centres so the pixel tests
can verify what is drawn where.
"""

from __future__ import annotations

import math
import random

import pygame

from neuroquest.ai.astar import a_star
from neuroquest.config import DifficultyParams
from neuroquest.core.maze import Cell, Maze
from neuroquest.core.rules import min_enemy_distance, time_left_s
from neuroquest.core.state import GameState, Outcome
from neuroquest.ui import theme as th
from neuroquest.ui.theme import SEP, card, icon_chip, icon_crystal, icon_dot, icon_drone, icon_heart, pill, pulse, ring, shadow, text

# ---------- identity colours (exported; tests check these at cell centres) ----------
BG = th.BG  # sky
FOOTER_BG = (255, 251, 240)  # cream ground band
HUD_BG = th.BG
PLAYER = th.AMBER  # the yellow explorer's body
PLAYER_DARK = (230, 170, 60)
PLAYER_HIT = (255, 240, 200)
ENEMY = th.CRIMSON  # the pink hunters
ENEMY_DARK = (205, 70, 100)
ENEMY_EYE = (255, 255, 255)
REWARD = (90, 209, 255)  # gem
REWARD_DARK = (46, 160, 220)
EXIT = (201, 138, 75)  # the wooden gate
EXIT_DARK = (150, 98, 48)
FLAG = (120, 205, 110)
HEDGE, HEDGE_DARK, HEDGE_LEAF, HEDGE_LIGHT = (124, 201, 107), (86, 160, 76), (158, 222, 136), (190, 240, 170)
SAND, SAND_DOT = (247, 232, 190), (226, 206, 150)
BOARD_FRAME = (255, 251, 240)
VISITED = (232, 210, 160)  # footprints on the path
INK = th.TEXT
LABEL_COLOURS = {"aggressive": th.CRIMSON, "cautious": th.CYAN, "explorer": th.EMERALD}
HUD_HEIGHT = 72
FOOTER_HEIGHT = 56
PANEL_WIDTH = 280
MARGIN = 16

_BOARDS: dict[tuple, pygame.Surface] = {}


# ---------- layout ----------


def bars(surface_size: tuple[int, int]) -> tuple[int, int]:
    """(top HUD height, bottom bar height) for this window; compact below 600 px tall."""
    return (HUD_HEIGHT, FOOTER_HEIGHT) if surface_size[1] >= 600 else (48, 40)


def panel_width(surface_size: tuple[int, int], fx: dict | None) -> int:
    """The intelligence panel takes a fixed width on the right when open and the window allows it."""
    if fx and fx.get("panel_open") and surface_size[0] >= 900 and surface_size[1] >= 600:
        return PANEL_WIDTH
    return 0


def layout(maze: Maze, surface_size: tuple[int, int], panel: int = 0) -> tuple[int, int, int]:
    """Cell size in pixels and the (x, y) offset that centres the board in the area left for it."""
    width, height = surface_size
    hud_h, foot_h = bars(surface_size)
    usable_w = width - panel - 2 * MARGIN
    usable_h = height - hud_h - foot_h - 2 * MARGIN
    size = max(4, min(usable_w // maze.width, usable_h // maze.height))
    ox = MARGIN + (usable_w - maze.width * size) // 2
    oy = hud_h + MARGIN + (usable_h - maze.height * size) // 2
    return size, ox, oy


def cell_rect(maze: Maze, surface_size: tuple[int, int], cell: Cell, panel: int = 0) -> pygame.Rect:
    size, ox, oy = layout(maze, surface_size, panel)
    return pygame.Rect(ox + cell[1] * size, oy + cell[0] * size, size, size)


def _centre(cell: Cell, size: int, ox: int, oy: int) -> tuple[int, int]:
    return (ox + cell[1] * size + size // 2, oy + cell[0] * size + size // 2)


# ---------- main entry ----------


def draw(
    surface: pygame.Surface,
    state: GameState,
    params: DifficultyParams,
    font: pygame.font.Font,
    prediction=None,
    notice: str | None = None,
    fx: dict | None = None,
    paused: bool = False,
    menu: bool = False,
) -> None:
    """Paint the whole frame. ``fx`` (optional, owned by the app) carries time, particles, shake,
    the trail with recency, the flash, the facing direction, the AI-vision and panel toggles."""
    fx = fx or {}
    t = float(fx.get("t", 0.0))
    size_px = surface.get_size()
    _draw_sky(surface)
    panel = panel_width(size_px, fx)
    size, ox, oy = layout(state.maze, size_px, panel)
    shake = float(fx.get("shake", 0.0))
    if shake > 0:
        ox += int(random.uniform(-1, 1) * shake * size * 0.25)
        oy += int(random.uniform(-1, 1) * shake * size * 0.25)

    board = pygame.Rect(ox, oy, state.maze.width * size, state.maze.height * size)
    frame = board.inflate(min(28, size), min(28, size))
    _frame_shadow(surface, frame)
    pygame.draw.rect(surface, BOARD_FRAME, frame, border_radius=th.RADIUS["lg"])
    _draw_board(surface, state, size, ox, oy, fx)
    if fx.get("show_paths"):
        _draw_enemy_paths(surface, state, size, ox, oy)
    for cell in state.rewards:
        _draw_reward(surface, _centre(cell, size, ox, oy), size, t)
    _draw_exit(surface, _centre(state.exit, size, ox, oy), size, t)
    near = min_enemy_distance(state.player.pos, state.enemies) if state.enemies else 10**9
    for enemy in state.enemies:
        _draw_enemy(surface, _centre(enemy.pos, size, ox, oy), size, t, enemy.id, state)
    _draw_player(surface, state, _centre(state.player.pos, size, ox, oy), size, t, fx)
    _draw_particles(surface, fx)

    flash = float(fx.get("flash", 0.0))
    warn = (near <= 3 and not state.outcome.is_over and pulse(t, 6.0) > 0.5) or flash > 0
    if warn:  # proximity / hit warning: a soft pink frame around the board, nothing over the play area
        pygame.draw.rect(surface, ENEMY, frame, width=4, border_radius=th.RADIUS["lg"])

    _draw_hud(surface, state, params, fx, t)
    _draw_footer(surface, state, params, prediction, notice, fx, paused)
    if panel:
        from neuroquest.ui.panel import draw_panel  # local import: panel.py imports this module's constants

        draw_panel(surface, pygame.Rect(surface.get_width() - panel, bars(size_px)[0], panel, surface.get_height() - sum(bars(size_px))), state, params, prediction, fx, t)
    if menu:
        _draw_menu(surface, state, fx, t)
    elif state.outcome.is_over:
        _draw_overlay(surface, state, params, prediction, fx)
    elif paused:
        _draw_pause(surface, fx)
    elif state.tick < 30 and fx.get("intro", False):
        _draw_intro(surface, oy)


# ---------- sky and board ----------


def _draw_sky(surface: pygame.Surface) -> None:
    """Flat pastel sky with two soft hills low down. The left margin and the top-left area stay plain
    sky: the pixel tests sample them."""
    w, h = surface.get_size()
    surface.fill(BG)
    pygame.draw.ellipse(surface, (203, 236, 180), pygame.Rect(-w // 4, int(h * 0.80), int(w * 0.9), int(h * 0.5)))
    pygame.draw.ellipse(surface, (184, 228, 160), pygame.Rect(int(w * 0.45), int(h * 0.84), int(w * 0.8), int(h * 0.45)))


def _frame_shadow(surface: pygame.Surface, frame: pygame.Rect) -> None:
    """The board's drop shadow (the theme's cached shadow, a little deeper than a card's)."""
    shadow(surface, frame, th.RADIUS["lg"], dy=8, alpha=45)


def _board_surface(maze: Maze, size: int) -> pygame.Surface:
    """Static tiles rendered once per maze and cell size: a sandy path with pebbles, and hedges drawn
    as rounded blocks whose junctions with neighbouring hedge cells are filled, so walls are continuous."""
    key = (id(maze), size)
    if key in _BOARDS:
        return _BOARDS[key]
    w, h = maze.width * size, maze.height * size
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    rng = random.Random(11)
    is_wall = lambda r, c: maze.in_bounds((r, c)) and not maze.is_open((r, c))  # noqa: E731
    radius = max(3, size // 3)
    for r, row in enumerate(maze.grid):
        for c, open_ in enumerate(row):
            if open_:
                rect = pygame.Rect(c * size, r * size, size, size)
                pygame.draw.rect(surf, SAND, rect)
                for _ in range(3):
                    pygame.draw.circle(surf, SAND_DOT, (rect.x + rng.randint(4, max(5, size - 4)), rect.y + rng.randint(4, max(5, size - 4))), max(1, size // 24))
    walls = [(r, c) for r in range(maze.height) for c in range(maze.width) if is_wall(r, c)]
    lift = int(size * 0.12)
    inset = int(size * 0.18)

    def top_of(rect: pygame.Rect) -> pygame.Rect:
        return rect.inflate(-inset, -inset).move(0, -lift)

    def block(colour, rect: pygame.Rect, r: int, c: int, target: pygame.Surface) -> None:
        pygame.draw.rect(target, colour, rect, border_radius=radius)
        if is_wall(r, c + 1):
            pygame.draw.rect(target, colour, pygame.Rect(rect.centerx, rect.y, size, rect.h))
        if is_wall(r + 1, c):
            pygame.draw.rect(target, colour, pygame.Rect(rect.x, rect.centery, rect.w, size))

    shade = pygame.Surface((w, h), pygame.SRCALPHA)
    for r, c in walls:
        block((60, 80, 40, 55), pygame.Rect(c * size, r * size, size, size).move(0, int(size * 0.18)), r, c, shade)
    surf.blit(shade, (0, 0))
    for r, c in walls:
        block(HEDGE_DARK, pygame.Rect(c * size, r * size, size, size), r, c, surf)
    for r, c in walls:
        block(HEDGE, top_of(pygame.Rect(c * size, r * size, size, size)), r, c, surf)
    for r, c in walls:
        top = top_of(pygame.Rect(c * size, r * size, size, size))
        if not is_wall(r - 1, c) and top.h > 8:
            pygame.draw.rect(surf, HEDGE_LIGHT, pygame.Rect(top.x + 3, top.y + 3, top.w - 6, max(2, top.h // 4)), border_radius=max(2, radius // 2))
        if size >= 14:
            for _ in range(3):
                pygame.draw.circle(surf, HEDGE_LEAF, (top.x + rng.randint(4, max(5, top.w - 4)), top.y + rng.randint(5, max(6, top.h - 3))), max(1, size // 14))
    if len(_BOARDS) > 8:
        _BOARDS.clear()
    _BOARDS[key] = surf
    return surf


def _draw_board(surface: pygame.Surface, state: GameState, size: int, ox: int, oy: int, fx: dict) -> None:
    surface.blit(_board_surface(state.maze, size), (ox, oy))
    trail: dict = fx.get("trail", {})
    tick = state.tick
    r_dot = max(1, size // 10)
    for cell in state.visited:
        if cell == state.player.pos:
            continue
        age = tick - trail.get(cell, -10**6)
        strength = 0.6 if not trail else max(0.25, 1.0 - age / 400.0)
        colour = th.mix(SAND, VISITED, strength)
        cx, cy = _centre(cell, size, ox, oy)
        pygame.draw.circle(surface, colour, (cx - r_dot, cy + r_dot // 2), r_dot)
        pygame.draw.circle(surface, colour, (cx + r_dot, cy - r_dot // 2), r_dot)


def _draw_enemy_paths(surface: pygame.Surface, state: GameState, size: int, ox: int, oy: int) -> None:
    """AI vision: the next cells of each hunter's shortest path to the player as a fading dotted trail."""
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for enemy in state.enemies:
        path = a_star(state.maze, enemy.pos, state.player.pos)
        if not path:
            continue
        pts = [_centre(cell, size, ox, oy) for cell in path[:13]]
        for i, p in enumerate(pts[1:]):
            pygame.draw.circle(layer, (*ENEMY, int(210 * (1 - i / 13))), p, max(2, size // 7))
            pygame.draw.circle(layer, (255, 255, 255, int(160 * (1 - i / 13))), p, max(1, size // 14))
    surface.blit(layer, (0, 0))


# ---------- things ----------


_GROUND: dict[tuple[int, int], pygame.Surface] = {}


def _ground_shadow(surface: pygame.Surface, cx: int, bottom: int, w: int, h: int) -> None:
    """The soft ellipse under a gem or a character; one surface per size, reused every frame."""
    layer = _GROUND.get((w, h))
    if layer is None:
        layer = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
        pygame.draw.ellipse(layer, (120, 100, 60, 60), pygame.Rect(1, 1, w, h))
        if len(_GROUND) > 32:
            _GROUND.clear()
        _GROUND[(w, h)] = layer
    surface.blit(layer, (cx - w // 2 - 1, bottom - h))


def _face(surface: pygame.Surface, cx: int, cy: int, r: int, mood: str, look: tuple[float, float]) -> None:
    """Eyes above the centre, mouth below it: the centre pixel always stays the body colour."""
    eye_r = max(2, r // 4)
    ex, ey = int(r * 0.36), cy - max(eye_r + 2, r // 2)
    for sx in (-1, 1):
        pygame.draw.circle(surface, (255, 255, 255), (cx + sx * ex, ey), eye_r)
        pygame.draw.circle(surface, INK, (cx + sx * ex + int(look[0] * eye_r * 0.4), ey + int(look[1] * eye_r * 0.4)), max(1, eye_r // 2))
    mouth_top = cy + max(2, r // 5)
    if mood == "happy":
        pygame.draw.arc(surface, INK, pygame.Rect(cx - int(r * 0.3), mouth_top - int(r * 0.15), int(r * 0.6) + 1, int(r * 0.4) + 1), math.pi, 2 * math.pi, max(1, r // 9))
        if r >= 8:
            for sx in (-1, 1):
                pygame.draw.circle(surface, (255, 170, 160), (cx + sx * int(r * 0.62), cy + max(1, r // 8)), max(1, r // 8))
    else:  # angry
        for sx in (-1, 1):
            pygame.draw.line(surface, INK, (cx + sx * int(r * 0.55), ey - eye_r - max(2, r // 5)), (cx + sx * int(r * 0.12), ey - eye_r - 1), max(1, r // 8))
        pygame.draw.arc(surface, INK, pygame.Rect(cx - int(r * 0.3), mouth_top, int(r * 0.6) + 1, int(r * 0.35) + 1), 0, math.pi, max(1, r // 9))
        if r >= 8:
            for sx in (-1, 1):
                fx_ = cx + sx * int(r * 0.22)
                pygame.draw.polygon(surface, (255, 255, 255), [(fx_ - max(1, r // 8), mouth_top + 1), (fx_ + max(1, r // 8), mouth_top + 1), (fx_, mouth_top + max(3, r // 3))])


def _draw_reward(surface: pygame.Surface, centre: tuple[int, int], size: int, t: float) -> None:
    """Blue gem: a two-tone diamond with an off-centre sparkle; bobs gently over a small shadow."""
    s = max(4, int(size * 0.34))
    cx, cy = centre[0], centre[1] + int(math.sin(t * 3.0 + centre[0] * 0.05) * size * 0.04)
    _ground_shadow(surface, cx, centre[1] + size // 2 - max(1, size // 10), int(s * 1.2), max(2, int(s * 0.35)))
    pygame.draw.polygon(surface, REWARD_DARK, [(cx, cy - s), (cx + int(s * 0.75), cy), (cx, cy + s), (cx - int(s * 0.75), cy)])
    pygame.draw.polygon(surface, REWARD, [(cx, cy - int(s * 0.72)), (cx + int(s * 0.45), cy - max(1, int(s * 0.05))), (cx, cy + int(s * 0.55)), (cx - int(s * 0.45), cy - max(1, int(s * 0.05)))])
    pygame.draw.circle(surface, (255, 255, 255), (cx - max(2, int(s * 0.2)), cy - max(3, int(s * 0.42))), max(1, int(s * 0.14)))


def _draw_exit(surface: pygame.Surface, centre: tuple[int, int], size: int, t: float) -> None:
    """A little wooden gate with a flag: the way out. The gate's centre pixel is wood."""
    cx, cy = centre
    gate = pygame.Rect(0, 0, max(6, int(size * 0.7)), max(6, int(size * 0.74)))
    gate.center = (cx, cy + int(size * 0.08))
    pygame.draw.rect(surface, EXIT_DARK, gate.inflate(4, 4), border_radius=max(2, size // 4))
    pygame.draw.rect(surface, EXIT, gate, border_radius=max(2, size // 4))
    door = pygame.Rect(0, 0, max(3, int(gate.w * 0.4)), max(3, int(gate.h * 0.42)))
    door.midbottom = (gate.centerx, gate.bottom - max(1, size // 20))
    pygame.draw.rect(surface, (255, 244, 214), door, border_top_left_radius=max(2, size // 5), border_top_right_radius=max(2, size // 5))
    if size >= 12:
        pole_top = gate.y - int(size * 0.28)
        pygame.draw.line(surface, EXIT_DARK, (gate.centerx, pole_top), (gate.centerx, gate.y), max(1, size // 14))
        wave = int(math.sin(t * 4.0) * size * 0.03)
        pygame.draw.polygon(surface, FLAG, [(gate.centerx, pole_top), (gate.centerx + int(size * 0.24), pole_top + int(size * 0.08) + wave), (gate.centerx, pole_top + int(size * 0.17))])


def _draw_enemy(surface: pygame.Surface, centre: tuple[int, int], size: int, t: float, idx: int, state: GameState) -> None:
    """Pink hunter: a round blob with spiky hair, an angry face looking at the player, and a '!' when it is
    within 3 cells (driven by the real distance)."""
    pr, pc = state.player.pos
    enemy = next((e for e in state.enemies if e.id == idx), None)
    er, ec = enemy.pos if enemy else (pr, pc)
    dist = abs(pr - er) + abs(pc - ec)
    dx, dy = (pc - ec), (pr - er)
    norm = math.hypot(dx, dy) or 1.0
    r = max(4, int(size * 0.34))
    cx, cy = centre[0], centre[1] + int(math.sin(t * 5.0 + idx) * size * 0.04)
    _ground_shadow(surface, centre[0], centre[1] + size // 2 - max(1, size // 12), int(r * 1.8), max(2, int(r * 0.5)))
    if size >= 12:
        for k in range(5):  # spiky hair
            ang = -math.pi / 2 + (k - 2) * 0.45
            pygame.draw.polygon(surface, ENEMY_DARK, [(cx + math.cos(ang - 0.2) * r * 0.9, cy + math.sin(ang - 0.2) * r * 0.9), (cx + math.cos(ang) * r * 1.38, cy + math.sin(ang) * r * 1.38), (cx + math.cos(ang + 0.2) * r * 0.9, cy + math.sin(ang + 0.2) * r * 0.9)])
    pygame.draw.circle(surface, ENEMY_DARK, (cx, cy + 2), r + 2)
    pygame.draw.circle(surface, ENEMY, (cx, cy), r)
    _face(surface, cx, cy, r, "angry", (dx / norm, dy / norm))
    if dist <= 3:
        bx, by = cx + int(r * 0.9), cy - int(r * 1.5)
        pygame.draw.circle(surface, (255, 255, 255), (bx, by), max(4, r // 2))
        pygame.draw.circle(surface, ENEMY_DARK, (bx, by), max(4, r // 2), width=1)
        mark = text(max(9, r), "!", ENEMY_DARK, True)
        surface.blit(mark, (bx - mark.get_width() // 2, by - mark.get_height() // 2))


def _draw_player(surface: pygame.Surface, state: GameState, centre: tuple[int, int], size: int, t: float, fx: dict) -> None:
    """Yellow explorer: a round body with feet, an antenna tipped in gem blue, and a happy face whose eyes look
    where you move; bobs as it walks and blinks pale while invulnerable."""
    invulnerable = state.tick < state.player.invulnerable_until
    hit_blink = invulnerable and (state.tick // 2) % 2 == 0
    body = PLAYER_HIT if hit_blink else PLAYER
    r = max(4, int(size * 0.36))
    bob = int(abs(math.sin(t * 8.0)) * size * 0.05)
    cx, cy = centre[0], centre[1] - bob
    bottom = centre[1] + size // 2 - max(1, size // 12)
    _ground_shadow(surface, centre[0], bottom, r * 2, max(2, int(r * 0.55)))
    if size >= 12:
        for sx in (-1, 1):  # feet
            pygame.draw.ellipse(surface, PLAYER_DARK, pygame.Rect(cx + sx * int(r * 0.45) - int(r * 0.3), bottom - int(r * 0.45), max(3, int(r * 0.6)), max(2, int(r * 0.4))))
    pygame.draw.circle(surface, PLAYER_DARK, (cx, cy + 2), r + 2)
    pygame.draw.circle(surface, body, (cx, cy), r)
    pygame.draw.circle(surface, (255, 235, 170), (cx - r // 3, cy - r // 3), max(1, r // 4))  # highlight, off-centre
    if size >= 12:  # antenna
        tip = (cx + int(r * 0.25), cy - int(r * 1.4))
        pygame.draw.line(surface, PLAYER_DARK, (cx, cy - r + 1), tip, max(1, size // 16))
        pygame.draw.circle(surface, (255, 110, 110) if invulnerable else REWARD, tip, max(2, r // 5))
    fr, fc = fx.get("facing", (0, 1))
    _face(surface, cx, cy, r, "happy", (fc, fr))


def _draw_particles(surface: pygame.Surface, fx: dict) -> None:
    particles = fx.get("particles") or []
    if not particles:
        return
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for p in particles:
        alpha = int(255 * max(0.0, min(1.0, p["life"])))
        pygame.draw.circle(layer, (*p["colour"], alpha), (int(p["x"]), int(p["y"])), max(1, int(p["size"])))
    surface.blit(layer, (0, 0))


# ---------- top HUD ----------


def _stat_card(surface: pygame.Surface, x: int, y: int, h: int, label: str, value_surf: pygame.Surface, icon=None, min_w: int = 0) -> int:
    """One floating HUD card: small caps label above a value, optional icon on the left. Returns the next x."""
    pad = th.SP["md"]
    icon_w = 22 if icon else 0
    w = max(min_w, pad + icon_w + max(value_surf.get_width(), text(th.SIZE["label"], label, th.TEXT_MUTED, True).get_width()) + pad)
    rect = pygame.Rect(x, y, w, h)
    card(surface, rect, radius=th.RADIUS["md"], alpha=245, lift=True)
    if icon:
        icon(surface, x + pad + 8, y + h // 2 + 2)
    lx = x + pad + icon_w
    roomy = h >= 44  # the compact HUD stacks label and value tighter so they never overlap
    surface.blit(text(th.SIZE["label"], label, th.TEXT_MUTED, True), (lx, y + (8 if roomy else 3)))
    surface.blit(value_surf, (lx, y + h - value_surf.get_height() - (6 if roomy else 2)))
    return rect.right + th.SP["sm"]


def _draw_hud(surface: pygame.Surface, state: GameState, params: DifficultyParams, fx: dict, t: float) -> None:
    width = surface.get_width()
    hud_h, _ = bars(surface.get_size())
    compact = hud_h < HUD_HEIGHT
    y, h = (6, hud_h - 12) if compact else (8, hud_h - 14)
    x = th.SP["lg"]

    # brand
    brand = text(16 if compact else th.SIZE["title"], "NEUROQUEST", th.TEXT, True)
    sub = text(th.SIZE["label"], "THE GAME THAT LEARNS YOU", th.CYAN_DIM, True)
    bw = max(brand.get_width(), sub.get_width()) + 2 * th.SP["md"] + 30
    rect = pygame.Rect(x, y, bw, h)
    card(surface, rect, radius=th.RADIUS["md"], alpha=245, lift=True)
    icon_chip(surface, x + th.SP["md"] + 10, y + h // 2, 20, th.AMBER)
    surface.blit(brand, (x + th.SP["md"] + 30, y + (5 if compact else 9)))
    if not compact:
        surface.blit(sub, (x + th.SP["md"] + 30, y + h - sub.get_height() - 8))
    x = rect.right + th.SP["md"]
    if compact:
        value_size = 16
        x = _stat_card(surface, x, y, h, "SCORE", text(value_size, str(state.score), th.TEXT, True))
        x = _stat_card(surface, x, y, h, "HEALTH", text(value_size, f"{state.player.health}/3", th.TEXT, True))
        x = _stat_card(surface, x, y, h, "TIME", text(value_size, f"{time_left_s(state, params):.0f}s", th.TEXT, True))
        return

    # score (pops on change)
    pop = float(fx.get("score_pop", 0.0))
    score_surf = text(th.SIZE["value"] + int(6 * pop), str(state.score), th.CYAN_DIM if pop > 0 else th.TEXT, True)
    x = _stat_card(surface, x, y, h, "SCORE", score_surf, icon=lambda s, cx, cy: icon_dot(s, cx, cy, 14, th.AMBER), min_w=110)

    # health: hearts
    hearts = pygame.Surface((3 * 24, 22), pygame.SRCALPHA)
    shake = int(fx.get("heart_shake", 0.0) * 3)
    for i in range(3):
        alive = i < state.player.health
        icon_heart(hearts, 12 + i * 24 + (random.randint(-shake, shake) if shake else 0), 10, 18, th.CRIMSON if alive else th.SURFACE_2, outline=th.LINE if not alive else None)
    x = _stat_card(surface, x, y, h, "HEALTH", hearts, min_w=104)

    # time: value + ring
    left = time_left_s(state, params)
    frac = left / max(1, params.time_limit_s)
    colour = th.EMERALD if left > 30 else (th.AMBER if left > 10 else th.CRIMSON)
    if left <= 10 and pulse(t, 8.0) > 0.5:
        colour = th.mix(colour, th.TEXT, 0.4)
    time_surf = pygame.Surface((112, 26), pygame.SRCALPHA)
    time_surf.blit(text(th.SIZE["value"], f"{left:.0f}s", th.TEXT, True), (0, 0))
    ring(time_surf, (98, 13), 12, frac, colour, width=5)
    x = _stat_card(surface, x, y, h, "TIME", time_surf, icon=lambda s, cx, cy: icon_dot(s, cx, cy, 14, colour))

    # gems collected / remaining
    collected = fx.get("coins_total", len(state.rewards)) - len(state.rewards)
    value = text(th.SIZE["value"], str(collected), th.TEXT, True)
    gems_surf = pygame.Surface((value.get_width() + 70, 24), pygame.SRCALPHA)
    gems_surf.blit(value, (0, 0))
    gems_surf.blit(text(th.SIZE["small"], f"/ {len(state.rewards)} left", th.TEXT_MUTED), (value.get_width() + 6, 8))
    x = _stat_card(surface, x, y, h, "GEMS", gems_surf, icon=lambda s, cx, cy: icon_crystal(s, cx, cy, 16, REWARD_DARK, REWARD))

    # hunters
    near = min_enemy_distance(state.player.pos, state.enemies) if state.enemies else 10**9
    hunt_colour = th.CRIMSON if near <= 3 else th.TEXT
    x = _stat_card(surface, x, y, h, "HUNTERS", text(th.SIZE["value"], str(len(state.enemies)), hunt_colour, True), icon=lambda s, cx, cy: icon_drone(s, cx, cy, 16, th.CRIMSON, ENEMY_EYE), min_w=96)


# ---------- bottom bar ----------


def enemy_summary(state: GameState, params: DifficultyParams) -> str:
    n = len(state.enemies)
    cps = state.ticks_per_second / max(1, params.enemy_move_interval)
    return f"{n} hunter{'s' if n != 1 else ''}{SEP}A* pursuit{SEP}move every {params.enemy_move_interval} ticks ({cps:.1f} cells/s)"


def _draw_footer(surface: pygame.Surface, state: GameState, params: DifficultyParams, prediction, notice: str | None, fx: dict, paused: bool) -> None:
    width, height = surface.get_size()
    hud_h, foot_h = bars(surface.get_size())
    top = height - foot_h
    pygame.draw.rect(surface, FOOTER_BG, pygame.Rect(0, top, width, foot_h))
    pygame.draw.rect(surface, HEDGE, pygame.Rect(0, top, width, 3))
    compact = foot_h < FOOTER_HEIGHT
    y = top + foot_h // 2 + 1
    x = th.SP["lg"]

    adaptive = bool(fx.get("adaptive", False))
    label = "ADAPTIVE" if adaptive else "FIXED"
    brect = pygame.Rect(x, y - 11, text(th.SIZE["label"], label, th.ON_ACCENT, True).get_width() + 18, 22)
    pill(surface, brect, th.CYAN if adaptive else th.TEXT_MUTED, label)
    x = brect.right + th.SP["md"]

    from neuroquest.ui.panel import elide, short_decision  # local import: panel imports this module's constants

    last = fx.get("last_decision")
    if last is not None and state.tick < fx.get("last_decision_until", 0):
        line = text(th.SIZE["body"], elide(th.SIZE["body"], "Difficulty: " + short_decision(last), width // 2, True), th.TEXT, True)
        pygame.draw.circle(surface, th.AMBER, (x + 5, y), 5)
        surface.blit(line, (x + 16, y - line.get_height() // 2))
        x += 16
    elif notice and notice.startswith("Difficulty changed"):
        line = text(th.SIZE["body"], elide(th.SIZE["body"], notice.removeprefix("Difficulty changed: "), width // 2, True), th.TEXT, True)
        pygame.draw.circle(surface, th.AMBER, (x + 5, y), 5)
        surface.blit(line, (x + 16, y - line.get_height() // 2))
        x += 16
    else:
        line = text(th.SIZE["body"], enemy_summary(state, params), th.TEXT_MUTED, True)
        surface.blit(line, (x, y - line.get_height() // 2))
    x += line.get_width() + th.SP["xl"]

    if not compact and prediction is not None and not isinstance(prediction, str):
        label = f"Style: {prediction.label} {prediction.confidence:.0%}"
        srect = pygame.Rect(x, y - 11, text(th.SIZE["label"], label, th.TEXT, True).get_width() + 20, 22)
        pill(surface, srect, th.mix(LABEL_COLOURS.get(prediction.label, th.TEXT), (255, 255, 255), 0.72), label, colour=th.TEXT)
        x = srect.right + th.SP["xl"]

    if paused:
        prect = pygame.Rect(x, y - 11, text(th.SIZE["label"], "PAUSED", th.ON_ACCENT, True).get_width() + 18, 22)
        pill(surface, prect, th.AMBER, "PAUSED")
        x = prect.right + th.SP["xl"]

    # controls: the longest version that fits to the right of everything else
    for hint_text in (
        f"ARROWS / WASD move{SEP}T paths{SEP}I AI panel{SEP}P pause{SEP}R restart{SEP}Esc quit",
        f"T paths{SEP}I AI panel{SEP}P pause{SEP}R restart",
        "P pause / controls",
    ):
        hint = text(th.SIZE["small"], hint_text, th.TEXT_MUTED, True)
        if x + hint.get_width() + th.SP["lg"] <= width:
            surface.blit(hint, (width - hint.get_width() - th.SP["lg"], y - hint.get_height() // 2))
            break


# ---------- overlays ----------


def _panel_box(surface: pygame.Surface, w: int, h: int) -> pygame.Rect:
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((230, 240, 255, 170))
    surface.blit(veil, (0, 0))
    rect = pygame.Rect((surface.get_width() - w) // 2, (surface.get_height() - h) // 2, w, h)
    shadow(surface, rect, th.RADIUS["lg"], dy=10, alpha=60)
    card(surface, rect, radius=th.RADIUS["lg"], fill=th.SURFACE, alpha=255, border=th.LINE, border_alpha=255)
    return rect


def _stat(surface: pygame.Surface, x: int, y: int, w: int, label: str, value: str, colour=th.TEXT) -> None:
    from neuroquest.ui.panel import elide

    rect = pygame.Rect(x, y, w, 54)
    card(surface, rect, fill=th.SURFACE_2, alpha=255, border_alpha=0, radius=th.RADIUS["sm"])
    surface.blit(text(th.SIZE["label"], label, th.TEXT_MUTED, True), (x + 10, y + 8))
    size = th.SIZE["title"] if th.font(th.SIZE["title"], True).size(value)[0] <= w - 20 else th.SIZE["body"]
    surface.blit(text(size, elide(size, value, w - 20, True), colour, True), (x + 10, y + 24 + (4 if size != th.SIZE["title"] else 0)))


def _draw_overlay(surface: pygame.Surface, state: GameState, params: DifficultyParams, prediction, fx: dict) -> None:
    titles = {
        Outcome.WON: ("YOU MADE IT!", th.EMERALD, "Through the gate before the hunters got you."),
        Outcome.LOST_TIME: ("TIME'S UP", th.AMBER, "The garden closed for the night. Faster next time."),
        Outcome.LOST_HEALTH: ("CAUGHT!", th.CRIMSON, "The hunters got you. Watch their paths (T)."),
    }
    title, colour, sub = titles[state.outcome]
    rect = _panel_box(surface, 520, 318)
    t1 = text(th.SIZE["display"], title, colour, True)
    surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 30))
    t2 = text(th.SIZE["body"], sub, th.TEXT_MUTED, True)
    surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 86))
    total = int(fx.get("coins_total", len(state.rewards)))
    collected = total - len(state.rewards)
    style = f"{prediction.label} {prediction.confidence:.0%}" if prediction is not None and not isinstance(prediction, str) else "no read"
    style_colour = LABEL_COLOURS.get(getattr(prediction, "label", ""), th.TEXT)
    stats = [
        ("SCORE", str(state.score), th.CYAN_DIM),
        ("TIME", f"{state.elapsed_s:.0f}s", th.TEXT),
        ("GEMS", f"{collected} / {total}", th.TEXT),
        ("HITS TAKEN", str(int(fx.get("hits", 3 - state.player.health))), th.CRIMSON if fx.get("hits") else th.TEXT),
        ("STYLE READ", style, style_colour),
        ("DIFFICULTY CHANGES", str(int(fx.get("changes", 0))), th.TEXT),
    ]
    gx, gy, gw, gap = rect.x + 24, rect.y + 118, (rect.width - 48 - 2 * 8) // 3, 8
    for i, (label, value, c) in enumerate(stats):
        _stat(surface, gx + (i % 3) * (gw + gap), gy + (i // 3) * (54 + gap), gw, label, value, c)
    t3 = text(th.SIZE["body"], "R  play again      Esc  quit", th.TEXT_MUTED, True)
    surface.blit(t3, (rect.centerx - t3.get_width() // 2, rect.bottom - 40))


def _draw_menu(surface: pygame.Surface, state: GameState, fx: dict, t: float) -> None:
    from neuroquest.ui.panel import wrap

    width = min(780, surface.get_width() - 32)
    rect = _panel_box(surface, width, 440)
    t1 = text(th.SIZE["display"], "NEUROQUEST", th.TEXT, True)
    tx = rect.centerx - t1.get_width() // 2 + 18
    icon_chip(surface, tx - 36, rect.y + 58, 30, th.AMBER)
    surface.blit(t1, (tx, rect.y + 34))
    t2 = text(th.SIZE["body"], "THE GAME THAT LEARNS YOU", th.CYAN_DIM, True)
    surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 86))
    meta = fx.get("model_meta") or {}
    f1 = meta.get("test", {}).get("macro_f1")
    cols = [
        ("OBJECTIVE", th.AMBER, ["Collect the blue gems, reach the wooden gate, keep your three hearts.", "", "Pink hunters chase you with A* pathfinding."]),
        ("CONTROLS", th.CYAN, ["Arrows / WASD: move", "T: show hunters' paths", "I: AI panel", "P: pause,  R: restart", "F12: screenshot,  Esc: quit"]),
        ("THE AI", th.EMERALD, [
            "A* pathfinding drives the hunters.",
            f"A {meta.get('model', 'classifier').replace('_', ' ')} reads your style every 2 s" + (f" (held-out F1 {f1:.2f})." if f1 is not None else "."),
            "Rule-based adaptive difficulty logs every decision it takes.",
        ]),
    ]
    gap, pad = 12, 14
    cw = (rect.width - 48 - 2 * gap) // 3
    card_h = 206
    for i, (title, colour, lines) in enumerate(cols):
        cx = rect.x + 24 + i * (cw + gap)
        card(surface, pygame.Rect(cx, rect.y + 118, cw, card_h), fill=th.SURFACE_2, alpha=255, border_alpha=0)
        pygame.draw.circle(surface, colour, (cx + pad + 5, rect.y + 136), 5)
        surface.blit(text(th.SIZE["label"], title, th.TEXT, True), (cx + pad + 16, rect.y + 130))
        y = rect.y + 152
        for line in lines:
            if not line:
                y += 10
                continue
            for part in wrap(th.SIZE["small"], line, cw - 2 * pad, th.TEXT):  # never wider than the card
                if y + 16 > rect.y + 118 + card_h - 8:
                    break
                surface.blit(part, (cx + pad, y))
                y += 17
    blink = th.mix(th.TEXT, th.CYAN, pulse(t, 3.0))
    t4 = text(th.SIZE["title"], "PRESS ENTER TO START", blink, True)
    surface.blit(t4, (rect.centerx - t4.get_width() // 2, rect.bottom - 82))
    t5 = text(th.SIZE["small"], f"maze seed {fx.get('seed', '?')}{SEP}adaptive difficulty {'on' if fx.get('adaptive') else 'off'}", th.TEXT_MUTED, True)
    surface.blit(t5, (rect.centerx - t5.get_width() // 2, rect.bottom - 44))


CONTROLS = [
    ("Arrows / WASD", "move"),
    ("T", "toggle AI vision (hunters' planned paths)"),
    ("I", "toggle the AI intelligence panel"),
    ("P", "pause / resume"),
    ("R", "restart with a new maze"),
    ("F12", "save a screenshot"),
    ("Esc", "quit"),
]


def _draw_intro(surface: pygame.Surface, board_top: int) -> None:
    line1 = text(th.SIZE["body"], f"Collect the gems{SEP}reach the gate{SEP}avoid the pink hunters", th.TEXT, True)
    line2 = text(th.SIZE["small"], f"Arrows / WASD move{SEP}T shows the hunters' paths{SEP}I opens the AI panel{SEP}P pause", th.TEXT_MUTED, True)
    width = max(line1.get_width(), line2.get_width()) + 36
    box = pygame.Rect(0, 0, width, 52)
    box.center = (surface.get_width() // 2, board_top + 34)
    card(surface, box, radius=th.RADIUS["lg"], alpha=245, lift=True)
    surface.blit(line1, (box.centerx - line1.get_width() // 2, box.y + 9))
    surface.blit(line2, (box.centerx - line2.get_width() // 2, box.y + 31))


def _draw_pause(surface: pygame.Surface, fx: dict) -> None:
    rect = _panel_box(surface, 460, 318)
    title = text(34, "PAUSED", th.TEXT, True)
    surface.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 24))
    sub = text(th.SIZE["body"], "The clock is stopped. Press P to continue.", th.TEXT_MUTED, True)
    surface.blit(sub, (rect.centerx - sub.get_width() // 2, rect.y + 66))
    y = rect.y + 104
    from neuroquest.ui.panel import elide

    for key, what in CONTROLS:
        surface.blit(text(th.SIZE["body"], key, th.CYAN_DIM, True), (rect.x + 44, y))
        surface.blit(text(th.SIZE["body"], elide(th.SIZE["body"], what, rect.right - 24 - (rect.x + 184)), th.TEXT, True), (rect.x + 184, y))
        y += 27
