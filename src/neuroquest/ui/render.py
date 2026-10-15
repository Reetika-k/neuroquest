"""Drawing only: reads a GameState (plus optional UI-side effects) and paints it. Never changes the state.

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
from neuroquest.ui.theme import bar, card, icon_bolt, icon_chip, icon_clock, icon_crystal, icon_drone, icon_heart, pulse, text

# ---------- identity colours (exported; tests check these at cell centres) ----------
BG = th.BG
FOOTER_BG = th.SURFACE
HUD_BG = th.SURFACE
PLAYER = th.CYAN
PLAYER_DARK = (0, 110, 130)
PLAYER_HIT = (220, 250, 255)
ENEMY = th.CRIMSON
ENEMY_DARK = (120, 16, 44)
ENEMY_EYE = (255, 240, 246)
REWARD = th.AMBER
REWARD_DARK = (170, 110, 10)
EXIT = th.EMERALD
WALL = (33, 42, 74)  # deep navy walls
WALL_EDGE = (78, 96, 140)  # crisp lighter top/left edge
WALL_SHADE = (18, 23, 44)  # bottom/right edge
WALL_OUTLINE = (14, 18, 36)
FLOOR = (88, 102, 142)  # light blue-grey corridors
FLOOR_ALT = (91, 105, 146)
FLOOR_AO = (72, 84, 120)  # faint shade on floor along walls
VISITED = (190, 225, 245)  # trail: a light, translucent tint
OUTLINE = (16, 22, 40)  # dark outline on every sprite
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
    surface.fill(BG)
    panel = panel_width(size_px, fx)
    size, ox, oy = layout(state.maze, size_px, panel)
    shake = float(fx.get("shake", 0.0))
    if shake > 0:
        ox += int(random.uniform(-1, 1) * shake * size * 0.25)
        oy += int(random.uniform(-1, 1) * shake * size * 0.25)

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
    if warn:  # proximity / hit warning: a thin crimson frame around the board, nothing over the play area
        board = pygame.Rect(ox, oy, state.maze.width * size, state.maze.height * size)
        pygame.draw.rect(surface, th.CRIMSON, board.inflate(6, 6), width=3, border_radius=4)

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


# ---------- board ----------


def _board_surface(maze: Maze, size: int) -> pygame.Surface:
    """Static tiles rendered once per maze and cell size.

    Floors: two navy tones with a faint shade along any wall (corridor separation). Walls: one slate
    fill, so adjacent wall cells form continuous structures; a highlight runs only along the top/left
    of wall↔floor boundaries and a slightly darker line along the bottom/right ones, which gives
    depth without outlining every block.
    """
    key = (id(maze), size)
    if key in _BOARDS:
        return _BOARDS[key]
    w, h = maze.width * size, maze.height * size
    surf = pygame.Surface((w, h))
    surf.fill(WALL)
    edge = max(2, size // 10)
    ao = max(1, size // 10)
    for r, row in enumerate(maze.grid):
        for c, open_ in enumerate(row):
            if not open_:
                continue
            rect = pygame.Rect(c * size, r * size, size, size)
            pygame.draw.rect(surf, FLOOR if (r + c) % 2 == 0 else FLOOR_ALT, rect)
            if not maze.is_open((r - 1, c)):
                pygame.draw.rect(surf, FLOOR_AO, pygame.Rect(rect.x, rect.y, size, ao))
            if not maze.is_open((r, c - 1)):
                pygame.draw.rect(surf, FLOOR_AO, pygame.Rect(rect.x, rect.y, ao, size))
    for r, row in enumerate(maze.grid):
        for c, open_ in enumerate(row):
            if open_:
                continue
            rect = pygame.Rect(c * size, r * size, size, size)
            if maze.is_open((r - 1, c)):
                pygame.draw.rect(surf, WALL_EDGE, pygame.Rect(rect.x, rect.y, size, edge))
            if maze.is_open((r, c - 1)):
                pygame.draw.rect(surf, WALL_EDGE, pygame.Rect(rect.x, rect.y, edge, size))
            if maze.is_open((r + 1, c)):
                pygame.draw.rect(surf, WALL_SHADE, pygame.Rect(rect.x, rect.bottom - edge, size, edge))
            if maze.is_open((r, c + 1)):
                pygame.draw.rect(surf, WALL_SHADE, pygame.Rect(rect.right - edge, rect.y, edge, size))
    pygame.draw.rect(surf, WALL_OUTLINE, surf.get_rect(), width=2)
    if len(_BOARDS) > 8:
        _BOARDS.clear()
    _BOARDS[key] = surf
    return surf


def _draw_board(surface: pygame.Surface, state: GameState, size: int, ox: int, oy: int, fx: dict) -> None:
    surface.blit(_board_surface(state.maze, size), (ox, oy))
    trail: dict = fx.get("trail", {})
    tick = state.tick
    layer = pygame.Surface((size, size), pygame.SRCALPHA)
    for cell in state.visited:
        r, c = cell
        age = tick - trail.get(cell, -10**6)
        strength = 0.35 if not trail else max(0.12, 0.55 - age / 500.0)
        layer.fill((*VISITED, int(70 * strength)))
        surface.blit(layer, (ox + c * size, oy + r * size))


def _draw_enemy_paths(surface: pygame.Surface, state: GameState, size: int, ox: int, oy: int) -> None:
    """AI vision: the next cells of each hunter's shortest path to the player, fading with distance."""
    layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for enemy in state.enemies:
        path = a_star(state.maze, enemy.pos, state.player.pos)
        if not path:
            continue
        pts = [_centre(cell, size, ox, oy) for cell in path[:13]]
        for i in range(len(pts) - 1):
            alpha = int(150 * (1 - i / 13))
            pygame.draw.line(layer, (*ENEMY, alpha), pts[i], pts[i + 1], max(2, size // 9))
        for i, p in enumerate(pts[1:]):
            pygame.draw.circle(layer, (*ENEMY, int(190 * (1 - i / 13))), p, max(2, size // 8))
    surface.blit(layer, (0, 0))


# ---------- things (bright arcade: outlined, no glow) ----------


def _outlined_polygon(surface: pygame.Surface, pts, fill, outline=OUTLINE, width: int = 2) -> None:
    pygame.draw.polygon(surface, fill, pts)
    pygame.draw.polygon(surface, outline, pts, width=width)


def _draw_reward(surface: pygame.Surface, centre: tuple[int, int], size: int, t: float) -> None:
    """Amber crystal: outlined diamond with a white facet; bobs gently."""
    cx, cy = centre[0], centre[1] + int(math.sin(t * 3.0 + centre[0] * 0.05) * size * 0.04)
    h = max(5, int(size * 0.27))
    w = max(3, int(h * 0.68))
    _outlined_polygon(surface, [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)], REWARD)
    pygame.draw.polygon(surface, (255, 240, 200), [(cx - 1, cy - h + 3), (cx - w + 3, cy - 1), (cx - 1, cy - 1)])


def _draw_exit(surface: pygame.Surface, centre: tuple[int, int], size: int, t: float) -> None:
    """Emerald portal: outlined ring with a rotating dashed inner ring and a lit core."""
    outer = max(6, int(size * 0.38))
    pygame.draw.circle(surface, (16, 96, 72), centre, outer)
    pygame.draw.circle(surface, OUTLINE, centre, outer, width=2)
    core = max(4, int(size * 0.24))
    pygame.draw.circle(surface, EXIT, centre, core)
    pygame.draw.circle(surface, (220, 255, 240), (centre[0] - core // 3, centre[1] - core // 3), max(1, size // 14))  # highlight, off-centre
    ring = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
    rr = max(5, int(size * 0.31))
    rect = pygame.Rect(size - rr, size - rr, rr * 2, rr * 2)
    for k in range(4):
        a0 = t * 1.8 + k * math.pi / 2
        pygame.draw.arc(ring, (210, 255, 235, 230), rect, a0, a0 + 0.7, width=max(2, size // 14))
    surface.blit(ring, (centre[0] - size, centre[1] - size))


def _draw_enemy(surface: pygame.Surface, centre: tuple[int, int], size: int, t: float, idx: int, state: GameState) -> None:
    """Crimson hunter: outlined dart turned toward the player, one eye; an exclamation mark appears when it is
    within 3 cells (driven by the real distance)."""
    pr, pc = state.player.pos
    enemy = next((e for e in state.enemies if e.id == idx), None)
    er, ec = enemy.pos if enemy else (pr, pc)
    dist = abs(pr - er) + abs(pc - ec)
    close = dist <= 3
    dx, dy = (pc - ec), (pr - er)
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    cx, cy = centre[0], centre[1] + int(math.sin(t * 5.0 + idx) * size * 0.04)
    L = size * 0.44
    pts = [
        (cx + ux * L, cy + uy * L),
        (cx + px * L * 0.78 - ux * L * 0.4, cy + py * L * 0.78 - uy * L * 0.4),
        (cx - ux * L * 0.5, cy - uy * L * 0.5),
        (cx - px * L * 0.78 - ux * L * 0.4, cy - py * L * 0.78 - uy * L * 0.4),
    ]
    _outlined_polygon(surface, pts, ENEMY)
    eye = (int(cx + ux * size * 0.14), int(cy + uy * size * 0.14))
    eye_r = max(3, size // 7)
    pygame.draw.circle(surface, (255, 255, 255), eye, eye_r)
    pygame.draw.circle(surface, OUTLINE, (int(eye[0] + ux * eye_r * 0.35), int(eye[1] + uy * eye_r * 0.35)), max(1, eye_r // 2))
    if close:
        mx, my = cx, cy - int(size * 0.62)
        _outlined_polygon(surface, [(mx - size * 0.12, my + size * 0.12), (mx + size * 0.12, my + size * 0.12), (mx, my - size * 0.12)], (255, 220, 70), width=1)
        pygame.draw.line(surface, OUTLINE, (mx, my - size * 0.02), (mx, my + size * 0.05), 2)
        pygame.draw.circle(surface, OUTLINE, (mx, my + int(size * 0.09)), 1)


def _draw_player(surface: pygame.Surface, state: GameState, centre: tuple[int, int], size: int, t: float, fx: dict) -> None:
    """Cyan robot: chunky rounded body with a dark outline, a visor band with two eyes that look where you move,
    an antenna and two feet; blinks pale while invulnerable."""
    invulnerable = state.tick < state.player.invulnerable_until
    hit_blink = invulnerable and (state.tick // 2) % 2 == 0
    body = PLAYER_HIT if hit_blink else PLAYER
    cx, cy = centre
    half_w, half_h = int(size * 0.34), int(size * 0.32)
    bob = int(math.sin(t * 6.0) * size * 0.03)
    rect = pygame.Rect(cx - half_w, cy - half_h + bob, half_w * 2, half_h * 2)
    radius = max(4, size // 5)
    pygame.draw.rect(surface, OUTLINE, rect.inflate(4, 4), border_radius=radius + 2)
    pygame.draw.rect(surface, body, rect, border_radius=radius)
    band_h = min(max(3, int(size * 0.2)), half_h - 3)  # the visor always ends above the centre pixel
    band = pygame.Rect(rect.x + 3, rect.y + 3, rect.width - 6, band_h)
    pygame.draw.rect(surface, PLAYER_DARK, band, border_radius=max(2, radius // 2))
    fr, fc = fx.get("facing", (0, 1))
    eye_r = max(2, size // 11)
    ey = band.centery
    for sx in (-1, 1):
        ex = cx + sx * int(size * 0.13)
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), eye_r)
        pygame.draw.circle(surface, OUTLINE, (ex + int(fc * eye_r * 0.45), ey + int(fr * eye_r * 0.45)), max(1, eye_r // 2))
    pygame.draw.line(surface, OUTLINE, (cx, rect.y - 1), (cx, rect.y - int(size * 0.12)), 2)
    pygame.draw.circle(surface, (255, 110, 110) if invulnerable else th.AMBER, (cx, rect.y - int(size * 0.13)), max(2, size // 14))
    for sx in (-1, 1):
        pygame.draw.rect(surface, OUTLINE, pygame.Rect(cx + sx * int(size * 0.16) - 3, rect.bottom - 1, 6, max(2, size // 12)), border_radius=2)


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
    """One HUD component: small caps label above a value, optional icon on the left. Returns the next x."""
    pad = th.SP["md"]
    icon_w = 22 if icon else 0
    w = max(min_w, pad + icon_w + max(value_surf.get_width(), text(th.SIZE["label"], label, th.TEXT_MUTED, True).get_width()) + pad)
    rect = pygame.Rect(x, y, w, h)
    card(surface, rect)
    if icon:
        icon(surface, x + pad + 8, y + h // 2 + 2)
    lx = x + pad + icon_w
    surface.blit(text(th.SIZE["label"], label, th.TEXT_MUTED, True), (lx, y + 8))
    surface.blit(value_surf, (lx, y + h - value_surf.get_height() - 6))
    return rect.right + th.SP["sm"]


def _draw_hud(surface: pygame.Surface, state: GameState, params: DifficultyParams, fx: dict, t: float) -> None:
    width = surface.get_width()
    hud_h, _ = bars(surface.get_size())
    pygame.draw.rect(surface, HUD_BG, pygame.Rect(0, 0, width, hud_h))
    pygame.draw.line(surface, th.LINE, (0, hud_h - 1), (width, hud_h - 1))
    compact = hud_h < HUD_HEIGHT
    y, h = (8, hud_h - 16)
    x = th.SP["lg"]

    # brand
    brand = text(16 if compact else th.SIZE["title"], "NEUROQUEST", th.TEXT, True)
    sub = text(th.SIZE["label"], "THE GAME THAT LEARNS YOU", th.CYAN_DIM, True)
    bw = max(brand.get_width(), sub.get_width()) + 2 * th.SP["md"] + 26
    rect = pygame.Rect(x, y, bw, h)
    card(surface, rect, fill=th.SURFACE_2, border=th.CYAN_DIM, border_alpha=120)
    icon_chip(surface, x + th.SP["md"] + 8, y + h // 2, 14, th.CYAN)
    surface.blit(brand, (x + th.SP["md"] + 26, y + (6 if compact else 9)))
    if not compact:
        surface.blit(sub, (x + th.SP["md"] + 26, y + h - sub.get_height() - 8))
    x = rect.right + th.SP["md"]
    if compact:
        value_size = 16
        x = _stat_card(surface, x, y, h, "SCORE", text(value_size, str(state.score), th.TEXT, True))
        x = _stat_card(surface, x, y, h, "HEALTH", text(value_size, f"{state.player.health}/3", th.TEXT, True))
        x = _stat_card(surface, x, y, h, "TIME", text(value_size, f"{time_left_s(state, params):.0f}s", th.TEXT, True))
        return

    # score (pops on change)
    pop = float(fx.get("score_pop", 0.0))
    score_surf = text(th.SIZE["value"] + int(6 * pop), str(state.score), th.AMBER if pop > 0 else th.TEXT, True)
    x = _stat_card(surface, x, y, h, "SCORE", score_surf, icon=lambda s, cx, cy: icon_bolt(s, cx, cy, 16, th.AMBER), min_w=110)

    # health: hearts
    hearts = pygame.Surface((3 * 24, 22), pygame.SRCALPHA)
    shake = int(fx.get("heart_shake", 0.0) * 3)
    for i in range(3):
        alive = i < state.player.health
        icon_heart(hearts, 12 + i * 24 + (random.randint(-shake, shake) if shake else 0), 10, 18, th.CRIMSON if alive else th.SURFACE_2, outline=th.LINE if not alive else None)
    x = _stat_card(surface, x, y, h, "HEALTH", hearts, min_w=104)

    # time: bar + value
    left = time_left_s(state, params)
    frac = left / max(1, params.time_limit_s)
    colour = th.CYAN if left > 30 else (th.AMBER if left > 10 else th.CRIMSON)
    if left <= 10 and pulse(t, 8.0) > 0.5:
        colour = th.mix(colour, th.TEXT, 0.5)
    time_surf = pygame.Surface((190, 22), pygame.SRCALPHA)
    bar(time_surf, pygame.Rect(0, 7, 130, 10), frac, colour, glow_alpha=60 if left <= 10 else 0)
    time_surf.blit(text(th.SIZE["body"], f"{left:.0f}s", th.TEXT, True), (138, 2))
    x = _stat_card(surface, x, y, h, "TIME", time_surf, icon=lambda s, cx, cy: icon_clock(s, cx, cy, 14, colour))

    # coins collected / remaining
    collected = fx.get("coins_total", len(state.rewards)) - len(state.rewards)
    coins_surf = pygame.Surface((90, 24), pygame.SRCALPHA)
    coins_surf.blit(text(th.SIZE["value"], str(collected), th.TEXT, True), (0, 0))
    coins_surf.blit(text(th.SIZE["small"], f"/ {len(state.rewards)} left", th.TEXT_MUTED), (text(th.SIZE["value"], str(collected), th.TEXT, True).get_width() + 6, 8))
    x = _stat_card(surface, x, y, h, "COINS", coins_surf, icon=lambda s, cx, cy: icon_crystal(s, cx, cy, 16, th.AMBER, (255, 236, 170)))

    # hunters
    near = min_enemy_distance(state.player.pos, state.enemies) if state.enemies else 10**9
    hunt_colour = th.CRIMSON if near <= 3 else th.TEXT
    x = _stat_card(surface, x, y, h, "HUNTERS", text(th.SIZE["value"], str(len(state.enemies)), hunt_colour, True), icon=lambda s, cx, cy: icon_drone(s, cx, cy, 16, th.CRIMSON, (255, 230, 236)), min_w=96)


# ---------- bottom bar ----------


def enemy_summary(state: GameState, params: DifficultyParams) -> str:
    n = len(state.enemies)
    cps = state.ticks_per_second / max(1, params.enemy_move_interval)
    return f"{n} hunter{'s' if n != 1 else ''}  ·  A* pursuit  ·  move every {params.enemy_move_interval} ticks ({cps:.1f} cells/s)"


def _draw_footer(surface: pygame.Surface, state: GameState, params: DifficultyParams, prediction, notice: str | None, fx: dict, paused: bool) -> None:
    width, height = surface.get_size()
    hud_h, foot_h = bars(surface.get_size())
    top = height - foot_h
    pygame.draw.rect(surface, FOOTER_BG, pygame.Rect(0, top, width, foot_h))
    pygame.draw.line(surface, th.LINE, (0, top), (width, top))
    compact = foot_h < FOOTER_HEIGHT
    y = top + foot_h // 2
    x = th.SP["lg"]

    adaptive = bool(fx.get("adaptive", False))
    badge = text(th.SIZE["label"], "ADAPTIVE" if adaptive else "FIXED", th.BG_DEEP, True)
    brect = pygame.Rect(x, y - 10, badge.get_width() + 14, 20)
    pygame.draw.rect(surface, th.CYAN if adaptive else th.TEXT_MUTED, brect, border_radius=10)
    surface.blit(badge, (brect.x + 7, brect.y + 4))
    x = brect.right + th.SP["md"]

    from neuroquest.ui.panel import elide, short_decision  # local import: panel imports this module's constants

    last = fx.get("last_decision")
    if last is not None and state.tick < fx.get("last_decision_until", 0):
        line = text(th.SIZE["body"], elide(th.SIZE["body"], "Difficulty: " + short_decision(last), width // 2, True), th.TEXT, True)
        pygame.draw.circle(surface, th.AMBER, (x + 5, y), 4)
        surface.blit(line, (x + 14, y - line.get_height() // 2))
        x += 14
    elif notice and notice.startswith("Difficulty changed"):
        line = text(th.SIZE["body"], elide(th.SIZE["body"], notice.removeprefix("Difficulty changed: "), width // 2, True), th.TEXT, True)
        pygame.draw.circle(surface, th.AMBER, (x + 5, y), 4)
        surface.blit(line, (x + 14, y - line.get_height() // 2))
        x += 14
    else:
        line = text(th.SIZE["body"], enemy_summary(state, params), th.TEXT_MUTED)
        surface.blit(line, (x, y - line.get_height() // 2))
    x += line.get_width() + th.SP["xl"]

    if not compact and prediction is not None and not isinstance(prediction, str):
        colour = LABEL_COLOURS.get(prediction.label, th.TEXT)
        pygame.draw.circle(surface, colour, (x + 5, y), 5)
        style = text(th.SIZE["body"], f"STYLE  {prediction.label} {prediction.confidence:.0%}", th.TEXT, True)
        surface.blit(style, (x + 16, y - style.get_height() // 2))
        x += 16 + style.get_width() + th.SP["xl"]

    if paused:
        pb = text(th.SIZE["label"], "PAUSED", th.BG_DEEP, True)
        prect = pygame.Rect(x, y - 10, pb.get_width() + 16, 20)
        pygame.draw.rect(surface, th.AMBER, prect, border_radius=10)
        surface.blit(pb, (prect.x + 8, prect.y + 4))
        x = prect.right + th.SP["xl"]

    # controls: the longest version that fits to the right of everything else
    for hint_text in (
        "ARROWS / WASD move  ·  T paths  ·  I AI panel  ·  P pause  ·  R restart  ·  Esc quit",
        "T paths  ·  I AI panel  ·  P pause  ·  R restart",
        "P pause · controls",
    ):
        hint = text(th.SIZE["small"], hint_text, th.TEXT_MUTED)
        if x + hint.get_width() + th.SP["lg"] <= width:
            surface.blit(hint, (width - hint.get_width() - th.SP["lg"], y - hint.get_height() // 2))
            break


# ---------- overlays (restyled in Step 7) ----------


def _panel_box(surface: pygame.Surface, w: int, h: int) -> pygame.Rect:
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((6, 9, 20, 190))
    surface.blit(veil, (0, 0))
    rect = pygame.Rect((surface.get_width() - w) // 2, (surface.get_height() - h) // 2, w, h)
    card(surface, rect, radius=th.RADIUS["lg"], fill=th.SURFACE, alpha=245, border=th.LINE, border_alpha=220)
    return rect


def _stat(surface: pygame.Surface, x: int, y: int, w: int, label: str, value: str, colour=th.TEXT) -> None:
    from neuroquest.ui.panel import elide

    rect = pygame.Rect(x, y, w, 54)
    card(surface, rect, fill=th.SURFACE_2, alpha=230, border_alpha=110, radius=th.RADIUS["sm"])
    surface.blit(text(th.SIZE["label"], label, th.TEXT_MUTED, True), (x + 10, y + 8))
    size = th.SIZE["title"] if th.font(th.SIZE["title"], True).size(value)[0] <= w - 20 else th.SIZE["body"]
    surface.blit(text(size, elide(size, value, w - 20, True), colour, True), (x + 10, y + 24 + (4 if size != th.SIZE["title"] else 0)))


def _draw_overlay(surface: pygame.Surface, state: GameState, params: DifficultyParams, prediction, fx: dict) -> None:
    titles = {
        Outcome.WON: ("YOU ESCAPED", th.EMERALD, "The portal took you out before the hunters got you."),
        Outcome.LOST_TIME: ("TIME'S UP", th.AMBER, "The maze sealed itself. Faster next time."),
        Outcome.LOST_HEALTH: ("DEFEATED", th.CRIMSON, "The hunters caught you. Watch their paths (T)."),
    }
    title, colour, sub = titles[state.outcome]
    rect = _panel_box(surface, 520, 318)
    t1 = text(th.SIZE["display"], title, colour, True)
    surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 30))
    t2 = text(th.SIZE["body"], sub, th.TEXT_MUTED)
    surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 86))
    total = int(fx.get("coins_total", len(state.rewards)))
    collected = total - len(state.rewards)
    style = f"{prediction.label} {prediction.confidence:.0%}" if prediction is not None and not isinstance(prediction, str) else "no read"
    style_colour = LABEL_COLOURS.get(getattr(prediction, "label", ""), th.TEXT)
    stats = [
        ("SCORE", str(state.score), th.AMBER),
        ("TIME", f"{state.elapsed_s:.0f}s", th.TEXT),
        ("COINS", f"{collected} / {total}", th.TEXT),
        ("HITS TAKEN", str(int(fx.get("hits", 3 - state.player.health))), th.CRIMSON if fx.get("hits") else th.TEXT),
        ("STYLE READ", style, style_colour),
        ("DIFFICULTY CHANGES", str(int(fx.get("changes", 0))), th.TEXT),
    ]
    gx, gy, gw, gap = rect.x + 24, rect.y + 118, (rect.width - 48 - 2 * 8) // 3, 8
    for i, (label, value, c) in enumerate(stats):
        _stat(surface, gx + (i % 3) * (gw + gap), gy + (i // 3) * (54 + gap), gw, label, value, c)
    t3 = text(th.SIZE["body"], "R  play again      Esc  quit", th.TEXT_MUTED)
    surface.blit(t3, (rect.centerx - t3.get_width() // 2, rect.bottom - 40))


def _draw_menu(surface: pygame.Surface, state: GameState, fx: dict, t: float) -> None:
    from neuroquest.ui.panel import wrap

    width = min(780, surface.get_width() - 32)
    rect = _panel_box(surface, width, 440)
    t1 = text(th.SIZE["display"], "NEUROQUEST", th.TEXT, True)
    tx = rect.centerx - t1.get_width() // 2 + 18
    icon_chip(surface, tx - 34, rect.y + 58, 22, th.CYAN)
    surface.blit(t1, (tx, rect.y + 34))
    t2 = text(th.SIZE["body"], "THE GAME THAT LEARNS YOU", th.CYAN_DIM, True)
    surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 86))
    meta = fx.get("model_meta") or {}
    f1 = meta.get("test", {}).get("macro_f1")
    cols = [
        ("OBJECTIVE", ["Collect the amber crystals, reach the green portal, keep your three hearts.", "", "Red hunters chase you with A* pathfinding."]),
        ("CONTROLS", ["Arrows / WASD · move", "T · show hunters' paths", "I · AI panel", "P · pause,  R · restart", "F12 · screenshot,  Esc · quit"]),
        ("THE AI", [
            "A* pathfinding drives the hunters.",
            f"A {meta.get('model', 'classifier').replace('_', ' ')} reads your style every 2 s" + (f" (held-out F1 {f1:.2f})." if f1 is not None else "."),
            "Rule-based adaptive difficulty logs every decision it takes.",
        ]),
    ]
    gap, pad = 12, 14
    cw = (rect.width - 48 - 2 * gap) // 3
    card_h = 206
    for i, (title, lines) in enumerate(cols):
        cx = rect.x + 24 + i * (cw + gap)
        card(surface, pygame.Rect(cx, rect.y + 118, cw, card_h), fill=th.SURFACE_2, alpha=230, border_alpha=110)
        surface.blit(text(th.SIZE["label"], title, th.CYAN, True), (cx + pad, rect.y + 130))
        y = rect.y + 152
        for line in lines:
            if not line:
                y += 10
                continue
            for part in wrap(th.SIZE["small"], line, cw - 2 * pad):  # never wider than the card
                if y + 16 > rect.y + 118 + card_h - 8:
                    break
                surface.blit(part, (cx + pad, y))
                y += 17
    blink = th.mix(th.TEXT, th.CYAN, pulse(t, 3.0))
    t4 = text(th.SIZE["title"], "PRESS ENTER TO START", blink, True)
    surface.blit(t4, (rect.centerx - t4.get_width() // 2, rect.bottom - 82))
    t5 = text(th.SIZE["small"], f"maze seed {fx.get('seed', '?')}   ·   adaptive difficulty {'on' if fx.get('adaptive') else 'off'}", th.TEXT_MUTED)
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
    line1 = text(th.SIZE["body"], "Collect the coins  ·  reach the green portal  ·  avoid the red hunters", th.TEXT, True)
    line2 = text(th.SIZE["small"], "Arrows / WASD move   ·   T shows the hunters' paths   ·   I opens the AI panel   ·   P pause", th.TEXT_MUTED)
    width = max(line1.get_width(), line2.get_width()) + 36
    pill = pygame.Rect(0, 0, width, 52)
    pill.center = (surface.get_width() // 2, board_top + 34)
    card(surface, pill, radius=th.RADIUS["lg"], alpha=230)
    surface.blit(line1, (pill.centerx - line1.get_width() // 2, pill.y + 9))
    surface.blit(line2, (pill.centerx - line2.get_width() // 2, pill.y + 31))


def _draw_pause(surface: pygame.Surface, fx: dict) -> None:
    rect = _panel_box(surface, 460, 318)
    title = text(34, "PAUSED", th.TEXT, True)
    surface.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 24))
    sub = text(th.SIZE["body"], "The clock is stopped. Press P to continue.", th.TEXT_MUTED)
    surface.blit(sub, (rect.centerx - sub.get_width() // 2, rect.y + 66))
    y = rect.y + 104
    from neuroquest.ui.panel import elide

    for key, what in CONTROLS:
        surface.blit(text(th.SIZE["body"], key, th.CYAN, True), (rect.x + 44, y))
        surface.blit(text(th.SIZE["body"], elide(th.SIZE["body"], what, rect.right - 24 - (rect.x + 184)), th.TEXT), (rect.x + 184, y))
        y += 27
