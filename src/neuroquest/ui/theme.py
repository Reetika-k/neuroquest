"""Design system for the NeuroQuest interface: palette, type scale, spacing, and the drawing helpers
every screen uses. Keeping these in one place is what makes the HUD, the panel and the overlays look
like one product."""

from __future__ import annotations

import math

import pygame

# ---------- palette (neon only on accents) ----------
BG_DEEP = (14, 18, 36)
BG = (26, 32, 58)  # window frame: mid navy, never black
SURFACE = (17, 22, 44)
SURFACE_2 = (24, 30, 58)
LINE = (46, 56, 96)
CYAN = (54, 228, 255)  # #36E4FF
CYAN_DIM = (0, 150, 178)
VIOLET = (143, 110, 255)
CRIMSON = (255, 56, 92)
AMBER = (255, 184, 48)
EMERALD = (24, 224, 170)
TEXT = (232, 238, 255)
TEXT_MUTED = (122, 136, 176)
WARN = (255, 138, 60)

# ---------- scale ----------
SP = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}
RADIUS = {"sm": 6, "md": 10, "lg": 14}
SIZE = {"display": 40, "title": 20, "label": 11, "value": 22, "body": 14, "small": 12}

_FONTS: dict[tuple[int, bool], pygame.font.Font] = {}
_TEXT: dict[tuple, pygame.Surface] = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    """A cached font. A cached font dies when the font module is quit and re-initialised (tests
    do that; the app never does), so the cache is checked with a cheap call and rebuilt if dead."""
    key = (size, bold)
    cached = _FONTS.get(key)
    if cached is not None:
        try:
            cached.get_height()
            return cached
        except pygame.error:  # "Invalid font (font module quit since font created)"
            _FONTS.clear()
            _TEXT.clear()
    try:
        _FONTS[key] = pygame.font.SysFont("avenirnext,helveticaneue,segoeui,arial", size, bold=bold)
    except Exception:  # no system fonts: the bundled one
        _FONTS[key] = pygame.font.Font(None, size)
    return _FONTS[key]


def text(size: int, s: str, colour=TEXT, bold: bool = False) -> pygame.Surface:
    """Rendered text, cached; survives Pygame being quit and re-initialised (tests do that)."""
    key = (size, s, colour, bold)
    if key in _TEXT:
        return _TEXT[key]
    try:
        surf = font(size, bold).render(s, True, colour)
    except pygame.error:  # "Invalid font": font module was re-initialised
        _FONTS.clear()
        _TEXT.clear()
        surf = font(size, bold).render(s, True, colour)
    if len(_TEXT) > 600:
        _TEXT.clear()
    _TEXT[key] = surf
    return surf


def mix(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def card(surface: pygame.Surface, rect: pygame.Rect, *, radius: int = RADIUS["md"], fill=SURFACE, alpha: int = 215, border=LINE, border_alpha: int = 150) -> None:
    """A translucent rounded surface with a 1-px border: the basic HUD component."""
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, (*fill, alpha), layer.get_rect(), border_radius=radius)
    pygame.draw.rect(layer, (*border, border_alpha), layer.get_rect(), width=1, border_radius=radius)
    surface.blit(layer, rect.topleft)


def bar(surface: pygame.Surface, rect: pygame.Rect, frac: float, colour, track=SURFACE_2, glow_alpha: int = 0) -> None:
    """A rounded progress bar (health, time, probabilities). ``glow_alpha`` adds a soft halo; the HUD
    uses it on the time bar during the last ten seconds."""
    radius = rect.height // 2
    pygame.draw.rect(surface, track, rect, border_radius=radius)
    width = int(rect.width * max(0.0, min(1.0, frac)))
    if width > 0:
        if glow_alpha:
            layer = pygame.Surface((rect.width + 8, rect.height + 8), pygame.SRCALPHA)
            pygame.draw.rect(layer, (*colour, glow_alpha), pygame.Rect(4, 4, width, rect.height).inflate(6, 6), border_radius=radius + 3)
            surface.blit(layer, (rect.x - 4, rect.y - 4))
        pygame.draw.rect(surface, colour, pygame.Rect(rect.x, rect.y, max(width, rect.height), rect.height), border_radius=radius)


# The glow, vignette and edge-pulse helpers of the sci-fi look (entry 8c) were removed in Phase 10:
# the bright-arcade renderer (entry 8e) stopped using them and they were dead, untested code.


# ---------- tiny vector icons (all drawn with primitives) ----------


def icon_heart(surface: pygame.Surface, x: int, y: int, size: int, colour, outline=None) -> None:
    r = max(2, size // 4)
    pts = [(x - 2 * r, y - r // 2 + 1), (x + 2 * r, y - r // 2 + 1), (x, y + size // 2)]
    pygame.draw.circle(surface, colour, (x - r, y - r // 2), r)
    pygame.draw.circle(surface, colour, (x + r, y - r // 2), r)
    pygame.draw.polygon(surface, colour, pts)
    if outline:
        pygame.draw.circle(surface, outline, (x - r, y - r // 2), r, width=1)
        pygame.draw.circle(surface, outline, (x + r, y - r // 2), r, width=1)


def icon_crystal(surface: pygame.Surface, x: int, y: int, size: int, colour, core=None) -> None:
    h = size // 2
    w = max(2, int(size * 0.32))
    pygame.draw.polygon(surface, colour, [(x, y - h), (x + w, y), (x, y + h), (x - w, y)])
    if core:
        pygame.draw.polygon(surface, core, [(x, y - h // 2), (x + w // 2, y), (x, y + h // 2), (x - w // 2, y)])


def icon_clock(surface: pygame.Surface, x: int, y: int, size: int, colour) -> None:
    r = size // 2
    pygame.draw.circle(surface, colour, (x, y), r, width=2)
    pygame.draw.line(surface, colour, (x, y), (x, y - r + 3), 2)
    pygame.draw.line(surface, colour, (x, y), (x + r - 4, y), 2)


def icon_drone(surface: pygame.Surface, x: int, y: int, size: int, colour, eye=None) -> None:
    h = size // 2
    pygame.draw.polygon(surface, colour, [(x, y - h), (x + h, y + h // 2), (x, y + h // 4), (x - h, y + h // 2)])
    if eye:
        pygame.draw.circle(surface, eye, (x, y - h // 4), max(1, size // 8))


def icon_bolt(surface: pygame.Surface, x: int, y: int, size: int, colour) -> None:
    h = size // 2
    w = size // 4
    pygame.draw.polygon(surface, colour, [(x + w // 2, y - h), (x - w, y + 1), (x, y + 1), (x - w // 2, y + h), (x + w, y - 1), (x, y - 1)])


def icon_chip(surface: pygame.Surface, x: int, y: int, size: int, colour) -> None:
    """A small 'AI core' chip: square with notched corners and a glowing centre."""
    r = size // 2
    pygame.draw.rect(surface, colour, pygame.Rect(x - r, y - r, size, size), width=2, border_radius=3)
    pygame.draw.rect(surface, colour, pygame.Rect(x - r // 2, y - r // 2, r, r), border_radius=2)
    for dx in (-r + 2, 0, r - 2):
        pygame.draw.line(surface, colour, (x + dx, y - r - 3), (x + dx, y - r), 1)
        pygame.draw.line(surface, colour, (x + dx, y + r), (x + dx, y + r + 3), 1)


def pulse(t: float, speed: float = 2.0, phase: float = 0.0) -> float:
    """0..1 breathing value."""
    return 0.5 + 0.5 * math.sin(t * speed + phase)
