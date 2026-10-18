"""Design system for the NeuroQuest interface ("cartoon world", entry 13): palette, type scale,
spacing, and the drawing helpers every screen uses. Keeping these in one place is what makes the
HUD, the panel and the overlays look like one product.

The look: a pastel sky, white rounded cards with soft shadows, dark-blue ink for text, and a few
saturated accents (gem blue, hero yellow, hunter pink, hedge green). Everything is drawn with
primitives; no image assets.
"""

from __future__ import annotations

import math

import pygame

# ---------- palette ----------
BG = (196, 228, 255)  # sky: the window frame behind everything
BG_DEEP = (52, 60, 84)  # ink (kept under its old name for callers)
SURFACE = (255, 255, 255)  # cards
SURFACE_2 = (240, 244, 252)  # tracks, secondary cards
LINE = (222, 229, 242)
CYAN = (74, 170, 255)  # accent blue (badges, time, "cautious")
CYAN_DIM = (60, 130, 220)
VIOLET = (160, 140, 255)
CRIMSON = (255, 107, 138)  # hunter pink ("aggressive")
AMBER = (255, 190, 70)  # hero yellow
EMERALD = (96, 196, 110)  # hedge green ("explorer")
TEXT = (52, 60, 84)  # ink
TEXT_MUTED = (134, 143, 170)
WARN = (255, 150, 80)
ON_ACCENT = (255, 255, 255)  # text on a coloured pill
SEP = "  /  "  # separator used in one-line texts (the rounded system font has no middle-dot glyph)

# ---------- scale ----------
SP = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}
RADIUS = {"sm": 8, "md": 14, "lg": 20}
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
        _FONTS[key] = pygame.font.SysFont("arialrounded,avenirnext,helveticaneue,segoeui,arial", size, bold=bold)
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


_SHADOWS: dict[tuple, pygame.Surface] = {}


def shadow(surface: pygame.Surface, rect: pygame.Rect, radius: int, dy: int = 5, alpha: int = 36) -> None:
    """A soft drop shadow under a card: a translucent ink rectangle offset downwards. Cached per size,
    because building a fresh alpha surface for every card on every frame was the main frame cost."""
    key = (rect.w, rect.h, radius, alpha)
    layer = _SHADOWS.get(key)
    if layer is None:
        layer = pygame.Surface((rect.w + 16, rect.h + 16), pygame.SRCALPHA)
        pygame.draw.rect(layer, (40, 50, 90, alpha), pygame.Rect(8, 8, rect.w, rect.h), border_radius=radius)
        if len(_SHADOWS) > 64:
            _SHADOWS.clear()
        _SHADOWS[key] = layer
    surface.blit(layer, (rect.x - 8, rect.y - 8 + dy))


_CARDS: dict[tuple, pygame.Surface] = {}


def card(surface: pygame.Surface, rect: pygame.Rect, *, radius: int = RADIUS["md"], fill=SURFACE, alpha: int = 235, border=LINE, border_alpha: int = 200, lift: bool = False) -> None:
    """A rounded surface with a 1-px border; ``lift`` adds the soft shadow: the basic HUD component.
    The translucent layer is cached per size and colour (the panel's card alone is 280×640)."""
    if lift:
        shadow(surface, rect, radius)
    key = (rect.w, rect.h, radius, tuple(fill), alpha, tuple(border), border_alpha)
    layer = _CARDS.get(key)
    if layer is None:
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (*fill, alpha), layer.get_rect(), border_radius=radius)
        pygame.draw.rect(layer, (*border, border_alpha), layer.get_rect(), width=1, border_radius=radius)
        if len(_CARDS) > 64:
            _CARDS.clear()
        _CARDS[key] = layer
    surface.blit(layer, rect.topleft)


def pill(surface: pygame.Surface, rect: pygame.Rect, fill, label: str, colour=ON_ACCENT, size: int = SIZE["label"]) -> None:
    """A coloured capsule with centred bold text (badges: ADAPTIVE, PAUSED, style)."""
    pygame.draw.rect(surface, fill, rect, border_radius=rect.height // 2)
    t = text(size, label, colour, True)
    surface.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))


def bar(surface: pygame.Surface, rect: pygame.Rect, frac: float, colour, track=SURFACE_2) -> None:
    """A rounded progress bar (style probabilities, difficulty knobs). The time display is a ``ring``
    since entry 13, so the old halo option went with it (no caller remained)."""
    radius = rect.height // 2
    pygame.draw.rect(surface, track, rect, border_radius=radius)
    width = int(rect.width * max(0.0, min(1.0, frac)))
    if width > 0:
        pygame.draw.rect(surface, colour, pygame.Rect(rect.x, rect.y, max(width, rect.height), rect.height), border_radius=radius)


def ring(surface: pygame.Surface, centre: tuple[int, int], radius: int, frac: float, colour, track=SURFACE_2, width: int = 5) -> None:
    """A progress ring that empties clockwise from the top (the time dial)."""
    pygame.draw.circle(surface, track, centre, radius, width)
    frac = max(0.0, min(1.0, frac))
    if frac > 0:
        rect = pygame.Rect(centre[0] - radius, centre[1] - radius, radius * 2, radius * 2)
        pygame.draw.arc(surface, colour, rect, math.pi / 2, math.pi / 2 + 2 * math.pi * frac, width)


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


def icon_drone(surface: pygame.Surface, x: int, y: int, size: int, colour, eye=None) -> None:
    """A small hunter head: a round face with two spikes."""
    r = size // 2
    pygame.draw.circle(surface, colour, (x, y), r)
    pygame.draw.polygon(surface, colour, [(x - r + 2, y - r + 3), (x - r // 2, y - r - r // 2), (x - 1, y - r + 2)])
    pygame.draw.polygon(surface, colour, [(x + 1, y - r + 2), (x + r // 2, y - r - r // 2), (x + r - 2, y - r + 3)])
    if eye:
        pygame.draw.circle(surface, eye, (x - r // 3, y - 1), max(1, size // 8))
        pygame.draw.circle(surface, eye, (x + r // 3, y - 1), max(1, size // 8))


def icon_chip(surface: pygame.Surface, x: int, y: int, size: int, colour) -> None:
    """The brand mark: a smiling round hero face in the accent colour."""
    r = size // 2
    pygame.draw.circle(surface, colour, (x, y), r)
    pygame.draw.circle(surface, (255, 255, 255), (x - r // 3, y - r // 4), max(1, r // 4))
    pygame.draw.circle(surface, (255, 255, 255), (x + r // 3, y - r // 4), max(1, r // 4))
    pygame.draw.arc(surface, (255, 255, 255), pygame.Rect(x - r // 2, y - r // 4, r, r), math.pi, 2 * math.pi, max(1, r // 4))


def icon_dot(surface: pygame.Surface, x: int, y: int, size: int, colour) -> None:
    pygame.draw.circle(surface, colour, (x, y), max(3, size // 2))


def pulse(t: float, speed: float = 2.0, phase: float = 0.0) -> float:
    """0..1 breathing value."""
    return 0.5 + 0.5 * math.sin(t * speed + phase)
