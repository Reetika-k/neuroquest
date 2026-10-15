"""The intelligence panel: what the AI knows and did, from real data only.

Sources: the live ``Prediction`` (label, three class probabilities, confidence), the current
``DifficultyParams``, the controller's ``Decision`` records (rule, lever, before/after, reason) and
the saved model's metadata (name, training date, held-out scores). Nothing is estimated here.
"""

from __future__ import annotations

import pygame

from neuroquest.adapt.bounds import BOUNDS
from neuroquest.ui import theme as th
from neuroquest.ui.theme import bar, card, icon_chip, text

RULE_NAMES = {
    "R1_ease_heavy_damage": "Eased: heavy damage",
    "R2_ease_behind_on_time": "Eased: behind on time",
    "R3_tighten_cruising": "Tightened: cruising",
    "R0_hold": "Holding",
    "R0_hold_warmup": "Holding: warm-up",
    "R0_hold_no_encounter": "Holding: no encounter yet",
    "R0_hold_cooldown": "Holding: cooldown",
}
LEVER_NAMES = {"enemy_count": "hunters", "enemy_move_interval": "enemy move interval", "time_limit_s": "time limit", "reward_density": "coin density"}
LABEL_COLOURS = {"aggressive": th.CRIMSON, "cautious": th.CYAN, "explorer": th.EMERALD}


def rule_name(rule: str) -> str:
    base, at_bound = (rule[:-9], True) if rule.endswith("_at_bound") else (rule, False)
    name = RULE_NAMES.get(base, base)
    return f"{name} (at bound)" if at_bound else name


def short_decision(d) -> str:
    """One line for the bottom bar: 'hunters 2 -> 1 · Eased: heavy damage'."""
    if d is None:
        return ""
    if not d.changed:
        return rule_name(d.rule)
    lever = LEVER_NAMES.get(d.lever, d.lever)
    before, after = d.before[d.lever], d.after[d.lever]
    return f"{lever} {before} -> {after}  ·  {rule_name(d.rule)}"


def _pieces(size: int, word: str, width: int, bold: bool) -> list[str]:
    """Split a single word that is wider than ``width`` into pieces that fit (Phase 10 edge case:
    an overlong token such as an episode id must not spill out of the panel)."""
    font = th.font(size, bold)
    pieces, piece = [], ""
    for ch in word:
        if piece and font.size(piece + ch)[0] > width:
            pieces.append(piece)
            piece = ch
        else:
            piece += ch
    return pieces + [piece] if piece else pieces


def wrap(size: int, s: str, width: int, colour=th.TEXT, bold: bool = False) -> list[pygame.Surface]:
    """Word-wrap ``s`` to ``width`` pixels; returns rendered lines. Words wider than the width are
    broken between characters so no line is ever wider than ``width``."""
    font = th.font(size, bold)
    words: list[str] = []
    for word in s.split():
        words += _pieces(size, word, width, bold) if font.size(word)[0] > width else [word]
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if font.size(trial)[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return [text(size, line, colour, bold) for line in lines]


def elide(size: int, s: str, width: int, bold: bool = False) -> str:
    if th.font(size, bold).size(s)[0] <= width:
        return s
    while s and th.font(size, bold).size(s + "…")[0] > width:
        s = s[:-1]
    return s.rstrip() + "…"


def _section(surface: pygame.Surface, x: int, y: int, title: str) -> int:
    surface.blit(text(th.SIZE["label"], title, th.TEXT_MUTED, True), (x, y))
    return y + 18


def _row_bar(surface: pygame.Surface, x: int, y: int, width: int, name: str, value: str, frac: float, colour) -> int:
    surface.blit(text(th.SIZE["small"], name, th.TEXT_MUTED), (x, y))
    v = text(th.SIZE["small"], value, th.TEXT, True)
    surface.blit(v, (x + width - v.get_width(), y))
    bar(surface, pygame.Rect(x, y + 17, width, 6), frac, colour)
    return y + 30


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, state, params, prediction, fx: dict, t: float) -> None:
    inner = rect.inflate(-th.SP["md"], -th.SP["md"])
    card(surface, inner, fill=th.SURFACE, alpha=235, radius=th.RADIUS["lg"])
    x = inner.x + th.SP["lg"]
    width = inner.width - 2 * th.SP["lg"]
    y = inner.y + th.SP["lg"]

    # header
    icon_chip(surface, x + 9, y + 10, 14, th.CYAN)
    surface.blit(text(th.SIZE["title"], "AI CORE", th.TEXT, True), (x + 28, y - 1))
    y += 26
    meta = fx.get("model_meta")
    if meta:
        f1 = meta.get("test", {}).get("macro_f1")
        line = f"{meta.get('model', '?').replace('_', ' ')}  ·  trained {str(meta.get('trained_at', ''))[:10]}" + (f"  ·  held-out F1 {f1:.2f}" if f1 is not None else "")
        surface.blit(text(th.SIZE["small"], elide(th.SIZE["small"], line, width), th.TEXT_MUTED), (x, y))
    else:
        surface.blit(text(th.SIZE["small"], "no model loaded (run: neuroquest train)", th.TEXT_MUTED), (x, y))
    y += 24

    # player style
    y = _section(surface, x, y, "PLAYER STYLE")
    if prediction is not None and not isinstance(prediction, str):
        colour = LABEL_COLOURS.get(prediction.label, th.TEXT)
        pygame.draw.circle(surface, colour, (x + 7, y + 11), 6)
        surface.blit(text(th.SIZE["value"], prediction.label, colour, True), (x + 20, y - 4))
        conf = text(th.SIZE["small"], f"{prediction.confidence:.0%} confident", th.TEXT_MUTED)
        surface.blit(conf, (x + width - conf.get_width(), y + 4))
        y += 30
        for cls in ("aggressive", "cautious", "explorer"):
            p = prediction.proba.get(cls, 0.0)
            y = _row_bar(surface, x, y, width, cls, f"{p:.0%}", p, LABEL_COLOURS[cls])
    else:
        msg = prediction if isinstance(prediction, str) else "no model loaded"
        if msg.startswith("AI style: "):
            msg = msg[len("AI style: "):]
        for line in wrap(th.SIZE["small"], msg, width, th.TEXT_MUTED):
            surface.blit(line, (x, y))
            y += 16
        y += 8
    y += 6

    # adaptive difficulty
    y = _section(surface, x, y, "ADAPTIVE DIFFICULTY")
    adaptive = bool(fx.get("adaptive", False))
    badge = text(th.SIZE["label"], "ADAPTIVE" if adaptive else "FIXED", th.BG_DEEP, True)
    brect = pygame.Rect(x, y, badge.get_width() + 14, 18)
    pygame.draw.rect(surface, th.CYAN if adaptive else th.TEXT_MUTED, brect, border_radius=9)
    surface.blit(badge, (brect.x + 7, brect.y + 3))
    y += 26
    lo, hi = BOUNDS["enemy_count"]
    y = _row_bar(surface, x, y, width, "hunters", str(params.enemy_count), (params.enemy_count - lo) / (hi - lo), th.CRIMSON)
    lo, hi = BOUNDS["enemy_move_interval"]
    cps = state.ticks_per_second / max(1, params.enemy_move_interval)
    y = _row_bar(surface, x, y, width, "enemy speed", f"every {params.enemy_move_interval} ticks = {cps:.1f} cells/s", (hi - params.enemy_move_interval) / (hi - lo), th.VIOLET)
    lo, hi = BOUNDS["time_limit_s"]
    y = _row_bar(surface, x, y, width, "time limit", f"{params.time_limit_s}s", (params.time_limit_s - lo) / (hi - lo), th.CYAN)
    lo, hi = BOUNDS["reward_density"]
    y = _row_bar(surface, x, y, width, "coin density", f"{params.reward_density:.2f}", (params.reward_density - lo) / (hi - lo), th.AMBER)
    y += 6

    # last decision
    y = _section(surface, x, y, "LAST DECISION")
    decisions = fx.get("decisions") or []
    last = decisions[-1] if decisions else None
    if last is None:
        for line in wrap(th.SIZE["small"], "none yet: the first decision comes after 3 s of play" if adaptive else "controller off (fixed difficulty)", width, th.TEXT_MUTED):
            surface.blit(line, (x, y))
            y += 15
        y += 5
    else:
        when = text(th.SIZE["small"], f"at {last.t / state.ticks_per_second:.0f}s", th.TEXT_MUTED)
        surface.blit(when, (x + width - when.get_width(), y + 2))
        if last.changed:
            lever = LEVER_NAMES.get(last.lever, last.lever)
            surface.blit(text(th.SIZE["body"], elide(th.SIZE["body"], f"{lever} {last.before[last.lever]} -> {last.after[last.lever]}", width - 50, True), th.AMBER, True), (x, y))
            y += 19
            surface.blit(text(th.SIZE["small"], rule_name(last.rule), th.TEXT, True), (x, y))
        else:
            surface.blit(text(th.SIZE["body"], elide(th.SIZE["body"], rule_name(last.rule), width - 50, True), th.TEXT_MUTED, True), (x, y))
        y += 18
        for line in wrap(th.SIZE["small"], last.reason, width, th.TEXT_MUTED)[:5]:
            surface.blit(line, (x, y))
            y += 15
    y += 8

    # recent
    if len(decisions) > 1 and y + 70 < inner.bottom:
        y = _section(surface, x, y, "RECENT")
        for d in reversed(decisions[-5:-1]):
            if y + 16 > inner.bottom - 28:
                break
            mark, colour = ("UP", th.CRIMSON) if d.direction == "tighten" and d.changed else (("DOWN", th.EMERALD) if d.direction == "ease" and d.changed else ("HOLD", th.TEXT_MUTED))
            surface.blit(text(th.SIZE["small"], f"{d.t / state.ticks_per_second:>3.0f}s", th.TEXT_MUTED), (x, y))
            surface.blit(text(th.SIZE["label"], mark, colour, True), (x + 34, y + 2))
            surface.blit(text(th.SIZE["small"], elide(th.SIZE["small"], short_decision(d), width - 80), th.TEXT), (x + 76, y))
            y += 16

    note = text(th.SIZE["label"], elide(th.SIZE["label"], "LOGGED TO reports/decisions.jsonl", width, True), th.TEXT_MUTED, True)
    surface.blit(note, (x, inner.bottom - note.get_height() - th.SP["md"]))
