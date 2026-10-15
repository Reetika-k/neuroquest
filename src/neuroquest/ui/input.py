"""Keyboard mapping. Arrow keys or WASD move; a held key repeats on every logic tick."""

from __future__ import annotations

import pygame

from neuroquest.core.state import Action

# Dictionary order is the priority order when several keys are held.
KEY_TO_ACTION: dict[int, Action] = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
}


def action_from_pressed(pressed) -> Action:
    """Turn the held-key table (``pygame.key.get_pressed()``) into one Action."""
    for key, action in KEY_TO_ACTION.items():
        if pressed[key]:
            return action
    return Action.STAY
