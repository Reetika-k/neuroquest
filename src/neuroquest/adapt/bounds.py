"""Safety limits for the difficulty knobs. Whatever the controller decides is clamped here, so the
game can never become unplayable (no enemies at all, enemies as fast as the player, a 10-second clock)."""

from __future__ import annotations

from dataclasses import replace

from neuroquest.config import DifficultyParams

BOUNDS: dict[str, tuple[float, float]] = {
    "enemy_count": (1, 5),
    "enemy_move_interval": (2, 8),  # 2 = half the player's speed (fast), 8 = a quarter of it (slow)
    "reward_density": (0.05, 0.25),
    "time_limit_s": (60, 240),
}

STEP: dict[str, float] = {"enemy_count": 1, "enemy_move_interval": 1, "reward_density": 0.05, "time_limit_s": 20}


def clamp(params: DifficultyParams) -> DifficultyParams:
    """Return a copy with every knob inside its bounds (integers stay integers)."""
    values = {}
    for name, (lo, hi) in BOUNDS.items():
        value = getattr(params, name)
        clamped = min(max(value, lo), hi)
        values[name] = int(clamped) if isinstance(lo, int) else round(float(clamped), 4)
    return replace(params, **values)


def within_bounds(params: DifficultyParams) -> bool:
    return all(lo <= getattr(params, name) <= hi for name, (lo, hi) in BOUNDS.items())
