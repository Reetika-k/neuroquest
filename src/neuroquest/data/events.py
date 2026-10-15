"""Episode logs: the events of one game plus metadata, saved as JSON Lines.

File format: the first line is ``{"meta": {...}}``; every following line is one event
``{"t": tick, "type": ..., "pos": [row, col], "detail": {...}}``. Events are the raw material
for every feature; features are never logged directly, so they can be recomputed later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.state import Event, GameState


def event_to_dict(event: Event) -> dict[str, Any]:
    return {"t": event.t, "type": event.type, "pos": list(event.pos), "detail": event.detail}


def event_from_dict(d: dict[str, Any]) -> Event:
    return Event(t=int(d["t"]), type=str(d["type"]), pos=(int(d["pos"][0]), int(d["pos"][1])), detail=dict(d.get("detail", {})))


@dataclass
class EpisodeLog:
    meta: dict[str, Any]
    events: list[Event] = field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write(json.dumps({"meta": self.meta}) + "\n")
            for event in self.events:
                fh.write(json.dumps(event_to_dict(event)) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> EpisodeLog:
        with path.open() as fh:
            lines = [line for line in fh.read().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"{path}: empty log")
        head = json.loads(lines[0])
        if "meta" not in head:
            raise ValueError(f"{path}: first line must hold the episode meta")
        return cls(meta=head["meta"], events=[event_from_dict(json.loads(line)) for line in lines[1:]])


class EventLogger:
    """Collects events during an episode and fills in the meta at the end."""

    def __init__(self, state: GameState, config: GameConfig, params: DifficultyParams, **meta: Any) -> None:
        distances = state.maze.reachable_from(state.player.pos)
        self.log = EpisodeLog(
            meta={
                **meta,
                "config": asdict(config),
                "params": asdict(params),
                "maze": state.maze.to_strings(),
                "start": list(state.player.pos),
                "exit": list(state.exit),
                "open_cells": len(state.maze.open_cells()),
                "exit_distance": distances[state.exit],
                "n_rewards": len(state.rewards),
                "n_enemies": len(state.enemies),
                "ticks_per_second": state.ticks_per_second,
                "max_health": state.player.health,
            }
        )

    def extend(self, events: list[Event]) -> None:
        self.log.events.extend(events)

    def finish(self, state: GameState) -> EpisodeLog:
        self.log.meta.update(
            {
                "outcome": state.outcome.value,
                "ticks": state.tick,
                "score": state.score,
                "health_end": state.player.health,
                "rewards_collected": self.log.meta["n_rewards"] - len(state.rewards),
            }
        )
        return self.log
