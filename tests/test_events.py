"""Phase 6: episode logs round-trip through JSON Lines."""

import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.rules import new_game, step
from neuroquest.core.state import Action, Event
from neuroquest.data.events import EpisodeLog, EventLogger, event_from_dict, event_to_dict


def test_event_dict_round_trip() -> None:
    event = Event(3, "move", (1, 2), {"action": "right", "new_cell": True})
    assert event_from_dict(event_to_dict(event)) == event


def test_logger_collects_events_and_fills_meta(tmp_path) -> None:
    config, params = GameConfig(), DifficultyParams(enemy_count=0, time_limit_s=1)
    state = new_game(config, params, seed=2)
    logger = EventLogger(state, config, params, episode_id="t-2", profile="test", seed=2)
    for _ in range(10):
        state, events = step(state, Action.RIGHT, params)
        logger.extend(events)
    log = logger.finish(state)
    assert log.meta["outcome"] == "lost_time" and log.meta["ticks"] == 10
    assert log.meta["maze"] == state.maze.to_strings() and log.meta["exit"] == list(state.exit)
    assert log.meta["open_cells"] == len(state.maze.open_cells()) and log.meta["exit_distance"] > 0
    assert log.events[-1].type == "timeout"

    path = log.save(tmp_path / "raw" / "t-2.jsonl")
    loaded = EpisodeLog.load(path)
    assert loaded.meta == log.meta and loaded.events == log.events


def test_load_rejects_bad_files(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        EpisodeLog.load(empty)
    no_meta = tmp_path / "nometa.jsonl"
    no_meta.write_text('{"t": 1, "type": "move", "pos": [1, 1]}\n')
    with pytest.raises(ValueError):
        EpisodeLog.load(no_meta)
