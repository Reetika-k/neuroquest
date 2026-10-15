"""Phase 9b: the maze environment — layout fixing, observation encoding, rewards, termination, guards."""

from dataclasses import replace

import pytest

from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.entities import Enemy
from neuroquest.core.rules import new_game
from neuroquest.core.state import Action, Event, Outcome
from neuroquest.rl.agents import ShortestPathAgent
from neuroquest.rl.env import ACTIONS, ENEMY_WINDOW, FAR, NO_ENEMY, MazeEnv, RewardSpec

TINY = GameConfig(width=7, height=7, loop_fraction=1.0)
NO_HUNTER = DifficultyParams(enemy_count=0, reward_density=0.1, time_limit_s=10)


def test_layout_seed_fixes_board_but_not_enemy_start() -> None:
    config, params = GameConfig(width=11, height=9), DifficultyParams(enemy_count=1)
    games = [new_game(config, params, seed, layout_seed=3) for seed in range(8)]
    assert all(g.maze == games[0].maze and g.exit == games[0].exit and g.rewards == games[0].rewards for g in games)
    assert len({g.enemies[0].pos for g in games}) > 1
    # without a layout seed, everything still follows the seed exactly as before
    assert new_game(config, params, 5) == new_game(config, params, 5, layout_seed=None)
    assert new_game(config, params, 5).maze != new_game(config, params, 6).maze


def test_reset_observation_and_crystal_mask() -> None:
    env = MazeEnv(TINY, NO_HUNTER, layout_seed=1)
    obs, info = env.reset(seed=0)
    assert obs == (1, 1, (1 << len(env.crystals)) - 1, NO_ENEMY, NO_ENEMY)
    assert info["outcome"] == "playing" and info["crystals"] == 0 and env.n_actions == len(ACTIONS) == 5
    # collecting a crystal clears its bit, nothing else changes in the mask
    state = env.state
    first = env.crystals[0]
    after = replace(state, rewards=state.rewards - {first})
    assert env.observe(after)[2] == obs[2] & ~1


def test_enemy_offset_exact_near_and_direction_far() -> None:
    env = MazeEnv(TINY, NO_HUNTER, layout_seed=1)
    env.reset(seed=0)
    base = env.state
    pr, pc = base.player.pos
    near = replace(base, enemies=(Enemy(0, (pr + 1, pc + 2)),))
    assert env.observe(near)[3:] == (1, 2)
    far = replace(base, enemies=(Enemy(0, (pr + ENEMY_WINDOW + 3, pc)),))
    assert env.observe(far)[3:] == (FAR, 0)
    far_left = replace(base, enemies=(Enemy(0, (pr, pc + ENEMY_WINDOW + 1)), Enemy(1, (pr + 4, pc + 4))))
    assert env.observe(far_left)[3:] == (0, FAR)  # the nearest of two is used
    on_top = replace(base, enemies=(Enemy(0, (pr, pc)),))
    assert env.observe(on_top)[3:] == (0, 0)


def test_reward_spec_reads_events_only() -> None:
    spec = RewardSpec()
    assert spec.of([]) == pytest.approx(-0.01)
    assert spec.of([Event(1, "move", (1, 2)), Event(1, "reward", (1, 2))]) == pytest.approx(0.99)
    assert spec.of([Event(1, "exit", (1, 2))]) == pytest.approx(4.99)
    assert spec.of([Event(1, "damage", (1, 2)), Event(1, "defeat", (1, 2))]) == pytest.approx(-7.01)
    assert spec.of([Event(1, "bump", (1, 1)), Event(1, "near_enemy", (1, 1))]) == pytest.approx(-0.01)


def test_shortest_path_episode_rewards_and_termination() -> None:
    env = MazeEnv(TINY, NO_HUNTER, layout_seed=1)
    obs, info = env.reset(seed=0)
    agent = ShortestPathAgent()
    total, steps, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, info["state"]))
        total, steps, done = total + reward, steps + 1, terminated or truncated
    assert terminated and not truncated and info["won"] and env.state.outcome is Outcome.WON
    assert total == pytest.approx(-0.01 * steps + 5.0 + info["crystals"] * 1.0)
    assert steps == env.maze.reachable_from((1, 1))[env.exit]


def test_timeout_is_truncation_not_termination() -> None:
    env = MazeEnv(TINY, DifficultyParams(enemy_count=0, reward_density=0.1, time_limit_s=1), layout_seed=1)
    env.reset(seed=0)
    stay = ACTIONS.index(Action.STAY)
    for _ in range(9):
        _, reward, terminated, truncated, _ = env.step(stay)
        assert reward == pytest.approx(-0.01) and not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(stay)
    assert truncated and not terminated and info["outcome"] == "lost_time"


def test_hunter_contact_is_penalised_and_can_end_the_episode() -> None:
    config = GameConfig(width=7, height=7, loop_fraction=1.0, enemy_kind="random", max_health=1)
    env = MazeEnv(config, DifficultyParams(enemy_count=1, enemy_move_interval=1, reward_density=0.1, time_limit_s=30), layout_seed=1)
    stay = ACTIONS.index(Action.STAY)
    hit_seen = False
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        while not done:
            _, reward, terminated, truncated, info = env.step(stay)
            done = terminated or truncated
            if info["hits"]:
                hit_seen = True
                assert reward == pytest.approx(-0.01 - 2.0 - 5.0) and terminated and info["outcome"] == "lost_health"
        if hit_seen:
            break
    assert hit_seen, "a random-walk hunter never reached a motionless player in 20 episodes"


def test_step_guards_and_describe() -> None:
    env = MazeEnv(TINY, NO_HUNTER, layout_seed=1)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(5)
    d = env.describe()
    assert d["board"] == "7x7" and d["crystals"] == len(env.crystals) and d["actions"] == ["up", "down", "left", "right", "stay"]
    assert d["state_space_bound"] == len(env.maze.open_cells()) * 2 ** len(env.crystals)
    assert MazeEnv().describe()["state_space_bound"] == len(MazeEnv().maze.open_cells()) * 2 ** len(MazeEnv().crystals) * 34
