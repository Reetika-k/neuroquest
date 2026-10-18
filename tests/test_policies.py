"""Phase 5: enemy policies and how step() applies them."""

import random

from neuroquest.ai.bfs import bfs
from neuroquest.ai.policies import DEFAULT_POLICIES, AStarChaser, RandomWalker
from neuroquest.config import DifficultyParams, GameConfig
from neuroquest.core.entities import Enemy, Player
from neuroquest.core.maze import Maze
from neuroquest.core.rules import new_game, step
from neuroquest.core.state import Action, GameState, Outcome

ROWS = [
    "#########",
    "#.......#",
    "#.#####.#",
    "#.......#",
    "#########",
]
PARAMS = DifficultyParams(enemy_count=1, enemy_move_interval=1, time_limit_s=100)


def make_state(enemy_pos, player_pos=(1, 1), kind="astar", health=3) -> GameState:
    return GameState(
        maze=Maze.from_strings(ROWS),
        player=Player(player_pos, health),
        rewards=frozenset(),
        exit=(3, 7),
        visited=frozenset({player_pos}),
        enemies=(Enemy(0, enemy_pos, kind),),
    )


def test_astar_chaser_takes_the_first_step_of_the_shortest_path() -> None:
    state = make_state(enemy_pos=(3, 1))
    nxt = AStarChaser().next_cell(state.enemies[0], state, random.Random(0))
    assert nxt == (2, 1)  # straight up the only corridor


def test_astar_chaser_stays_when_already_on_the_player() -> None:
    state = make_state(enemy_pos=(1, 1))
    assert AStarChaser().next_cell(state.enemies[0], state, random.Random(0)) == (1, 1)


def test_random_walker_moves_to_an_open_neighbour() -> None:
    state = make_state(enemy_pos=(1, 4), kind="random")
    for seed in range(10):
        nxt = RandomWalker().next_cell(state.enemies[0], state, random.Random(seed))
        assert nxt in state.maze.neighbours((1, 4))


def test_step_moves_enemies_only_on_their_interval() -> None:
    state = make_state(enemy_pos=(3, 7))
    params = DifficultyParams(enemy_count=1, enemy_move_interval=3, time_limit_s=100)
    positions = []
    for _ in range(6):
        state, _ = step(state, Action.STAY, params, random.Random(0))
        positions.append(state.enemies[0].pos)
    assert positions[0] == positions[1] == (3, 7)  # ticks 1, 2: no move
    assert positions[2] != (3, 7)  # tick 3: moved
    assert positions[2] == positions[3] == positions[4]  # ticks 4, 5: no move
    assert positions[5] != positions[4]  # tick 6: moved again


def test_invalid_policy_moves_are_ignored() -> None:
    class Teleporter:
        name = "teleport"

        def next_cell(self, enemy, state, rng):
            return (3, 7)  # not adjacent to (1, 1)

    state = make_state(enemy_pos=(1, 7), kind="teleport")
    nxt, _ = step(state, Action.STAY, PARAMS, random.Random(0), policies={"teleport": Teleporter()})
    assert nxt.enemies[0].pos == (1, 7)


def test_unknown_policy_kind_keeps_the_enemy_still() -> None:
    state = make_state(enemy_pos=(1, 7), kind="ghost")
    nxt, _ = step(state, Action.STAY, PARAMS, random.Random(0))
    assert nxt.enemies[0].pos == (1, 7)


def test_astar_enemy_reaches_a_stationary_player_in_exactly_path_length_ticks() -> None:
    state = make_state(enemy_pos=(3, 7))
    expected = len(bfs(state.maze, (3, 7), (1, 1))) - 1
    ticks = 0
    while not any(e.pos == state.player.pos for e in state.enemies):
        state, _ = step(state, Action.STAY, PARAMS, random.Random(0))
        ticks += 1
    assert ticks == expected


def test_contact_damages_once_per_invulnerability_window() -> None:
    state = make_state(enemy_pos=(1, 2))
    hit, events = step(state, Action.STAY, PARAMS, random.Random(0))
    assert hit.player.health == 2 and [e.type for e in events] == ["damage", "near_enemy"]
    assert events[0].detail["source"] == "enemy:0"
    for _ in range(14):  # still inside the 15-tick window
        hit, events = step(hit, Action.STAY, PARAMS, random.Random(0))
        assert all(e.type != "damage" for e in events)
    assert hit.player.health == 2
    again, events = step(hit, Action.STAY, PARAMS, random.Random(0))
    assert again.player.health == 1 and any(e.type == "damage" for e in events)


def test_player_walking_into_an_enemy_takes_damage() -> None:
    state = make_state(enemy_pos=(1, 2))
    params = DifficultyParams(enemy_count=1, enemy_move_interval=100, time_limit_s=100)  # enemy stands still
    hit, events = step(state, Action.RIGHT, params, random.Random(0))
    assert hit.player.pos == (1, 2) and hit.player.health == 2
    assert [e.type for e in events] == ["move", "damage", "near_enemy"]


def test_losing_all_health_to_enemies_ends_the_game() -> None:
    state = make_state(enemy_pos=(1, 2), health=1)
    dead, events = step(state, Action.STAY, PARAMS, random.Random(0))
    assert dead.outcome is Outcome.LOST_HEALTH
    assert [e.type for e in events][:2] == ["damage", "defeat"]
    frozen, events = step(dead, Action.STAY, PARAMS, random.Random(0))
    assert frozen == dead and events == []


def test_near_enemy_event_reports_distance_and_approach() -> None:
    state = make_state(enemy_pos=(1, 4))
    params = DifficultyParams(enemy_count=1, enemy_move_interval=100, time_limit_s=100)
    _, events = step(state, Action.STAY, params)  # distance 3: not near
    assert all(e.type != "near_enemy" for e in events)
    moved, events = step(state, Action.RIGHT, params)  # now at (1, 2), distance 2, moved toward
    near = [e for e in events if e.type == "near_enemy"]
    assert near and near[0].detail == {"distance": 2, "approach": True}
    _, events = step(moved, Action.LEFT, params)  # back to (1, 1): distance 3, no event
    assert all(e.type != "near_enemy" for e in events)


def test_reaching_the_exit_wins_even_with_an_enemy_adjacent() -> None:
    state = make_state(enemy_pos=(3, 6), player_pos=(3, 6))  # player and enemy share the cell next to the exit
    won, events = step(state, Action.RIGHT, PARAMS, random.Random(0))
    assert won.outcome is Outcome.WON and all(e.type != "damage" for e in events)


def test_new_game_places_enemies_away_from_the_player() -> None:
    config, params = GameConfig(), DifficultyParams(enemy_count=3)
    state = new_game(config, params, seed=5)
    assert len(state.enemies) == 3 and all(e.kind == "astar" for e in state.enemies)
    distances = state.maze.reachable_from(state.player.pos)
    farthest = max(distances.values())
    for enemy in state.enemies:
        assert state.maze.is_open(enemy.pos)
        assert enemy.pos not in state.rewards and enemy.pos != state.exit
        assert distances[enemy.pos] >= config.enemy_min_start_distance * farthest
    assert new_game(config, DifficultyParams(enemy_count=0), seed=5).enemies == ()
    assert all(e.kind == "random" for e in new_game(GameConfig(enemy_kind="random"), params, seed=5).enemies)
    assert set(DEFAULT_POLICIES) == {"astar", "random", "anticipator"}  # anticipator added in Phase 7b
