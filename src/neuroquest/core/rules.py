"""The rules of NeuroQuest: how a game starts and how one tick advances it.

``step()`` is the single function every consumer uses: the Pygame loop, the headless
simulation harness, the RL environment and the tests. It has no side effects.

Order inside a tick: the player moves (or bumps) → a reward on the new cell is collected →
the exit is checked (a win ends the tick) → enemies move on their interval → enemy contact
does damage (once per invulnerability window) → a ``near_enemy`` event is recorded when an
enemy is within two cells → the clock is checked.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from neuroquest.config import REWARD_VALUE, DifficultyParams, GameConfig
from neuroquest.core.entities import Enemy, Player
from neuroquest.core.maze import Cell, Maze, manhattan
from neuroquest.core.state import Action, Event, GameState, Outcome

if TYPE_CHECKING:  # the Protocol lives in ai/; importing it here at runtime would create a cycle
    from neuroquest.ai.policies import EnemyPolicy

START: Cell = (1, 1)
NEAR_DISTANCE = 2  # "near an enemy" means within this Manhattan distance
FAR = 10**9


def new_game(config: GameConfig, params: DifficultyParams, seed: int, *, layout_seed: int | None = None) -> GameState:
    """Build a fresh, fully determined game from ``seed``.

    The exit is the open cell farthest (by walking distance) from the start. Rewards are sampled
    from the remaining open cells; enemies from cells at least ``enemy_min_start_distance`` of the
    farthest distance away, so the player always has a few seconds before the chase begins.

    ``layout_seed`` (optional, used by the RL environment) fixes the maze, exit and rewards to
    one layout while ``seed`` still varies the enemies' start cells. Left as None, ``seed``
    determines everything, exactly as before.
    """
    rng = random.Random(seed)
    layout_rng = rng if layout_seed is None else random.Random(layout_seed)
    maze = Maze.generate(config.width, config.height, layout_rng, loop_fraction=config.loop_fraction)
    distances = maze.reachable_from(START)
    exit_cell = max(distances, key=lambda cell: (distances[cell], cell))
    open_cells = maze.open_cells()
    candidates = [cell for cell in open_cells if cell not in (START, exit_cell)]
    n_rewards = min(max(1, round(params.reward_density * len(open_cells))), len(candidates))
    rewards = frozenset(layout_rng.sample(candidates, n_rewards))

    farthest = max(distances.values())
    free = [cell for cell in candidates if cell not in rewards]
    enemy_candidates = [cell for cell in free if distances[cell] >= config.enemy_min_start_distance * farthest] or free
    n_enemies = min(max(0, params.enemy_count), len(enemy_candidates))
    enemies = tuple(
        Enemy(id=i, pos=pos, kind=config.enemy_kind) for i, pos in enumerate(rng.sample(enemy_candidates, n_enemies))
    )
    return GameState(
        maze=maze,
        player=Player(pos=START, health=config.max_health),
        rewards=rewards,
        exit=exit_cell,
        visited=frozenset({START}),
        enemies=enemies,
        ticks_per_second=config.ticks_per_second,
        invulnerability_ticks=config.invulnerability_ticks,
    )


def time_limit_ticks(state: GameState, params: DifficultyParams) -> int:
    return params.time_limit_s * state.ticks_per_second


def time_left_s(state: GameState, params: DifficultyParams) -> float:
    return max(0.0, params.time_limit_s - state.elapsed_s)


def min_enemy_distance(pos: Cell, enemies: tuple[Enemy, ...]) -> int:
    return min((manhattan(pos, enemy.pos) for enemy in enemies), default=FAR)


def step(
    state: GameState,
    action: Action,
    params: DifficultyParams,
    rng: random.Random | None = None,
    policies: Mapping[str, EnemyPolicy] | None = None,
) -> tuple[GameState, list[Event]]:
    """Advance the game by one tick. A finished game returns itself unchanged with no events.

    ``rng`` drives random enemy policies; pass one ``random.Random(seed)`` per game for
    reproducible episodes. ``policies`` maps ``Enemy.kind`` to a movement policy; by default the
    registry in ``ai/policies.py`` is used.
    """
    if state.outcome.is_over:
        return state, []

    tick = state.tick + 1
    events: list[Event] = []
    player, rewards, visited, score = state.player, state.rewards, state.visited, state.score
    enemies_before = state.enemies
    distance_before = min_enemy_distance(player.pos, enemies_before)
    moved = False

    if action is not Action.STAY:
        dr, dc = action.delta
        target = (player.pos[0] + dr, player.pos[1] + dc)
        if state.maze.is_open(target):
            events.append(Event(tick, "move", target, {"action": action.value, "new_cell": target not in visited}))
            player = replace(player, pos=target)
            visited = visited | {target}
            moved = True
        else:
            events.append(Event(tick, "bump", player.pos, {"action": action.value}))

    if player.pos in rewards:
        rewards = rewards - {player.pos}
        score += REWARD_VALUE
        events.append(Event(tick, "reward", player.pos, {"score": score, "remaining": len(rewards)}))

    if player.pos == state.exit:
        events.append(Event(tick, "exit", player.pos, {"score": score}))
        won = replace(state, player=player, rewards=rewards, visited=visited, tick=tick, score=score, outcome=Outcome.WON)
        return won, events

    enemies = enemies_before
    outcome = Outcome.PLAYING
    if enemies:
        if tick % max(1, params.enemy_move_interval) == 0:
            interim = replace(state, player=player, rewards=rewards, visited=visited, tick=tick, score=score)
            enemies = _move_enemies(enemies, interim, rng or random.Random(tick), policies)

        hitter = next((enemy for enemy in enemies if enemy.pos == player.pos), None)
        if hitter is not None and tick >= player.invulnerable_until:
            health = max(0, player.health - 1)
            player = replace(player, health=health, invulnerable_until=tick + state.invulnerability_ticks)
            events.append(Event(tick, "damage", player.pos, {"amount": 1, "source": f"enemy:{hitter.id}", "health": health}))
            if health == 0:
                outcome = Outcome.LOST_HEALTH
                events.append(Event(tick, "defeat", player.pos, {"source": f"enemy:{hitter.id}"}))

        distance_after = min_enemy_distance(player.pos, enemies)
        if distance_after <= NEAR_DISTANCE:
            detail: dict = {"distance": distance_after}
            if moved:  # did the player's own move bring it closer to the nearest enemy?
                detail["approach"] = min_enemy_distance(player.pos, enemies_before) < distance_before
            events.append(Event(tick, "near_enemy", player.pos, detail))

    if outcome is Outcome.PLAYING and tick >= time_limit_ticks(state, params):
        outcome = Outcome.LOST_TIME
        events.append(Event(tick, "timeout", player.pos, {"score": score}))

    next_state = replace(
        state, player=player, rewards=rewards, visited=visited, enemies=enemies, tick=tick, score=score, outcome=outcome
    )
    return next_state, events


def _move_enemies(
    enemies: tuple[Enemy, ...],
    interim: GameState,
    rng: random.Random,
    policies: Mapping[str, EnemyPolicy] | None,
) -> tuple[Enemy, ...]:
    """Ask each enemy's policy for its next cell; moves that are not to an adjacent open cell are ignored."""
    if policies is None:
        from neuroquest.ai.policies import DEFAULT_POLICIES  # local import: keeps core free of a hard dependency on ai

        policies = DEFAULT_POLICIES
    moved = []
    for enemy in enemies:
        policy = policies.get(enemy.kind)
        target = policy.next_cell(enemy, interim, rng) if policy is not None else enemy.pos
        if target != enemy.pos and target not in interim.maze.neighbours(enemy.pos):
            target = enemy.pos
        moved.append(replace(enemy, pos=target))
    return tuple(moved)


def apply_damage(state: GameState, amount: int = 1, source: str = "hazard") -> tuple[GameState, list[Event]]:
    """Reduce health from a non-enemy source; at zero the game is lost."""
    if state.outcome.is_over or amount <= 0:
        return state, []
    health = max(0, state.player.health - amount)
    events = [Event(state.tick, "damage", state.player.pos, {"amount": amount, "source": source, "health": health})]
    outcome = state.outcome
    if health == 0:
        outcome = Outcome.LOST_HEALTH
        events.append(Event(state.tick, "defeat", state.player.pos, {"source": source}))
    return replace(state, player=replace(state.player, health=health), outcome=outcome), events


def reconcile(
    state: GameState, params: DifficultyParams, rng: random.Random, previous: DifficultyParams | None = None
) -> tuple[GameState, list[Event]]:
    """Bring a live game in line with *changed* difficulty parameters.

    Enemy count: spawn new enemies far from the player, or remove the farthest ones. Reward
    availability: only when the density is *higher than before* (``previous``; None means "treat
    as higher"), top up rewards to ``reward_density × open cells`` on unvisited cells. A lower or
    unchanged density never adds or removes rewards, so collecting rewards is never refunded by
    an unrelated change. Speed and time limit need no reconciliation: ``step`` reads them from
    ``params`` every tick. Called by the adaptive controller, never by ``step`` itself.
    """
    if state.outcome.is_over:
        return state, []
    events: list[Event] = []
    enemies = list(state.enemies)
    rewards = set(state.rewards)
    open_cells = state.maze.open_cells()
    occupied = {state.player.pos, state.exit} | rewards | {e.pos for e in enemies}

    target_enemies = max(0, params.enemy_count)
    if len(enemies) < target_enemies:
        distances = state.maze.reachable_from(state.player.pos)
        farthest = max(distances.values())
        far = [c for c, d in distances.items() if d >= 0.5 * farthest and c not in occupied]
        candidates = far or [c for c in distances if c not in occupied]
        kind = enemies[0].kind if enemies else "astar"
        next_id = max((e.id for e in enemies), default=-1) + 1
        while len(enemies) < target_enemies and candidates:
            pos = rng.choice(candidates)
            candidates.remove(pos)
            occupied.add(pos)
            enemies.append(Enemy(id=next_id, pos=pos, kind=kind))
            events.append(Event(state.tick, "enemy_spawn", pos, {"id": next_id}))
            next_id += 1
    elif len(enemies) > target_enemies:
        enemies.sort(key=lambda e: manhattan(e.pos, state.player.pos))  # keep the nearest, remove the farthest
        while len(enemies) > target_enemies:
            gone = enemies.pop()
            events.append(Event(state.tick, "enemy_despawn", gone.pos, {"id": gone.id}))

    density_raised = previous is None or params.reward_density > previous.reward_density
    target_rewards = round(params.reward_density * len(open_cells))
    if density_raised and len(rewards) < target_rewards:
        fresh = [c for c in open_cells if c not in occupied and c not in state.visited]
        candidates = fresh or [c for c in open_cells if c not in occupied]
        while len(rewards) < target_rewards and candidates:
            pos = rng.choice(candidates)
            candidates.remove(pos)
            rewards.add(pos)
            events.append(Event(state.tick, "reward_spawn", pos, {}))

    return replace(state, enemies=tuple(enemies), rewards=frozenset(rewards)), events
