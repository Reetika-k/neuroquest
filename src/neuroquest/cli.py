"""Command-line entry point.

Usage::

    python -m neuroquest --version
    python -m neuroquest check                 # environment smoke test (headless)
    python -m neuroquest play [--seed N]       # play the game (Phase 4; enemies from Phase 5)
    python -m neuroquest benchmark-enemies     # A* vs random chaser, A* vs BFS (Phase 5)
    python -m neuroquest simulate              # scripted player profiles → data/raw/*.jsonl (Phase 6)
    python -m neuroquest build-dataset         # data/raw → data/processed/episodes.csv (Phase 6)
    python -m neuroquest train                 # compare models, save the best + metadata (Phase 7)
    python -m neuroquest ab-test               # fixed vs adaptive difficulty on simulated players (Phase 8)
    python -m neuroquest dashboard             # figures + reports/dashboard.html from recorded data (Phase 9)
    python -m neuroquest rl-train              # Q-learning agent vs baselines on the RL board (Phase 9b)
    python -m neuroquest play --agent          # watch the trained agent play its board (Phase 9b)
    python -m neuroquest build-moves           # episode logs → data/processed/moves.csv (Phase 7b)
    python -m neuroquest train-next-move       # next-move model vs baselines, temporal split (Phase 7b)
    python -m neuroquest benchmark-pursuit     # anticipating hunters vs plain A* against moving players (Phase 7b)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from neuroquest import __version__


def cmd_check(_: argparse.Namespace) -> int:
    """Smoke test: import every runtime dependency and start Pygame without a window."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    import matplotlib
    import numpy
    import pandas
    import sklearn

    matplotlib.use("Agg")

    import pygame

    pygame.init()
    surface = pygame.Surface((64, 64))
    surface.fill((30, 30, 30))
    assert surface.get_at((0, 0))[:3] == (30, 30, 30)
    pygame.quit()

    print(f"neuroquest    {__version__}")
    print(f"python        {sys.version.split()[0]}")
    for name, module in (
        ("pygame-ce", pygame),
        ("numpy", numpy),
        ("pandas", pandas),
        ("scikit-learn", sklearn),
        ("matplotlib", matplotlib),
    ):
        print(f"{name:<14}{module.__version__}")
    print("pygame headless init: ok")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Open the game window."""
    from neuroquest.config import DifficultyParams, GameConfig
    from neuroquest.ui.app import run

    width, height = args.width, args.height
    if width % 2 == 0 or height % 2 == 0:
        width += width % 2 == 0
        height += height % 2 == 0
        print(f"maze dimensions must be odd; using {width}x{height}")
    config = GameConfig(width=width, height=height, enemy_kind=args.enemy_kind)
    params = DifficultyParams(
        time_limit_s=args.time_limit,
        reward_density=args.reward_density,
        enemy_count=args.enemies,
        enemy_move_interval=args.enemy_interval,
    )
    from pathlib import Path

    log_dir = None if args.no_log else Path("data/raw")
    agent = layout_seed = None
    adaptive = not args.fixed
    if args.agent:
        # The learned policy only knows its training board, so the board, difficulty and layout
        # are forced to the ones it trained on; the adaptive controller stays off (it would move
        # the difficulty away from the training conditions) and nothing is logged as human play.
        from neuroquest.rl.agents import QAgent
        from neuroquest.rl.env import ACTIONS, LAYOUT_SEED, RL_CONFIG, RL_PARAMS, MazeEnv
        from neuroquest.rl.pipeline import load_q_table

        try:
            table = load_q_table()
        except FileNotFoundError as err:
            print(err)
            return 1
        env, q = MazeEnv(), QAgent(table, seed=args.seed)
        agent = lambda state: ACTIONS[q.act(env.observe(state), state)]  # noqa: E731
        config, params, layout_seed, log_dir, adaptive = RL_CONFIG, RL_PARAMS, LAYOUT_SEED, None, False
        print(f"Q-learning agent playing its training board (layout seed {LAYOUT_SEED}); R restarts with a new hunter start")
    outcome = run(config, params, seed=args.seed, log_dir=log_dir, adaptive=adaptive, decisions_path=Path("reports/decisions.jsonl"), panel=not args.no_panel, menu=not args.no_menu, agent=agent, layout_seed=layout_seed)
    print(f"outcome: {outcome.value}")
    if log_dir is not None:
        print(f"episode logged to {log_dir}/ (profile=human); use --no-log to disable")
    return 0


def cmd_build_moves(args: argparse.Namespace) -> int:
    """Replay every simulated episode log and write the next-move rows."""
    from pathlib import Path

    from neuroquest.ml.next_move import build_moves

    df, report = build_moves(Path(args.raw), Path(args.out))
    print(f"episodes replayed: {report['episodes']}  rows: {report['rows']}  skipped (human/adaptive): {len(report['skipped'])}  replay failed: {len(report['replay_failed'])}")
    for name in report["replay_failed"][:10]:
        print(f"  replay mismatch: {name}")
    print("targets: " + ", ".join(f"{k} {v}" for k, v in report["targets"].items()))
    print(f"written {args.out}")
    return 0


def cmd_train_next_move(args: argparse.Namespace) -> int:
    """Train and compare next-move predictors with a temporal split; save the best."""
    from pathlib import Path

    from neuroquest.ml.next_move import format_next_move_report, train_next_move

    report = train_next_move(Path(args.data), model_dir=Path(args.model_dir), report_path=Path(args.report), seed=args.seed, fast=args.fast)
    print(format_next_move_report(report))
    print(f"model saved to {report['artifacts']['model']}; full report in {args.report}")
    return 0


def cmd_benchmark_pursuit(args: argparse.Namespace) -> int:
    """Anticipating hunters (next-move model) vs plain A* hunters against moving scripted players."""
    from pathlib import Path

    from neuroquest.ai.benchmark import format_pursuit, pursuit_benchmark, summarize_pursuit, write_csv

    rows = pursuit_benchmark(episodes=args.episodes, seed0=args.seed)
    write_csv(rows, Path(args.out))
    print(format_pursuit(summarize_pursuit(rows)))
    print(f"rows written to {args.out}")
    return 0


def cmd_rl_train(args: argparse.Namespace) -> int:
    """Train the Q-learning agent, evaluate it against the baselines, write the files, print the table."""
    from pathlib import Path

    from neuroquest.rl.pipeline import format_report, run_pipeline

    report = run_pipeline(episodes=args.episodes, seed=args.seed, eval_episodes=args.eval_episodes, model_dir=Path(args.model_dir), report_dir=Path(args.report_dir), progress=print)
    print(format_report(report))
    print(f"files: {', '.join(report['artifacts'].values())}")
    return 0


def cmd_benchmark_enemies(args: argparse.Namespace) -> int:
    """Run the chase and search benchmarks, write CSVs, print a summary."""
    from pathlib import Path

    from neuroquest.ai.benchmark import (
        chase_benchmark,
        format_summary,
        search_benchmark,
        summarize_chase,
        summarize_search,
        write_csv,
    )

    out = Path(args.out)
    chase_rows = chase_benchmark(episodes=args.episodes, max_ticks=args.max_ticks)
    search_rows = search_benchmark(mazes=args.mazes)
    write_csv(chase_rows, out / "enemy_baseline.csv")
    write_csv(search_rows, out / "search_benchmark.csv")
    print(format_summary(summarize_chase(chase_rows), summarize_search(search_rows)))
    print(f"rows written to {out / 'enemy_baseline.csv'} and {out / 'search_benchmark.csv'}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Run scripted player profiles headlessly and save one JSONL log per episode."""
    from pathlib import Path

    from neuroquest.config import DifficultyParams, GameConfig
    from neuroquest.sim.harness import run_batch

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    config = GameConfig(enemy_kind=args.enemy_kind)
    params = DifficultyParams(
        time_limit_s=args.time_limit,
        enemy_count=args.enemies,
        enemy_move_interval=args.enemy_interval,
        reward_density=args.reward_density,
    )
    paths = run_batch(profiles, args.episodes, config, params, args.seed, Path(args.out), progress=print)
    print(f"{len(paths)} episode logs written to {args.out}")
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Turn raw episode logs into the validated feature table and print an inspection summary."""
    from pathlib import Path

    from neuroquest.data.dataset import build, summary

    df, report = build(Path(args.raw), Path(args.out))
    print(f"episodes: {report['episodes']}  rows: {report['rows']}  invalid rows: {len(report['invalid'])}")
    for ident, problems in report["invalid"][:10]:
        print(f"  {ident}: {'; '.join(problems)}")
    print(summary(df))
    print(f"written {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run the training protocol and print the comparison table."""
    from pathlib import Path

    from neuroquest.ml.train import format_report, train

    report = train(Path(args.data), model_dir=Path(args.model_dir), report_path=Path(args.report), seed=args.seed, fast=args.fast)
    print(format_report(report))
    print(f"model saved to {report['artifacts']['model']}; full report in {args.report}")
    return 0


def cmd_ab_test(args: argparse.Namespace) -> int:
    """Fixed vs adaptive difficulty: same profiles, same seeds, both conditions; CSV + summary."""
    from pathlib import Path

    from neuroquest.config import DifficultyParams, GameConfig
    from neuroquest.sim.ab import ab_compare, deltas, summarize_ab

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    rows = ab_compare(profiles, args.episodes, GameConfig(), DifficultyParams(), args.seed, out_csv=Path(args.out), progress=print)
    print(summarize_ab(rows))
    for name, d in deltas(rows).items():
        print(f"  {name:<11}adaptive − fixed: win rate {d['win_rate']:+.0%}, median ticks {d['median_ticks']:+.0f}, mean damage {d['mean_damage']:+.2f}")
    print(f"rows written to {args.out}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Build the explainability dashboard from whatever recorded data exists."""
    from pathlib import Path

    from neuroquest.dashboard.report import build

    summary = build(out_html=Path(args.out), figures_dir=Path(args.figures_dir))
    print(f"sections: {', '.join(summary['sections'])}")
    print(f"figures: {len(summary['figures'])} written to {args.figures_dir}/")
    print(f"dashboard: {summary['html']}  (open it in a browser)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser; each phase registers its own sub-command here."""
    parser = argparse.ArgumentParser(prog="neuroquest", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="environment smoke test (headless)").set_defaults(func=cmd_check)

    play = sub.add_parser("play", help="play the game")
    play.add_argument("--seed", type=int, default=1, help="maze seed (R in-game restarts with seed+1)")
    play.add_argument("--width", type=int, default=21, help="maze width in cells (odd)")
    play.add_argument("--height", type=int, default=15, help="maze height in cells (odd)")
    play.add_argument("--time-limit", type=int, default=120, help="seconds to reach the exit")
    play.add_argument("--reward-density", type=float, default=0.10, help="fraction of open cells with a reward")
    play.add_argument("--enemies", type=int, default=2, help="number of enemies")
    play.add_argument("--enemy-interval", type=int, default=4, help="ticks between enemy moves (smaller is faster)")
    play.add_argument("--enemy-kind", choices=["astar", "random", "anticipator"], default="astar", help="enemy movement policy (anticipator uses the next-move model)")
    play.add_argument("--no-log", action="store_true", help="do not save the episode log to data/raw")
    play.add_argument("--fixed", action="store_true", help="keep the difficulty fixed (no adaptive controller)")
    play.add_argument("--no-panel", action="store_true", help="start with the AI intelligence panel closed (I toggles it)")
    play.add_argument("--no-menu", action="store_true", help="skip the start screen")
    play.add_argument("--agent", action="store_true", help="let the trained Q-learning agent play its training board (needs models/q_table.npz)")
    play.set_defaults(func=cmd_play)

    rl = sub.add_parser("rl-train", help="train the Q-learning agent and compare it with the baselines")
    rl.add_argument("--episodes", type=int, default=20_000, help="training episodes (ε-greedy)")
    rl.add_argument("--seed", type=int, default=0, help="exploration seed")
    rl.add_argument("--eval-episodes", type=int, default=100, help="held-out episodes per agent")
    rl.add_argument("--model-dir", default="models")
    rl.add_argument("--report-dir", default="reports")
    rl.set_defaults(func=cmd_rl_train)

    bench = sub.add_parser("benchmark-enemies", help="A* chaser vs random walker, A* vs BFS")
    bench.add_argument("--episodes", type=int, default=100, help="seeded chase episodes per enemy kind")
    bench.add_argument("--max-ticks", type=int, default=600, help="give up a chase after this many ticks")
    bench.add_argument("--mazes", type=int, default=200, help="seeded mazes for the search comparison")
    bench.add_argument("--out", default="reports", help="output folder for the CSV files")
    bench.set_defaults(func=cmd_benchmark_enemies)

    sim = sub.add_parser("simulate", help="run scripted player profiles headlessly and log episodes")
    sim.add_argument("--profiles", default="aggressive,cautious,explorer", help="comma-separated profile names")
    sim.add_argument("--episodes", type=int, default=120, help="episodes per profile (same seeds across profiles)")
    sim.add_argument("--seed", type=int, default=0, help="first seed")
    sim.add_argument("--out", default="data/raw", help="folder for the JSONL logs")
    sim.add_argument("--time-limit", type=int, default=120)
    sim.add_argument("--enemies", type=int, default=2)
    sim.add_argument("--enemy-interval", type=int, default=4)
    sim.add_argument("--reward-density", type=float, default=0.10)
    sim.add_argument("--enemy-kind", choices=["astar", "random", "anticipator"], default="astar")
    sim.set_defaults(func=cmd_simulate)

    moves = sub.add_parser("build-moves", help="replay episode logs → next-move rows (data/processed/moves.csv)")
    moves.add_argument("--raw", default="data/raw", help="folder with *.jsonl episode logs")
    moves.add_argument("--out", default="data/processed/moves.csv")
    moves.set_defaults(func=cmd_build_moves)

    nm = sub.add_parser("train-next-move", help="next-move model vs majority / repeat-last / greedy baselines, temporal split")
    nm.add_argument("--data", default="data/processed/moves.csv")
    nm.add_argument("--model-dir", default="models")
    nm.add_argument("--report", default="reports/next_move_metrics.json")
    nm.add_argument("--seed", type=int, default=42)
    nm.add_argument("--fast", action="store_true", help="fewer trees / repeats (for quick checks)")
    nm.set_defaults(func=cmd_train_next_move)

    pursuit = sub.add_parser("benchmark-pursuit", help="anticipating hunters vs plain A* hunters against moving scripted players")
    pursuit.add_argument("--episodes", type=int, default=40, help="seeded episodes per profile and hunter kind")
    pursuit.add_argument("--seed", type=int, default=2000, help="first seed (2000+ are used by no other experiment)")
    pursuit.add_argument("--out", default="reports/pursuit_benchmark.csv")
    pursuit.set_defaults(func=cmd_benchmark_pursuit)

    build = sub.add_parser("build-dataset", help="raw logs → validated feature CSV + schema file")
    build.add_argument("--raw", default="data/raw", help="folder with *.jsonl episode logs")
    build.add_argument("--out", default="data/processed/episodes.csv", help="output CSV")
    build.set_defaults(func=cmd_build_dataset)

    tr = sub.add_parser("train", help="train and compare behaviour classifiers; save the best")
    tr.add_argument("--data", default="data/processed/episodes.csv")
    tr.add_argument("--model-dir", default="models")
    tr.add_argument("--report", default="reports/metrics.json")
    tr.add_argument("--seed", type=int, default=42, help="split and model seed")
    tr.add_argument("--fast", action="store_true", help="fewer trees / repeats (for quick checks)")
    tr.set_defaults(func=cmd_train)

    ab = sub.add_parser("ab-test", help="compare fixed and adaptive difficulty on simulated players")
    ab.add_argument("--profiles", default="aggressive,cautious,explorer")
    ab.add_argument("--episodes", type=int, default=40, help="seeds per profile (each played under both conditions)")
    ab.add_argument("--seed", type=int, default=500, help="first seed (500+ were never used for training)")
    ab.add_argument("--out", default="reports/ab_results.csv")
    ab.set_defaults(func=cmd_ab_test)

    dash = sub.add_parser("dashboard", help="build reports/dashboard.html and its figures")
    dash.add_argument("--out", default="reports/dashboard.html")
    dash.add_argument("--figures-dir", default="reports/figures", help="where the PNG figures are written")
    dash.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"neuroquest {__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 2
    return int(args.func(args))
