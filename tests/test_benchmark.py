"""Phase 5: the benchmark module on a few seeds (the full run is a CLI command)."""

from neuroquest.ai.benchmark import chase_benchmark, format_summary, search_benchmark, summarize_chase, summarize_search, write_csv


def test_astar_enemy_catches_in_exactly_path_length_ticks() -> None:
    rows = chase_benchmark(episodes=3, max_ticks=400)
    astar = [r for r in rows if r["kind"] == "astar"]
    assert len(astar) == 3 and all(r["caught"] for r in astar)
    assert all(r["ticks_to_catch"] == r["path_length"] for r in astar)
    random_rows = [r for r in rows if r["kind"] == "random"]
    assert len(random_rows) == 3 and all(isinstance(r["caught"], bool) for r in random_rows)


def test_search_benchmark_agrees_on_path_length() -> None:
    rows = search_benchmark(mazes=5)
    assert len(rows) == 5 and all(r["bfs_length"] == r["astar_length"] for r in rows)
    summary = summarize_search(rows)
    assert summary["equal_length"] == 5


def test_summaries_and_csv(tmp_path) -> None:
    chase = summarize_chase(chase_benchmark(episodes=2, max_ticks=200))
    search = summarize_search(search_benchmark(mazes=2))
    text = format_summary(chase, search)
    assert "Chase:" in text and "Search:" in text
    out = write_csv(search_benchmark(mazes=2), tmp_path / "x" / "search.csv")
    assert out.exists() and out.read_text().splitlines()[0].startswith("seed,start,goal,bfs_length")
