"""Phase 3 smoke tests: the package imports, the CLI runs, Pygame works headlessly."""

from neuroquest import __version__
from neuroquest.cli import main


def test_package_has_a_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_version_flag(capsys) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"neuroquest {__version__}"


def test_cli_without_command_prints_help(capsys) -> None:
    assert main([]) == 2
    assert "usage: neuroquest" in capsys.readouterr().out


def test_check_command_runs_headless(capsys) -> None:
    assert main(["check"]) == 0
    assert "pygame headless init: ok" in capsys.readouterr().out


def test_pygame_surface_without_a_window() -> None:
    import pygame

    pygame.init()
    try:
        surface = pygame.Surface((8, 8))
        surface.fill((1, 2, 3))
        assert surface.get_at((0, 0))[:3] == (1, 2, 3)
    finally:
        pygame.quit()
