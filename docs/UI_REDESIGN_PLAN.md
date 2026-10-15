# UI redesign plan — "premium sci-fi survival" (approved 2026-10-11)

Brief: the learner's UI/UX redesign prompt (recorded in `DEVELOPMENT_LOG.md`, entry 8c). Scope: the Pygame layer only. Everything in `core`, `ai`, `data`, `ml`, `adapt` and `sim` is preserved untouched.

## Inspection findings

- Rendering is separated from logic: `ui/render.py` paints a `GameState` plus a UI-only `fx` dict; `ui/app.py` runs a fixed 10-tick/s loop with drawing at ≤ 60 FPS. Nothing in the UI mutates state.
- Coordinates: `(row, col)` cells → pixels via `layout()`; default window 1024×768 → 41 px cells; tests also use 320×240 and 640×480.
- Frame cost before the redesign: 1.8 ms with AI vision on.
- Truthful data available: `Prediction(label, proba, confidence)` every 2 s after 3 s; `Decision(t, rule, direction, lever, before, after, inputs, reason)` every 3 s including holds; `DifficultyParams`; game stats; the saved model's metadata (name, training date, test scores). Nothing else exists, so nothing else is displayed.
- Test contracts to keep: base colour at the centre of the player / exit / coin / enemy cells, frame colour beside the board, footer colour at the bottom-left pixel, and the `draw()` / `run()` signatures.

## Files

New: `ui/theme.py` (tokens + helpers), `ui/panel.py` (intelligence panel). Modified: `ui/render.py`, `ui/app.py`, `cli.py` (`--no-menu`), `tests/test_ui.py`.

## Steps

3. Design system: palette, type scale, spacing, radii, `card`/`bar`/`glow`/`text`/icons.
4. Top HUD as components; bottom bar with status, enemy summary, pause badge, controls.
5. Maze (pre-rendered tiles, grid, shadows, vignette, player light), player drone, hunter drones with pursuit and proximity feedback, crystals, portal.
6. Intelligence panel (key I) from real predictions, parameters, decisions and model metadata.
7. Start, pause, victory, defeat screens; feedback on pickup, damage, pause, escape.
8. Full suite, headless play-through with screenshots of every state, frame-time check (target < 6 ms).

Each step ends with `pytest` and a headless render reviewed before the next.
