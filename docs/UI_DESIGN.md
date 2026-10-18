# NeuroQuest — Interface design ("cartoon world")

Status: current look since 2026-10-17 (development-log entry 13) · Earlier directions: sci-fi (entry 8c), bright arcade (entry 8e); both are in the log with the learner's reactions.

## 1. Direction

A friendly garden maze in the family of Equation Catcher's look, which the learner had accepted: pastel sky, rounded hedges on a sandy path, characters with faces, white rounded cards, soft shadows. No image or audio assets: everything is drawn with Pygame primitives, so the game stays a single Python package. The learner chose this direction from two rendered mockups (`docs/screenshots/mockup-cartoon-whole.png`, `mockup-cartoon-camera.png`) before any game code changed. The whole maze stays visible, as in the first mockup; a zoomed camera that follows the player is a possible later step.

## 2. Design system (`ui/theme.py`)

| Token | Value | Used for |
|---|---|---|
| BG | (196, 228, 255) | sky, the window frame |
| SURFACE / SURFACE_2 | white / (240, 244, 252) | cards / tracks and secondary cards |
| LINE | (222, 229, 242) | card borders |
| TEXT / TEXT_MUTED | (52, 60, 84) / (134, 143, 170) | ink / labels |
| CYAN | (74, 170, 255) | accent blue: badges, "cautious", time ring |
| CRIMSON | (255, 107, 138) | hunters, "aggressive", hit warning |
| AMBER | (255, 190, 70) | the hero, score accents, brand mark |
| EMERALD | (96, 196, 110) | "explorer", time ring when plenty is left |
| VIOLET | (160, 140, 255) | enemy-speed bar |

Type: Arial Rounded (macOS) → Avenir Next → Helvetica Neue → Segoe UI → Arial, bold for values and labels; scale display 40 / title 20 / value 22 / body 14 / small 12 / label 11 (caps). The rounded font has no middle-dot or arrow glyph, so one-line texts use `/` as the separator (`theme.SEP`) and `->` for changes. Spacing 4 · 8 · 12 · 16 · 24; radii 8 · 14 · 20. Helpers: `card` (optional `lift` drop shadow), `shadow`, `pill` (coloured capsule with centred text), `bar`, `ring` (the time dial), cached `font`/`text`, and the icons (heart, crystal, clock, hunter head, bolt, smiling brand mark, dot).

## 3. Board (`ui/render.py`)

- **Frame:** a cream rounded card with a soft shadow around the maze, on the sky; two pale hills sit low behind it.
- **Floor:** sand with seeded pebble dots; visited cells show faint footprints that fade with age (from the app's trail).
- **Hedges:** each wall cell is a rounded block (dark base, lit top face, a light strip along exposed top edges, leaf dots), and the junctions between neighbouring wall cells are filled, so walls read as continuous hedges rather than separate bushes. The board is rendered once per maze and cell size and blitted each frame.
- **Gems:** two-tone blue diamonds with an off-centre sparkle, bobbing over a small ground shadow.
- **Gate:** a wooden arch with a cream doorway and a small green flag that waves; the exit.
- **Hunters:** pink round blobs with spiky hair, an angry face whose pupils look at the player, a ground shadow; a white "!" bubble appears within three cells, driven by the real distance.
- **Hero:** a yellow round explorer with feet, a blue-tipped antenna and a happy face whose eyes look where you move; bobs while walking, blinks pale while invulnerable.
- **AI vision (T):** the hunters' planned A* paths as fading dotted trails.
- **Warning:** a pink frame around the board while a hunter is within three cells or right after a hit; nothing is drawn over the play area.

## 4. HUD, footer, panel and screens

- **Top HUD:** floating white cards with shadows: the brand (smiling mark, NEUROQUEST, subtitle), SCORE, HEALTH (three hearts that shake on a hit), TIME (value plus a ring that empties and turns yellow then pink), GEMS (collected / left), HUNTERS (pink when one is close). Compact variant below 600 px tall.
- **Bottom bar:** a cream ground band with a hedge-green top line: the ADAPTIVE/FIXED pill, the current difficulty or the latest decision ("Difficulty: enemy move interval 5 -> 6 / Eased: heavy damage"), the style pill in a pastel of the class colour, a PAUSED pill, and the controls hint sized to the space left.
- **Intelligence panel (I):** a lifted white card: AI CORE header with model metadata, style probabilities as bars, the four difficulty knobs against their bounds, the last decision with its recorded reason, recent decisions, the log note. Content unchanged from Phase 8c; only the skin changed.
- **Screens:** start menu, pause and end screens are white cards over a light veil; the end screen shows score, time, gems, hits, the style read and the number of difficulty changes.

## 5. Contracts kept

The pixel tests pin the base colour at each cell centre: the hero's and hunters' faces are built with the eyes above and the mouth below the centre by construction, the gem's sparkle and the hero's highlight are off-centre, and the gate's centre pixel is wood. The left margin and the top-left of the window stay plain sky, and the bottom band is `FOOTER_BG` across the full width, which is what `tests/test_ui.py` samples. Frame cost at 1024×768 is measured in entry 13 of the development log.

Screenshots of this look: `docs/screenshots/ui5-*.png`; animations: `neuroquest-play.gif`, `neuroquest-agent.gif`.
