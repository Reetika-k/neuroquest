# Interface design (current: "bright arcade", 2026-10-14; earlier sci-fi pass below)

The learner found the sci-fi pass too dark and too glowing. Chosen direction, from a two-question check: **bright arcade** (light blue-grey corridors, deep navy walls with crisp edges, bold outlined sprites, no glow). Changes, renderer only:

- Palette: frame (26,32,58); floor (88,102,142) / (91,105,146); walls (33,42,74) with a lighter top/left edge (78,96,140), a dark bottom/right edge and a 2-px outline around the board; trail a light translucent tint; no vignette.
- Sprites, all with a dark 2-px outline and no glow: a cyan robot (rounded body, visor band kept above the centre pixel, eyes that look where you move, antenna, feet, a bob); crimson hunter darts turned toward the player with one eye and a yellow "!" when within 3 cells; amber crystals with a white facet and a gentle bob; an emerald portal with a rotating dashed inner ring and an off-centre highlight.
- Warnings: the screen-edge pulse is gone; proximity (≤ 3 cells) and hits draw a thin crimson frame around the board, so nothing covers the play area. Particles halved. Overlay and menu glows removed.
- Frame cost 0.9 ms. Pixel contracts kept; two drawing mistakes were caught by them during the pass (the visor band and the portal highlight reaching the centre pixel) and fixed in the drawings.

Screens: `docs/screenshots/ui4-playing.png`, `ui4-zoom.png`, `ui4-hit-panel.png`, `ui4-end.png`.

---

# Earlier pass: "premium sci-fi survival" (2026-10-11)

Scope: the Pygame layer only (`ui/theme.py`, `ui/render.py`, `ui/panel.py`, `ui/app.py`). Game rules, A* enemies, the trained classifier, event logging and the adaptive controller are untouched; the full test suite passes and the UI tests still verify base colours at the cell centres of the player, exit, crystal and hunter.

Plan and inspection findings: `UI_REDESIGN_PLAN.md`. Screens: `docs/screenshots/ui2-*.png`.

## 1. Design system (`ui/theme.py`)

| Token | Value | Used for |
|---|---|---|
| BG / BG_DEEP | #0B1020 (11,16,32) / (8,12,26) | window, vignette |
| FLOOR / FLOOR_ALT | #141B30 (20,27,48) / (23,31,54) | walkable tiles (two tones, no grid lines) |
| WALL / WALL_EDGE / WALL_SHADE | #273451 (39,52,81) / #344461 (52,68,97) / (30,40,64) | walls as continuous slate structures; highlight on top/left wall↔floor boundaries, shade on bottom/right |
| SURFACE / SURFACE_2 | (17,22,44) / (24,30,58) | bars, cards |
| LINE | (46,56,96) | 1-px borders |
| CYAN / CYAN_DIM | #36E4FF (54,228,255) / (0,150,178) | player, primary accent, adaptive badge |
| VIOLET | (143,110,255) | secondary accent (enemy-speed bar) |
| CRIMSON | (255,56,92) | hunters, damage, warnings |
| AMBER | (255,184,48) | crystals, score, decisions |
| EMERALD | (24,224,170) | portal, explorer style, victory |
| TEXT / TEXT_MUTED | (232,238,255) / (122,136,176) | primary / secondary text |

Spacing 4 · 8 · 12 · 16 · 24; radii 6 · 10 · 14; type scale display 40 / title 20 / value 22 / body 14 / small 12 / label 11 (caps). Font: Avenir Next → Helvetica Neue → Segoe UI → Arial, falling back to Pygame's bundled font. Helpers: `card` (translucent rounded surface with a 1-px border), `bar`, cached `font`/`text` (the caches are validated and rebuilt if Pygame's font module was re-initialised), and six vector icons (heart, crystal, clock, drone, bolt, chip) drawn with primitives. The glow, vignette and edge-pulse helpers of the earlier sci-fi look were removed in Phase 10 as dead code. No image or audio assets.

## 2. Layout

Top HUD (72 px) · board, centred in the remaining area and reflowing to the left when the panel is open · bottom bar (56 px). Below 600 px tall the bars shrink to 48/40 px and the panel is suppressed, which keeps the 320×240 and 640×480 test windows working. The board is rendered once per maze and cell size (floor tiles with a faint grid, charcoal walls with a cyan-tinted top/left edge and a drop shadow onto the floor, a vignette) and blitted each frame; the trail, light and sprites are drawn on top. Frame cost at 1024×768: 1.6–4.9 ms depending on effects (budget 16.7 ms).

## 3. Components

**Top HUD.** Brand card (chip icon, NEUROQUEST, "the game that learns you"), then stat cards: SCORE (bolt icon; the value pops amber on change), HEALTH (three heart icons, empty ones outlined, shaking briefly on a hit), TIME (clock icon, a bar that is cyan above 30 s, amber above 10 s, then crimson and flashing), COINS (crystal icon, collected / remaining), HUNTERS (drone icon; the count turns crimson when one is within 3 cells).

**Board.** Cyan explorer drone: rounded-diamond hull with a 1-px outline, a core ring and a visor (both off-centre so the centre pixel stays cyan), two thruster dots trailing opposite the facing direction, a 4 % breathing pulse, a controlled glow; a pale blink and a ring while invulnerable. Crimson hunter drones: dart silhouette turned toward the player, a single glowing eye, three rotating ring segments. Pursuit feedback from the real Manhattan distance: within 6 cells the eye brightens and the ring spins faster; within 3 the glow intensifies and a restrained red edge pulse runs around the screen. Amber energy crystals: diamond with a lit facet and two orbiting sparks. Turquoise portal: breathing glow, dark well, six rotating dashes, three slow counter-rotating outer dashes, a lit core. AI vision (T): each hunter's next twelve A* cells as a fading red polyline with dots.

**Bottom bar.** ADAPTIVE / FIXED badge; the enemy behaviour summary ("2 hunters · A* pursuit · move every 4 ticks (2.5 cells/s)") or, for 6 s after a change, the decision in short form ("Difficulty: hunters 2 -> 1 · Eased: heavy damage"); the STYLE badge with the label and confidence; a PAUSED badge when paused; controls on the right in the longest form that fits.

**Intelligence panel (I).** 280 px on the right. AI CORE header with the model name, training date and held-out macro-F1 from the saved metadata. PLAYER STYLE: predicted label, confidence, three probability bars (or "watching your first moves…" / "no model loaded"). ADAPTIVE DIFFICULTY: status badge and four bars positioned within the controller's bounds (hunters 1–5, enemy speed as interval and cells/s, time limit 60–240 s, coin density 0.05–0.25). LAST DECISION: lever change, humanised rule name ("Eased: heavy damage", "Tightened: cruising", "Holding: warm-up"…) and the controller's recorded reason, verbatim and wrapped. RECENT: the previous decisions with UP / DOWN / HOLD markers and timestamps. A note says where decisions are logged. Every value comes from `Prediction`, `DifficultyParams`, `Decision` records or the model meta; nothing is estimated in the UI.

**Screens.** Start menu (title, three cards: objective, controls, what the AI does, with the real model name and score; Enter to start; seed and adaptive status), intro banner for the first 3 s, pause panel with every control, and victory / time-out / defeat panels in one style with a stats grid: score, time, coins collected / total, hits taken, the final style read with confidence, and the number of difficulty changes.

## 4. Effects (UI-only state, never logged)

`fx` in `ui/app.py` holds time, particles, shake, flash, heart shake, the trail with recency, score pop, facing, toggles, counters (hits, difficulty changes) and the decision history shown in the panel. Events map to effects: coin → amber burst and score pop; hit → crimson burst, shake, edge flash, heart shake; escape → green-and-gold burst. Nothing in `fx` reaches `GameState`, the event log or the dataset.

## 5. Verification

- `pytest`: full suite green, including the pixel contracts and a test that draws the panel, menu, pause and end screens from real objects and checks the panel reserves width only when open.
- Headless renders reviewed at each step; defects found and fixed along the way: a footer collision between the style badge and the controls, decision text overflowing the bar, the menu icon overlapping the title, a panel line overflowing, the PAUSED badge colliding with the STYLE badge, a long value spilling out of a stat card, and a test that checked the footer under an overlay veil.
- The real-window feel (input latency, animation smoothness) is checked by playing: `neuroquest play`.

## 6. Not done, on purpose

No sprite sheets or audio (nothing to license), no post-processing shaders (Pygame has none), no animation system beyond time-based pulses and particles.
