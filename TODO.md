# Plotter project: log, state and plan

Branch: `claude/plotter-gcode-improvements-*`. Status key: `[x]` done, `[ ]` open.
Test images: `timik.jpg` (portrait, light face, grey wall) and `image35.jpg` (dense illustration)
from the `add-scribble-code` branch.

Contents:
1. What we did and why (decision log, in order)
2. Current state (what exists, how to run it)
3. Development plan (prioritised)
4. Appendix A: review of the original `radon_renderer`
5. Appendix B: review of `scribble-code/` (scripts and G-code writers)

---

## 1. What we did and why

### 1.1 Direct G-code output (no more hand-editing)

**Problem.** The old workflow was JSON → SVG (Inkscape) → svg2gcode → hand-edited header, and nobody
remembered the edits. The file that plotted correctly used this dialect:

```
G21
G90
G0 Z5        ; pen up
G4 P0.05
G0 X.. Y..
G0 Z0        ; pen down
G4 P0.05
G1 X.. Y.. F1500
```

**Done.**
- [x] `plotter/gcode.py` writes exactly this dialect by default, 3 decimals instead of 14.
- [x] Every machine detail lives in a JSON profile (`python -m plotter profile --dump my_plotter.json`):
      pen up/down commands, dwell, feeds, header/footer, Y flip, offsets, `M0` pause every N mm.
      **Why:** a header change becomes a config change, not a manual edit of every output file.
- [x] Stroke ordering: greedy nearest neighbour with reversal; KD-tree version for 2000+ strokes.
      Sample radon file: pen-up travel 57.8 m → 0.7 m.
- [x] SVG written next to every G-code file as a fallback.
- [x] Plot-time estimate counts pen lifts (`pen_move_s`), which dominate for scribble styles.
- [x] **Smooth joints** (`corner_tol_mm`, default 0.05): FluidNC already blends joints (junction
      deviation), but traced strokes bend ~45° at the median joint, so the planner brakes at almost every
      one. Each corner is now replaced by a short curve within 0.05 mm (invisible with a pen).
      **Why not more:** one extra point per corner already halves each bend; more points mostly add
      file size. Plot runs from SD, so 2-3x more lines is fine.
- [x] Time estimate replays the firmware planner (trapezoids, junction deviation, 16-block look-ahead,
      stop at every pen move); machine values in the profile, defaults from the FluidNC config. F is
      written once (modal) instead of on every line.
      Measured on a real 4400-lift scribble job: old estimate 115 min, planner says 149 min.
      Same job with rounding + profile `dwell_s 0`, pen up `G0 Z3`, F3000 + FluidNC
      `junction_deviation_mm: 0.05`, Z accel 1500: **64 min** (82 without rounding).

### 1.2 String art for pen (priority from the start)

**Problem.** Nail-to-nail lines on a circle, drawn with a pen: unlike thread, ink has a fixed width and
stacks opaquely, so naive greedy string art turns into a black mess.

**Done.**
- [x] Pen-width ink model: each line deposits pen width × length of ink; overlaps combine as
      `1-(1-c)(1-a)`. Dense areas saturate and stop attracting lines. **Why:** physically honest
      simulation is what prevents the black mess.
- [x] Asymmetric error (`overshoot`): ink on areas lighter than the photo costs more → clean highlights.
- [x] `min_skip`, no repeated chord, `nail_spread_mm` (line ends land at slightly different rim points so
      hundreds of ends don't pool into blobs), pen lifts when stuck, `continuous` for a single path.
- [x] `min_lines`: the solver stops when no line improves the picture (after ~270 lines on the test photo
      with a 0.3 mm pen); this forces it to keep going.
- [x] **Settings optimiser** (`plotter/tune.py`, `--auto`, UI "Suggest settings"): coordinate search over
      overshoot, viewing cell, gamma, CLAHE, edge boost, darkness, vignette, min lines. Score =
      SSIM − 0.5·MAE of the simulated plot vs a *fixed* reference at viewing blur. **Why fixed:** otherwise
      changing gamma/CLAHE changes the target and games the score. Result: 0.47 → 0.60 (0.3 mm pen).
      It consistently picks viewing cell 2.2 mm, gamma 1.2–1.5, low highlight protection, no CLAHE.
- [x] **Finding: physics dominates.** A 190 mm circle with a 0.3 mm pen fits only ~450 lines before the light
      face is "done"; a portrait can't form. A 0.1 mm pen gets ~3× the lines (score 0.72).
- [x] **Curved chords** (`chords`): `poincare` = hyperbolic geodesics (meet the rim at 90°, bow towards the
      centre, always inside); `warp` = the solver picks a bend per line from {0, ±0.5, ±1}. **Why:** more paths
      through each area, arcs can follow round shapes. Result: warp 0.719 → 0.730; Poincaré-only is worse
      (the solver piles short arcs into a fringe at the rim).
- [x] **Tapered dashes** (`dashed`, `taper`, `dash_period_mm`): lines are solid in dark areas and break into
      shorter, sparser dashes as the photo lightens, then fade out. The solver deposits ink by dash duty.
      Per-line dash phase (golden ratio) avoids moiré. **Why:** a straight chord meant for the dark hair also
      darkens the light face; dashes put ink only where it is needed. A hard on/off cut (first try) looked
      like a stencil (0.74); the taper fixed it.
      Result: **0.835** (best so far). Cost: ~14k pen lifts at 5 mm period, ~4.4 h plot.

| String art, A4, 0.1 mm pen, test portrait | score | pen lifts | plot |
|---|---|---|---|
| straight | 0.719 | 21 | 2.5 h |
| warp | 0.730 | 21 | 2.5 h |
| warp + hard-cut dashes | 0.740 | 322 | 2.5 h |
| warp + tapered dashes, 5 mm | 0.835 | 14k | 4.4 h |

### 1.3 Flow hatching (crosshatch that is not just parallel)

**Problem.** The original radon renders were families of straight parallel chords; crosshatching looked
mechanical.

**Done.**
- [x] Strokes are streamlines of a structure-tensor direction field: layer 1 wraps around shapes, darker
      tones add crossing layers (across, then diagonals). Evenly spaced streamlines (Jobard–Lefer), spacing
      by tone, min spacing 2.5 × pen so dark areas never turn solid.
- [x] Stubs and seams: smoothed layer map, seeds only well inside a layer, minimum length in spacings,
      random overrun per line. Short-stroke ink in the darkest layer 33% → 8%.
- [x] 2× faster with byte-identical output (list-based tracer, vectorised seeding/stamping, iterative RDP).
- [x] Options: `tone_model=coverage` (ink-matched tones, ~45% more ink), `merge_gap` (−18% pen lifts),
      `outline` levels.

### 1.4 Your scribble code (`scribble-code/`, branch `add-scribble-code`)

**Problem.** Many versions, slow (image35: 201 s, plus 223 s for G-code on a *half-size* image), unclear
lineage, and a copy of the G-code tools with changes.

**Done.**
- [x] Mapped the versions (Appendix B): `c_v25.py` is the latest tracer, `gcode_generator.py` the latest
      writer. Found two real G-code bugs fixed between versions: `G4 P15` (= 15 *seconds* per pen move on
      GRBL) and arcs always written as `G2`.
- [x] `plotter/scribble.py`: port of c_v25 with **identical output** (checked line for line), ~25× faster.
      `numba` was imported everywhere but never used; the cost was rebuilding a 9×9 weight grid 324k times,
      full-image sampling per line, and a quadratic loop check.
- [x] Presets `c_v25`, `app5`, `app7` reproduce those scripts exactly. Key difference found: in app5/7
      `prob_mask = ipix` aliases the darkness map, so light areas stay white (accidental, but good practice).

### 1.5 Scribble visual quality

**Problem.** Crosshatching "creates weird shapes".

**Diagnosis (measured on image35):**
- 56–76% of hatch lines make U-turns: they follow the gradient *vector*, which flips 180° across every ridge.
- 29–46% spiral: a field rotated 30–60° makes streamlines circle around sinks.
- No spacing between hatch lines: up to 491 lines through one pixel → ropes, knots, black fans.
- Seeds only on edges → flat dark areas (no gradient) get almost nothing (37% ink vs 73% target).

**Done** (all opt-in; defaults keep your original look):
- [x] `hatch_mode=clean`: orientation field with consistent heading, tone-based spacing, total-turn limit,
      sub-pixel steps, edge seeds then an even grid. U-turns/spirals 0%, max 3–4 lines per pixel.
- [x] `hatch_fill_px`: flat areas borrow direction from surrounding structure.
- [x] `main_momentum` (0.5) and `smooth_main` (3): calmer contours, no pixel staircase. Momentum 0.8 cuts corners.
- [x] `hatch_mode=ink`: hatching only adds the darkness the contour lines haven't drawn
      (`1 − (1−target)/(1−contours)` at viewing scale), second pass calibrates achieved density.
      Most tone-accurate hatching (portrait error 0.208 → 0.083).
- [x] `hatch_walls` (none | major | all): hatching stops at contours. Visible gain for clean/ink (crisp
      silhouettes); barely any for legacy (its strokes are short and start inside shapes).

### 1.6 Squiggle tone layers

**Goal (yours):** a pseudo-random squiggle that respects gradient flow, tonality and outlines; organic;
may cross itself; "pen held and squiggling" with momentum and random perturbation.

- [x] `squiggle` (experimental): maze curve on a quadtree (spanning-tree outline). Continuous and
      tone-accurate, but grid-like. **Rejected as a direction:** Hilbert-like by construction.
- [x] Research: closest published work is *Tone- and Feature-Aware Circular Scribble Art* (Chiu et al.
      2015): a virtual path that avoids crossing features, followed by circular scribbles whose size and
      speed carry tone.
- [x] `freehand`: simulated hand. Loops around a wandering centre; momentum on the *turn rate* (smooth
      arcs); random drift of loop size, speed, shape and orientation; tone by erasing a "still needed" map
      as it draws; strokes of random length, then lift and restart where darkness is missing; crosses
      itself and contours.
- [x] Bands fixed (tight loops laid saturated ropes side by side): bigger minimum loops, faster travel,
      wider erase footprint, less flow pull, more noise, random choice of the missing-ink direction.
- [x] Walls `major` = drawn contours on edges that survive at a coarse scale (silhouettes, big shapes).
- [x] Black pockets fixed: the look-ahead saw darkness behind walls, so a hand trapped in a small walled
      pocket never left. Too-black pixels 2.3% → 1.1% (all walls 7.2% → 2.5%).
- [x] No double counting: every squiggle/ink mode draws only the darkness the contour lines haven't.
- [x] **Legacy base + tone layer** (`hatch_base=legacy`, `hatch_base_credit`): legacy hatching first, then
      ink/clean/freehand/squiggle add only the darkness contours + legacy haven't drawn. **Why:** legacy looks
      good in gradient-rich areas but adds no tone in flat ones. Walls stay contour-only. On a base, clean
      is routed through ink (a raw residual never reaches clean's layer thresholds). Test photo (circle,
      0.15 mm): tone error legacy 0.163 → legacy+ink 0.042, +squiggle 0.049, +freehand 0.074.

### 1.7 Preview UI

- [x] `python -m plotter ui`: stdlib server + one HTML page. Upload, crop (square for circle styles,
      free for scribble), true-pen-width preview, zoom, travel moves, plot-order animation, target vs
      simulation, plotter profile, G-code/SVG/JSON downloads.
- [x] Every option is a slider/toggle/dropdown, grouped; only the selected tone layer's section shows.
- [x] Help text (`?` + tooltip) for every setting.
- [x] Scribble circle mask; string-art "Suggest settings" button.
- [x] Classic radon removed from the UI: on photos it produced chords with no likeness. Code kept for
      `main.py` and `lines2gcode`.
- [x] Compare page (published artifact "Plotter Style Evolution"): all styles on one portrait with stats.
- [x] **Sheet layout** (`/layout`, linked from the preview): load several G-codes (files, or recent renders
      with thumbnails), drag to move, corner to resize (aspect locked), rotate 90°, centre, fit, duplicate;
      snapping to paper edges/margin/centre and other drawings; off-paper/overlap warnings. "Make G-code"
      combines them in list order through the normal writer (profile from the preview page, bounds check,
      planner estimate). Parser reads our files and svg2gcode-style ones (G2/G3 arcs, M3/M5 servo pens).

---

## 2. Current state

| Style | Command | Best settings so far |
|---|---|---|
| String art | `python -m plotter stringart photo.jpg` | 0.1 mm pen, `--chords warp --dashed`, `--auto` |
| Flow hatch | `python -m plotter hatch photo.jpg` | `--tone-model coverage --merge-gap 4` |
| Scribble | `python -m plotter scribble photo.jpg` | `--preset app5 --momentum 0.5 --smooth 3` + `--hatch-mode ink` or `freehand --density 0.55` |
| UI | `python -m plotter ui` | everything above, with help |
| Old renders | `python -m plotter lines2gcode render.json` | |

Tests: `uv run --with pytest pytest -q tests` (19 tests).

---

## 3. Development plan (prioritised)

### P1: needs the real plotter
- [ ] **Plot one of each style on paper** and compare with the previews: line weight, pen bleed at
      crossings, whether dashes read as tone.
- [ ] Verify profile: Y direction (`flip_y`), origin, Z values, dwell. Record the working profile in the repo.
- [ ] Time a real plot and compare with the estimate (it now models the planner; check it's honest).
- [ ] Try on the machine, one at a time: `dwell_s 0` (stepper lift moves in order, no settle needed?),
      pen up `G0 Z3`, FluidNC `junction_deviation_mm: 0.05`, Z `acceleration_mm_per_sec2` 300 → 1000+,
      then raise `draw_feed` until lines get wobbly.
- [ ] Pen wear: suggested `pause_every_mm` per pen type (a 0.1 mm fineliner won't do 200 m).
- [x] Bounds check: G-code is refused (nothing written, clear error in CLI/UI) when any X/Y target,
      including header/footer moves, falls outside 0..`machine_x_mm` × 0..`machine_y_mm` (320 × 380 from
      FluidNC `max_travel_mm`) after flip/offsets, or a stroke leaves the paper. A3 doesn't fit; Legal
      only in portrait.
- [ ] Measure real `max_travel_mm` (config says "measure"), then set FluidNC `soft_limits: true` as a
      second guard. Check that the work origin is the homed corner (the bounds check assumes it).

### P2: string art likeness and plot time
- [x] Dash pen lifts, investigated. **Finding:** joining or reordering dashes can't cut lifts (one per gap
      by definition; travel is already ~0). On a real 13k-lift dashed job, pen moves were 133 of 224 min.
      What helps: lift cost. Profile `dwell_s 0` + pen up Z3 + FluidNC Z accel 1500 → 131 min.
      Added short hop (`hop_up`, e.g. `G0 Z2.5` for moves < `hop_max_mm`): small extra gain (Z time ~ √distance).
      Tried a tone-preserving adaptive dash period (longer where duty is near 0 or 1): −12% dashes, but a
      plain 12 mm period gives the same −13% at the same tone change, so not added. Fewer lifts = longer
      `dash_period_mm` (the slider already exists).
- [x] Poincaré fringe, investigated. **Finding:** short arcs are not the cause. The rim isn't over-inked
      (0.347 vs target 0.336), and a minimum arc span for bent chords only lowers the score
      (0.669 → 0.664 at 12% of the circle, 0.600 at 25%), so it was not added. The fringe is a
      sunburst: a geodesic always meets the rim at 90°, so every line end points straight at the centre.
      Gentler fixed bends score better and look calmer (bend 1.0: 0.669, 0.7: 0.683, 0.5: 0.688,
      0.3: 0.691), but mixed bends (`warp`, 0.717) beat all of them. Use `warp`. `poincare` stays as
      a look, not for likeness.
- [ ] Refinement after greedy: try removing/swapping lines (local search / annealing), typically a large gain.
- [ ] Importance mask (face/eyes weighted higher): UI brush or face detection; also for hatching/scribble.
- [ ] Include `chords`/`dashed` in the optimiser search; optimiser progress + cancel in the UI.
- [ ] Two pens / grey levels (e.g. 0.1 mm grey pass + 0.3 mm black pass).

### P3: scribble / freehand
- [ ] Freehand speed: pure-Python step loop (~10–20 s at 500–600 px). Numba or a vectorised multi-hand step.
- [ ] Freehand: occasional dark knots where the photo is very dark; try a loop-size floor tied to pen width.
- [ ] Quadrant cleanup vectorised (app presets spend most time there).
- [ ] Taper/stagger hatch ends at tone boundaries for clean/ink (like flow hatch `overrun`).
- [ ] Optional fast arc fitting (G2/G3) to shrink G-code, with correct > 180° handling.
- [ ] 2-opt refinement of stroke order on top of the KD greedy.

### P4: flow hatch
- [ ] Short strokes where flow converges: merge end-to-end neighbours.
- [ ] Other non-parallel fills: tone-modulated spiral/concentric, stipple → TSP single line, wave lines.
- [ ] Mix styles: string-art circle with hatched features on top.

### P5: UI and project hygiene
- [ ] Progress bar / cancel for long renders; load and preview an existing `.gcode`.
- [ ] Layout: keep uploaded G-codes across server restarts (now only renders are restored); free rotation;
      optimise stroke order across drawings.
- [ ] A/B compare inside the UI (the compare page does this for fixed renders).
- [ ] Move the compare page and its renders into `docs/` if it should live in the repo.
- [ ] `pyproject.toml`: description, declare `scipy`/`matplotlib`, relax the Python 3.14 pin.
- [ ] Move generated artefacts (`output.json`, `final_mode4b.json`, PNGs) to `examples/` or ignore them;
      `.claude/settings.local.json` shouldn't be committed.
- [ ] Decide whether `radon_renderer` stays (see Appendix A) or is archived.

---

## Appendix A: review of the original `radon_renderer`

Not fixed (the style is no longer offered in the UI). Correctness issues found:
- `compute_target_image`: `self.image` is already darkness, then the target uses `1 - intensity_norm` again,
  so the intensity term asks for lines in **light** areas.
- `_render_with_residual` (hierarchical) swaps in the residual, but `generate_chord_at_angle` reads only the
  precomputed sinogram, so the residual has no effect.
- `spacing_metric` is constructed and reported but never used; `MSEOptimizedSpacingMetric` is a stub.
- Modes 2/3 undo a trial with `self.lines[:-lines_to_add]`, but fewer lines may have been added → removes
  accepted lines.
- Zero-length lines emitted; blind centre crop loses the subject in padded images; render adds a fixed 0.1
  per pixel regardless of pen width; sinogram position mapping assumes 1 px = 1 mm.
- Hygiene: debug print `jsorn arg` in `cli.py`; `PAPER_SIZES` landscape only.

## Appendix B: review of `scribble-code/`

| File | What it is |
|---|---|
| `app.py`, `app2.py`, `app3py` (= `app4.py`), `app5–7.py` | notebook-style scripts, each a copy-paste evolution; app5+ add G-code |
| `image2sketch.py` | first `SketchConverter` class |
| `c_v14.py` → `c_v22.py` → **`c_v25.py`** | SketchConverter versions; c_v25 is the latest |
| `gcode_old.py` | first G-code writer; broken top-level call (`lines2` undefined) |
| `gcode_utils.py` | Eulerian merge, greedy+2-opt ordering, line/arc fitting, several writers |
| `path_optimizer.py` | `PathOptimizer` / `PathOptimizerV2`: KD-tree greedy + windowed 2-opt |
| `gcode_generator.py` | **latest** writer: fit-to-page, arc sanitising, G2/G3 by cross product |
| `plotter.py`, `plotter_v2.py`, `rpi-plotter_app.py` | separate: 2-servo arm plotter on RP2040 (MicroPython) |

G-code findings:
- Fixed in your later versions: `G4 P15` → `P0.015` (GRBL P is seconds); G2/G3 chosen by cross product
  (older writers always wrote G2; image4.jpg.gcode has 153k G2 and 0 G3).
- Still open in `GCodeGenerator`: arcs > 180° would be drawn as their complement; no Y flip (image rows → X).
- `generate_final_gcode` maps Y negative and prints 14 decimals (unused).
- app7 uses `merge_lines` before importing it (notebook-only); app5 imports `finalize_and_generate_gcode`,
  which no longer exists.
