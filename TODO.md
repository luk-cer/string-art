# TODO / Review notes

Status key: `[x]` done on `claude/plotter-gcode-improvements-*`, `[ ]` open.

## 0. Direct G-code output (no more hand-editing)

The old workflow was: JSON → SVG (Inkscape) → svg2gcode → hand-edited header.
The G-code that plotted correctly looked like:

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

- [x] `plotter/gcode.py` writes exactly this dialect by default (3-decimal coords instead of `X50.77875100000003`)
- [x] Every knob in a JSON profile (`python -m plotter profile --dump my_plotter.json`):
      pen up/down commands, dwell, feed, header/footer, Y flip, offsets, optional `M0` pause every N mm (pen check)
- [x] `python -m plotter lines2gcode output.json` converts existing radon renders **without changing the look**,
      with nearest-neighbour ordering + segment reversal (sample file: pen-up travel 57.8 m → 0.7 m)
- [x] SVG export alongside (`.svg`, mm units) so Inkscape / svg2gcode still works as a fallback
- [ ] **Verify on the real plotter**: Y direction (`flip_y`), origin, Z values. If the old manual edit was something
      else (e.g. a `G92 X0 Y0` homing line, `M3/M5` servo pen), put it in the profile `header`/`pen_up`/`pen_down`
- [ ] Bounds check: refuse to write moves outside the paper / machine area

## 1. String art portrait (priority)

`python -m plotter stringart photo.jpg -o out/portrait` → `.gcode`, `.svg`, `_preview.png`, `_compare.png`, `.json`

Done:
- [x] Nails on a circle, greedy nail-to-nail line selection
- [x] **Pen-width-aware ink model** (the anti-"black mess" part): each line deposits `pen_width × length` of ink,
      overlaps combine as opaque ink `1-(1-c)(1-a)`, so dense areas saturate and stop attracting lines
- [x] Asymmetric error (`--overshoot`): ink over areas lighter than the target costs more → clean highlights
- [x] `--min-skip` (no short chords hugging the rim), no repeated chord
- [x] `--nail-spread` mm: each visit to a nail lands at a slightly different point on the rim, so hundreds of
      line ends don't pool into ink blobs at the nails
- [x] Pen lifts when stuck (best line anywhere); `--continuous` for a single unbroken path
- [x] Auto-crop trims white borders (e.g. `timik.png` is a small photo on a big white canvas), `--crop X Y W H` manual
- [x] Ink length + plot-time estimate printed

Open (in rough order of value):
- [ ] **Quality is still limited**: on the test photo (3/4 profile, busy café background) ~1000 lines give
      hair / face / jacket masses but not eyes/mouth. Ideas:
  - [ ] Importance mask: weight the face region higher (manual ellipse in the UI, or face detection)
  - [ ] Multi-scale error: compare blurred sim vs blurred target (viewer distance) + a fine-scale edge term
  - [ ] Refinement pass after greedy: try removing / swapping each line (or simulated annealing) – usually a large win
  - [ ] Allow *partial* chords (pen down between two points on the chord) for feature lines – breaks pure "string
        look" but a plotter can do it; make it an option
  - [ ] Several pens / grey levels (e.g. 0.1 grey + 0.3 black passes)
- [ ] Choose start nail / stroke order to minimise pen lifts
- [ ] Nail dots / tick marks option (decorative) and `--frame` circle (exists, untested on paper)
- [ ] Pen-wear: optional `pause_every_mm` defaults per pen type (a fineliner won't do 300 m)
- [ ] Recommended source photo: front-facing, even light, plain background, face filling the circle

## 2. Better crosshatch (keep the current look as one style)

Keep `radon_renderer` modes as-is (`lines2gcode` plots them unchanged). New style next to them:
`python -m plotter hatch photo.jpg -o out/hatch` — **curved, not just parallel**.

- [x] **Flow hatching**: strokes are streamlines of a direction field from the image (structure tensor),
      so layer 1 wraps around shapes (cheek, jaw, hair) instead of running as straight parallel chords
- [x] **Tonal layering**: layer 2 crosses layer 1 (follows the gradient), layers 3/4 diagonal, each only in
      darker tones (`--thresholds`); spacing varies continuously with darkness, min spacing = 2.5 x pen width
- [x] Evenly-spaced streamline placement (Jobard–Lefer): seeds beside finished lines, lines stop when too close
- [x] `--flow fixed` = classic straight crosshatch with the same tone logic, for comparison
- [x] Layer-by-layer drawing, path-optimised (vectorised nearest neighbour), RDP-simplified strokes
- [x] Circle (matches string art) or `--shape rect`
- [x] Stubs: smoothed layer map (`--tone-smooth`), seeds only well inside a layer (`--seed-margin`),
      min length relative to spacing (`--min-len-spacings`). Darkest layer: short-stroke ink 33% → 8%
- [x] Seams: each line ends at its own randomly lowered threshold (`--overrun`), so ends are staggered
- [x] Speed: ~2x faster with byte-identical output (vectorised seeding/stamping, list-based tracer, iterative RDP)
- [x] `--tone-model coverage` (physically matched tones), `--merge-gap` (fewer pen lifts), `--outline LEVEL`
- [x] Time estimate includes pen lifts (`pen_move_s` in the profile)
- [ ] Faster still: numba/cython for the tracer (would add a dependency)
- [ ] Remaining short strokes in converging flow (lines squeezed out) – try merging end-to-end neighbours
- [ ] Optional outline pass (edge contours, simplified) on top of the hatching
- [ ] Other non-parallel fills worth trying: concentric/spiral modulated by tone, stippling-to-TSP single line,
      wave lines with amplitude ∝ darkness
- [ ] Mix: string-art circle with flow-hatched features (eyes/mouth) on top

## 3. Preview / UI

`python -m plotter ui` (stdlib server + one HTML page, no extra deps)

- [x] Upload photo, interactive square/circle crop (auto-zooms to content, auto-crop default)
- [x] All three styles (string art, flow hatch, classic radon) with sliders; settings remembered per style
- [x] Preview at true pen width, zoom/pan, travel moves overlay, animated plot order
- [x] Target vs simulation view
- [x] Editable plotter profile → downloads `.gcode` / `.svg` / `.json`
- [ ] Progress bar / cancel for long string-art renders (currently a timer)
- [ ] Load an existing `.gcode` file and preview it (pen-down vs travel, bounds vs paper)
- [ ] Side-by-side compare of two renders (A/B of settings)
- [ ] Importance brush: paint the face/eyes to weight them higher (string art + hatch)

## 4. Review of existing `radon_renderer` code (bugs / cleanups)

Correctness:
- [ ] `compute_target_image`: `self.image` is already *darkness* (`1 - gray`), then the target uses
      `1 - intensity_norm` again → the intensity term asks for lines in **light** areas. The comment says the
      opposite. Fixing it changes the look, so do it as a flag (`--fix-intensity`) and compare.
- [ ] `_render_with_residual` (hierarchical mode) swaps `self.image/edges` for the residual, but
      `generate_chord_at_angle` reads only the precomputed sinogram, so the residual has **no effect** on placement.
      Fix: recompute `radon(residual)` per angle (cheap at 190 px).
- [ ] `spacing_metric` is constructed and reported in JSON but never used (`--spacing-metric` is a no-op);
      `MSEOptimizedSpacingMetric` is a stub.
- [ ] Modes 2/3 undo a trial with `self.lines[:-lines_to_add]`, but `generate_chord_at_angle` can return fewer
      lines than requested → removes accepted lines. Use `del self.lines[n_before:]`.
- [ ] Degenerate zero-length lines are emitted (first line of `final_mode4b.json`); filter `len < pen`.
- [ ] `load_and_preprocess` centre-crops blindly; with padded images (`timik.png`) the subject is lost.
- [ ] `render_lines_to_image` adds a fixed 0.1 per pixel regardless of pen width / pixel size → error metric
      doesn't model what the paper will look like (see the ink model in `plotter/stringart.py`).
- [ ] Sinogram index → position mapping assumes 1 px = 1 mm (true only because `target_size = int(diameter)`).

Hygiene:
- [ ] `pyproject.toml`: description placeholder; `scipy` used but not declared; `matplotlib` used by
      `compare_modes.py` but not declared; Python 3.14 pin is stricter than needed
- [ ] Generated artefacts committed (`output.json`, `final_mode4b.json`, PNGs) – move to `examples/` or ignore
- [ ] `.claude/settings.local.json` committed (local settings)
- [ ] Debug print `jsorn arg` in `cli.py`
- [ ] No tests – add small ones for geometry (chord endpoints on circle), G-code writer, ink model
- [ ] `PAPER_SIZES` landscape only; add orientation flag to radon CLI (string art has `--portrait`)

## 5. scribble-code (branch `add-scribble-code`) – review

What is what (single commit, so lineage is from content diffs):

| File | What it is |
|---|---|
| `app.py`, `app2.py`, `app3py`(= `app4.py`), `app5-7.py` | early notebook-style scripts, each a copy-paste evolution; app5+ add G-code |
| `image2sketch.py` | first `SketchConverter` class (zuzka.jpg) |
| `c_v14.py` → `c_v22.py` → **`c_v25.py`** | successive SketchConverter versions; **c_v25 is the latest** (adaptive cleanup from intensity+variance, optional NL-means denoise) |
| `gcode_old.py` | first G-code writer (DPI scale, pen lift per line); has a broken top-level call (`lines2` undefined) |
| `gcode_utils.py` | merge (Eulerian), greedy+2-opt ordering, line/arc fitting (`GeometryOptimizer`), several G-code writers |
| `path_optimizer.py` | `PathOptimizer` / `PathOptimizerV2` (same code + comments): KD-tree greedy + windowed 2-opt |
| `gcode_generator.py` | **latest** writer `GCodeGenerator`: fit-to-page, arc sanitising, G2/G3 by cross product |
| `plotter.py`, `plotter_v2.py`, `rpi-plotter_app.py` | unrelated: 2-servo arm plotter kinematics on RP2040 (MicroPython) |

G-code changes found across the versions:
- [x] Dwell fixed: early writers emit `G4 P15` (GRBL: P = **seconds** → 15 s per pen move; image35.gcode would
      pause ~125 h). `GCodeGenerator` uses `P{ms/1000}` → `P0.015`. Our writer: `dwell_s` (default 0.05, like the
      hand-edited file).
- [x] Arc direction: `to_gcode`, `to_gcode_a4`, `to_gcode_continuous` always emit `G2` (half the arcs drawn the
      wrong way – image4.jpg.gcode has 153k G2 and 0 G3). `GCodeGenerator` picks G2/G3 by cross product.
- [ ] `GCodeGenerator` still assumes every arc is < 180° (center side ⇒ direction); a fitted arc > 180° would be
      drawn as its complement. Our port avoids arcs (RDP polylines) – optional arc fitting could be added back.
- [ ] `GCodeGenerator`: no Y flip; X = image row, so on a bottom-left-origin plotter the image is rotated 90°
      (our `--rotate auto` does this explicitly). Fit-to-page uses the drawing's bounding box, not the image.
- [ ] `generate_final_gcode` rotation maps Y negative (`ny = -nx`) and prints 14 decimals – unused now.

Performance (image35 736x1093):
- [x] Tracing: 201 s → 8 s, identical output (`plotter/scribble.py`; see its docstring for what changed)
- [x] G-code: `GeometryOptimizer` arc DP took 223 s for the *half-size* image; replaced by vectorised RDP +
      end joining + KD-tree ordering (~3 s full size). Fewer pen lifts (2748 vs 4159 on half size); G-code file
      is larger (line segments instead of arcs).
- [x] `numba` was imported in every c_v* file but never used
- [ ] Optional fast arc fitting (G2/G3) to shrink files, with correct >180° handling
- [ ] 2-opt refinement of stroke order (PathOptimizerV2 style) on top of KD greedy
- [x] app5 / app7 reproduced as presets (identical lines to the original scripts on half-size image35;
      44 s -> 4.8 s and 53 s -> 5.4 s). Differences vs c_v25: white cut (prob_mask aliases ipix in the scripts),
      quadrant cleanup up to 20 px, stop at 0.1% gradient left, 10000-step lines, hatch angles 225/245/205/265,
      hatch look-ahead 11x11 (trace() default d=5), hatch from longest (app5) / merged (app7) lines
- [ ] app7 calls merge_lines before importing it (notebook-only); app5 imports finalize_and_generate_gcode,
      which no longer exists in gcode_utils
- [ ] Vectorise quadrant cleanup (app presets spend most time there)

## 6. Scribble – visual quality (what made hatching look odd)

Measured on image35 (half size), legacy hatching:
- 56–76% of hatch lines make a U-turn: they follow the gradient *vector*, which flips 180° across every
  dark line/ridge -> hooks, V shapes, back-tracking
- 29–46% spiral (>270° total turn): a field rotated by 30–60° makes streamlines circle around sinks
- no spacing between hatch lines: up to 491 hatch lines through one pixel -> ropes, knots, black fans
- seeds only on main lines -> hatching follows edge density, not tone; flat dark areas (no gradient) get
  almost nothing (photo test: 37% ink density vs 73% target darkness; main lines 0.6%)

Done (all options, defaults keep the original look):
- [x] `hatch_mode="clean"`: orientation field with consistent sign, per-direction spacing by tone,
      total-turn limit, sub-pixel steps, edge seeds first then an even grid (fills flat darks: 61%)
- [x] `hatch_fill_px`: flat areas borrow direction from a larger scale (lines wrap around shapes
      instead of running in fixed parallel rows)
- [x] `main_momentum` (0.5 recommended): no U-turns/spirals in main lines, calmer contours; 0.8 starts
      cutting corners. `smooth_main` (3) removes the pixel staircase.
- [ ] Coverage-based tone for clean hatching that accounts for main-line ink already present
- [ ] Taper/stagger hatch ends at tone boundaries (as in flow hatch overrun)
