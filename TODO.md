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
- [ ] Remaining short strokes in converging flow (lines squeezed out) – try merging end-to-end neighbours
- [ ] Optional outline pass (edge contours, simplified) on top of the hatching
- [ ] Other non-parallel fills worth trying: concentric/spiral modulated by tone, stippling-to-TSP single line,
      wave lines with amplitude ∝ darkness
- [ ] Mix: string-art circle with flow-hatched features (eyes/mouth) on top

## 3. Preview / UI

- [ ] Local web UI (stdlib `http.server` or a tiny FastAPI app, no build step):
      upload photo → crop circle interactively → sliders (nails, lines, pen, overshoot, gamma, …)
      → live preview at true pen width → download `.gcode` / `.svg`
- [ ] Animated "plot order" preview (draw lines in G-code order) to sanity-check pen lifts
- [ ] G-code viewer: load any `.gcode`, show pen-down vs travel, bounds vs paper

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
