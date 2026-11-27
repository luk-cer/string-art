# Stepper pen lift for ExtruH

`stepper_pen_lift.scad`: a compact, spring-loaded lever lift using a 17HS4023 pancake stepper.
It replaces the ExtruH servo lift (parts 15 and 16 plus the 80 mm 2020 piece) and bolts to the
same 4 tapped M3 holes on `7_pencarriage`.

The springs push the pen down and the motor arm only lifts it. Pen pressure therefore comes only
from the springs, whatever the Z position.

Size: 32 × 75 × 81 mm body (the geodave NEMA17 mount is 108 × 68 × 28 mm).

## Print (PLA, 3+ perimeters, no supports)
STLs are not in the repo. Generate them from the .scad file (command below).

| file | orientation |
|---|---|
| `body.stl` | lying on its outer side wall (the wall next to the motor) |
| `slide.stl` | upside down (ledge flush with the bed) |
| `arm.stl` | flat |

Re-export after changing parameters with
`openscad -D 'part="body"' -o body.stl stepper_pen_lift.scad` (also `slide` and `arm`).
Main parameters: `pen_d` (pen barrel diameter), `motor_len` (measure your motor), `arm_r`.

## Hardware
- 17HS4023 pancake stepper + cable to the DLC32 **Z** socket (A4988/DRV8825/TMC2209 stepstick)
- 2× 3 mm steel rod, 60 mm. Use silver steel, drill blanks, or DVD-drive rods.
- 2× small springs over the rods, ~3.5 mm ID, ~20 mm free length. Ballpoint pen springs work.
- 4× M3x10 (body to carriage), 4× M3x8 (motor), 2× M3x8 (pen clamp), 1× M3 grub screw (arm)
- 1× M3x16 + nut (switch pusher), 1× KW10-type microswitch (20 × 10 × 6.4 mm) + 2× M2x8

## Assembly
1. Ream the rod holes to 3 mm. Tap or self-tap the 2.5 mm holes for M3 (ExtruH convention).
2. Bolt the body to the pen carriage with 4× M3x10 **before** fitting the motor. The motor drops
   in from the top. The notch in the front plate gives hex-key access to the lowest screw later.
3. Fit the motor with its cable connector facing up, using 4× M3x8 through the front plate.
4. Push one rod up through the lower block, through the slide, and through a spring, into the
   upper block. Repeat for the other rod.
5. Fit the arm on the shaft so the tip sits under the slide's ledge. Tighten the grub screw.
6. Screw the switch to the switch plate, lever down. Thread the M3x16 pusher into the slide tab
   so its head presses the lever just before the slide reaches the top. Lock it with the nut.

## Driver current
Set the Z driver to ~0.4 A, which is plenty and keeps the motor cool. The 17HS4023 is rated 0.7 A.
- A4988 (0.1 Ω sense): Vref ≈ 0.32 V
- TMC2209 standalone: Vref ≈ 0.55 V

## Calibrating pen height
1. `$H`: Z homes up to the switch and sets Z = 8. X and Y home after it.
2. `G0 Z0`: arm moves down.
3. Loosen the pen clamp, let the pen rest on the paper, and lift it so there is ~1–2 mm
   between the slide's ledge and the arm tip. Tighten the clamp.
4. `G0 Z5` should now lift the pen ~3–4 mm. For more lift, use `G0 Z7` as the plotter's pen-up
   command.

Pen force is set by the springs. Shorten or stretch them, or add washers.

Credits: ExtruH by pittance (CC BY-SA 4.0); lift concept after geodave's V-slot pen mount (CC BY 4.0).
This derivative is CC BY-SA 4.0.
