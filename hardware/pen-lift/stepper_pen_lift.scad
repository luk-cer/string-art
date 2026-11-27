/*  Compact spring-loaded stepper pen lift for the ExtruH pen plotter
    ------------------------------------------------------------------
    - Bolts to ExtruH part 7_pencarriage using the same 4 tapped M3 holes
      as the original 15_penliftbase2 (ExtruH by pittance, CC BY-SA 4.0).
    - Motor: 17HS4023 NEMA17 pancake (42.3 x 42.3 x 23 mm), shaft along +X.
    - Mechanism idea from geodave's "Z-axis servo/stepper pen plotter mount
      for V-slot" (CC BY 4.0): springs push the pen DOWN, the motor only LIFTS
      it, so pen pressure = spring force, independent of Z accuracy.

    Assembly coordinates
      origin = outer face of the pen carriage, at the 2020 pen-arm centreline
      +X away from the carriage, +Y along the 2020 arm, +Z up
      paper / board top is at Z = -70 (from hbot_010.scad: rail centre 70 mm)

    Parts (set `part`, or from CLI:  openscad -D 'part="body"' -o body.stl ...)
      body   : back plate + motor cradle + rod blocks + switch plate (1 print)
      slide  : pen carriage riding on two 3 mm rods
      arm    : lever on the motor shaft
      print  : all three in print orientation (no supports needed)

    Hardware
      1x 17HS4023 pancake stepper      4x M3x8  (motor to front plate)
      4x M3x10 (body to pen carriage)  2x 3 mm steel rod, 60 mm
      2x small springs over the rods (pen springs, ~3.5 mm ID, ~20 mm free)
      2x M3x8  pen clamp screws        1x M3 grub screw (arm to shaft)
      1x M3x16 + nut (switch pusher)   1x KW10-type microswitch + 2x M2x8
*/

part = "assembly";  // [assembly, body, slide, arm, print]
$fn = 64;

// ---------------- interface to ExtruH pen carriage ----------------
carriage_holes = [[0, 15], [15, 5], [-15, 5], [0, -3]];  // [Y, Z]
paper_z = -70;
m3_clear = 3.3;
m3_tap   = 2.5;    // ExtruH convention: 2.5 mm holes, tap or self-tap M3
m2_tap   = 1.8;

// ---------------- motor ----------------
motor_w       = 42.3;
motor_len     = 23;     // body length of 17HS4023 - measure yours
motor_boss_d  = 22;
motor_hole_sp = 31;
motor_y       = -12;    // shaft position
motor_z       = -35;

// ---------------- frame ----------------
back_t  = 5;
front_t = 4;
wall    = 4;
clr     = 0.3;

mh       = motor_w / 2;
motor_x0 = back_t + 0.5;            // motor rear face
front_x0 = motor_x0 + motor_len;    // motor flange face = front plate rear
front_x1 = front_x0 + front_t;
wall_y1  = motor_y - mh - clr;
wall_y0  = wall_y1 - wall;
floor_z1 = motor_z - mh - clr;
floor_z0 = floor_z1 - wall;
body_z0  = floor_z0;
body_z1  = 0;
inner_y  = 12;                      // cradle ends here, slide starts at 13

// ---------------- pen slide ----------------
pen_d    = 11.5;        // pen barrel diameter where it is clamped
rod_d    = 3;
slide_y0 = 13;
slide_y1 = 35;
pen_y    = (slide_y0 + slide_y1) / 2;
slide_x0 = back_t + 1;
slide_x1 = 35;
rod_x    = [10, 31];
pen_x    = (rod_x[0] + rod_x[1]) / 2;
slide_h  = 20;
ledge_x1 = 46;
ledge_y1 = 26;
ledge_t  = 6;
blk_lo_z1 = body_z0 + 6;            // top of lower rod block (slide rests here)
blk_up_z0 = -14;                    // underside of upper rod block

// switch (KW10 style 20 x 10.2 x 6.4) on the +Y face of the switch plate
sw_plate_t = 3;
sw_plate_x1 = 32;
sw_cx      = pen_x;
sw_hole_z  = -6;
sw_hole_sp = 9.5;
tab_x0 = 24; tab_x1 = 33; tab_y1 = 44;
pusher_x = 28.5; pusher_y = slide_y1 + sw_plate_t + 3.2;

// ---------------- arm ----------------
arm_r  = 28;            // shaft to tip centre
arm_x0 = front_x1 + 4;  // clears motor screw heads
arm_t  = 8;
hub_d  = 14;
tip_d  = 8;
bar_w  = 6;

// ---------------- assembly pose ----------------
arm_angle = 0;
// slide bottom when its ledge rests on the arm tip
slide_z = motor_z + arm_r * sin(arm_angle) + tip_d / 2 - (slide_h - ledge_t);

echo(str("Z steps/mm at 1/16 microstep (tip arc): ", 3200 / (2 * PI * arm_r)));
echo(str("Body size X x Y x Z: ", front_x1, " x ", slide_y1 + sw_plate_t - wall_y0,
         " x ", 21 - body_z0));

// ==================================================================
module teardrop2d(d) {          // point towards +y (= up when printed)
    hull() {
        circle(d = d);
        translate([0, d / 2 * sqrt(2)]) square(0.01, center = true);
    }
}

module body() {
    difference() {
        union() {
            // back plate (lower full-width part)
            translate([0, wall_y0, body_z0])
                cube([back_t, slide_y1 + sw_plate_t - wall_y0, body_z1 - body_z0]);
            // ear covering the carriage holes
            hull() {
                for (h = carriage_holes)
                    translate([0, h[0], h[1]]) rotate([0, 90, 0]) cylinder(d = 12, h = back_t);
                translate([0, -21, -6]) cube([back_t, 42, 1]);
            }
            // outer side wall
            translate([0, wall_y0, body_z0]) cube([front_x1, wall, body_z1 - body_z0]);
            // floor under motor
            translate([0, wall_y0, floor_z0]) cube([front_x1, inner_y - wall_y0, wall]);
            // front (motor) plate
            translate([front_x0, wall_y0, body_z0])
                cube([front_t, inner_y - wall_y0, body_z1 - body_z0]);
            // lower rod block
            translate([0, inner_y - 1, body_z0])
                cube([slide_x1, slide_y1 - inner_y + 1, blk_lo_z1 - body_z0]);
            // upper rod block
            translate([0, inner_y - 1, blk_up_z0])
                cube([slide_x1, slide_y1 - inner_y + 1, body_z1 - blk_up_z0]);
            // switch plate
            translate([0, slide_y1 - 0.01, blk_up_z0])
                cube([sw_plate_x1, sw_plate_t + 0.01, body_z1 - blk_up_z0]);
        }
        // carriage screws (M3x10 from the front)
        for (h = carriage_holes)
            translate([-1, h[0], h[1]]) rotate([0, 90, 0]) cylinder(d = m3_clear, h = back_t + 2);
        // hex-key access to the lowest carriage screw through the front plate
        translate([front_x0 - 1, -3.75, -3 - 3.75]) cube([front_t + 2, 7.5, 10]);
        // motor boss and screws
        translate([front_x0 - 1, motor_y, motor_z]) rotate([0, 90, 0])
            linear_extrude(front_t + 2) teardrop2d(motor_boss_d + 1);
        for (dy = [-1, 1], dz = [-1, 1])
            translate([front_x0 - 1, motor_y + dy * motor_hole_sp / 2, motor_z + dz * motor_hole_sp / 2])
                rotate([0, 90, 0]) cylinder(d = m3_clear, h = front_t + 2);
        // guide rods (press fit, ream if tight)
        for (x = rod_x)
            translate([x, pen_y, body_z0 - 1]) cylinder(d = rod_d - 0.1, h = body_z1 - body_z0 + 2);
        // pen clearance through both blocks
        translate([pen_x, pen_y, body_z0 - 1])
            linear_extrude(body_z1 - body_z0 + 2) teardrop2d(pen_d + 3);
        // microswitch screws (self-tap M2)
        for (s = [-1, 1])
            translate([sw_cx + s * sw_hole_sp / 2, slide_y1 - 8, sw_hole_z])
                rotate([-90, 0, 0]) cylinder(d = m2_tap, h = 20);
    }
}

module slide() {        // local z = 0 at slide bottom
    difference() {
        union() {
            translate([slide_x0, slide_y0, 0])
                cube([slide_x1 - slide_x0, slide_y1 - slide_y0, slide_h]);
            // ledge the arm lifts (flush with top -> prints upside down without supports)
            translate([slide_x1 - 1, slide_y0, slide_h - ledge_t])
                cube([ledge_x1 - slide_x1 + 1, ledge_y1 - slide_y0, ledge_t]);
            // tab carrying the switch pusher screw
            translate([tab_x0, slide_y1 - 1, slide_h - 4])
                cube([tab_x1 - tab_x0, tab_y1 - slide_y1 + 1, 4]);
        }
        for (x = rod_x)
            translate([x, pen_y, -1]) cylinder(d = rod_d + 0.3, h = slide_h + 2);
        translate([pen_x, pen_y, -1]) cylinder(d = pen_d + 0.4, h = slide_h + 2);
        // pen clamp screws from +Y
        for (z = [5, 14])
            translate([pen_x, pen_y, z]) rotate([-90, 0, 0]) cylinder(d = m3_tap, h = 20);
        // switch pusher screw (M3x16 + lock nut, adjust trigger height)
        translate([pusher_x, pusher_y, slide_h - 10]) cylinder(d = m3_tap, h = 20);
    }
}

module arm2d() {
    difference() {
        union() {
            circle(d = hub_d);
            translate([-bar_w / 2, 0]) square([bar_w, arm_r]);
            translate([0, arm_r]) circle(d = tip_d);
        }
        circle(d = 5.2);
    }
}

module arm_flat() {     // shaft along z, arm along +y
    difference() {
        linear_extrude(arm_t) arm2d();
        // M3 grub screw from the side opposite the bar
        translate([0, 0, arm_t / 2]) rotate([90, 0, 0]) cylinder(d = m3_tap, h = hub_d);
    }
}

module arm_placed(a) {
    translate([arm_x0, motor_y, motor_z]) rotate([a, 0, 0]) rotate([0, 90, 0]) arm_flat();
}

// ---------------- reference geometry ----------------
module motor() {
    color("DimGray") translate([motor_x0, motor_y - mh, motor_z - mh])
        cube([motor_len, motor_w, motor_w]);
    color("Silver") translate([front_x0, motor_y, motor_z]) rotate([0, 90, 0]) {
        cylinder(d = motor_boss_d, h = 2);
        cylinder(d = 5, h = 20);
    }
}

module assembly() {
    color("Orange") body();
    motor();
    color("SkyBlue") translate([0, 0, slide_z]) slide();
    color("Gold") arm_placed(arm_angle);
    color("Silver") for (x = rod_x)
        translate([x, pen_y, body_z0]) cylinder(d = rod_d, h = body_z1 - body_z0);
    color("White", 0.6) translate([pen_x, pen_y, paper_z]) cylinder(d = pen_d, h = 130);
    color("Black") translate([sw_cx - 10, slide_y1 + sw_plate_t, sw_hole_z - 5])
        cube([20, 6.4, 10.2]);
    // ExtruH pen carriage plate and paper, for reference
    %translate([-6, -26, -22]) cube([6, 52, 44]);
    %translate([-20, -60, paper_z - 1]) cube([90, 120, 1]);
}

module print_layout() {
    // body lying on its outer side wall
    translate([0, 0, -wall_y0]) rotate([90, 0, 0]) body();
    // slide upside down
    translate([50, slide_y1 + 20, slide_h]) rotate([180, 0, 0]) slide();
    // arm flat
    translate([115, 20, 0]) arm_flat();
}

if (part == "assembly") assembly();
else if (part == "body") body();
else if (part == "slide") slide();
else if (part == "arm") arm_flat();
else if (part == "print") print_layout();
