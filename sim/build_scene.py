"""Blender script: build a procedural railway scene and render a forward cab-camera drive.

Run with Blender (no GUI needed):
    blender -b --factory-startup --python sim/build_scene.py -- --out sim_out/run1 --scenario vehicle

Scenarios
  vehicle          left-hand curve, rail vehicle standing on the ego track       -> expect BRAKE
  person           left-hand curve, person standing between the ego rails        -> expect BRAKE
  switch_blocked   straight line, facing switch set STRAIGHT, vehicle on the
                   straight track beyond the switch (in the train's route)       -> expect BRAKE
  switch_clear     same switch set to the DIVERGING (right) track; the vehicle
                   stands on the straight track the train does not take          -> expect no alert

Writes to --out:
  frames/000000.jpg ...   rendered frames (each written atomically, so a live reader is safe)
  gt.csv                  frame, time_s, camera_s_m, true_distance_m, speed_mps, in_route
  meta.json               camera intrinsics/pose, speed, scenario, obstacle, expected outcome
  done.txt                written last, when all frames are rendered
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import bpy

# ----------------------------------------------------------------------------- arguments
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="sim_out/run1")
ap.add_argument("--scenario", choices=["vehicle", "person", "switch_blocked", "switch_clear"], default="vehicle")
ap.add_argument("--speed", type=float, default=10.0, help="train speed, m/s (10 m/s = 36 km/h)")
ap.add_argument("--fps", type=float, default=10.0, help="frames per simulated second")
ap.add_argument("--obstacle-s", type=float, default=None, help="obstacle position along the track, m")
ap.add_argument("--stop-gap", type=float, default=12.0, help="end the drive this far before the obstacle, m")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--lens", type=float, default=30.0, help="focal length mm on a 36 mm sensor")
ap.add_argument("--cam-height", type=float, default=3.0, help="camera height above rail top, m")
ap.add_argument("--pitch", type=float, default=2.0, help="camera pitch down, degrees")
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args(argv)

random.seed(args.seed)
SWITCH = args.scenario.startswith("switch")
ROUTE_DIVERGE = args.scenario == "switch_clear"
OUT = os.path.abspath(args.out)
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True)
for f in ("done.txt", "gt.csv"):
    if os.path.exists(os.path.join(OUT, f)):
        os.remove(os.path.join(OUT, f))
for f in os.listdir(FRAMES):  # frames from an earlier, longer run would confuse a live reader
    if f.endswith((".jpg", ".part")):
        os.remove(os.path.join(FRAMES, f))

# ----------------------------------------------------------------------------- track geometry
GAUGE = 1.435          # inner faces of the rail heads
BALLAST_TOP = 0.42     # crib ballast, just below the sleeper tops
SLEEPER_BOT, SLEEPER_TOP = 0.28, 0.46
RAIL_BASE = SLEEPER_TOP + 0.012                 # on a rail pad / baseplate
HEAD_W, WEB_W, FOOT_W = 0.072, 0.017, 0.150     # UIC60-like rail profile, m
FOOT_H, WEB_TOP, RAIL_H = 0.012, 0.11, 0.159
RAIL_TOP = RAIL_BASE + RAIL_H
RAIL_OFF = GAUGE / 2 + HEAD_W / 2               # rail centre line from track centre
FLANGEWAY = 0.045
BLADE_LEN = 9.0
TRACK_LEN = 560.0

L_STRAIGHT, RADIUS = 90.0, 450.0          # curve layout
S_SWITCH = 200.0                          # switch layout: switch toe position on the main line
R_TURNOUT, TURNOUT_ANGLE = 190.0, math.atan(1 / 9)   # 1:9 turnout, diverging to the right
SECOND_TRACK = 4.5 if SWITCH else -4.5    # neighbouring parallel track (left in switch layout)


def main_center(s):
    """(x, y, heading) of the main (ego) line at arc length s."""
    if SWITCH or s <= L_STRAIGHT:
        return s, 0.0, 0.0
    phi = (s - L_STRAIGHT) / RADIUS
    return L_STRAIGHT + RADIUS * math.sin(phi), RADIUS * (1 - math.cos(phi)), phi


def branch_center(u):
    """Diverging track, u metres past the switch toe (curves right, then runs straight)."""
    u_curve = R_TURNOUT * TURNOUT_ANGLE
    if u <= u_curve:
        phi = u / R_TURNOUT
        return S_SWITCH + R_TURNOUT * math.sin(phi), -R_TURNOUT * (1 - math.cos(phi)), -phi
    x0, y0, th = branch_center(u_curve)
    d = u - u_curve
    return x0 + d * math.cos(th), y0 + d * math.sin(th), th


def route_center(s):
    """The path the train actually drives."""
    if SWITCH and ROUTE_DIVERGE and s > S_SWITCH:
        return branch_center(s - S_SWITCH)
    return main_center(s)


def offset(center, s, d):
    x, y, th = center(s)
    return x - d * math.sin(th), y + d * math.cos(th), th


# ----------------------------------------------------------------------------- mesh helpers
def make_obj(name, verts, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if mat:
        obj.data.materials.append(mat)
    return obj


def sweep(name, center, profile, closed, s0, s1, ds, d_base, mat, z_shift=0.0):
    """Extrude a 2-D cross-section [(d, z), ...] along a centre-line function."""
    verts, faces = [], []
    prof_at = profile if callable(profile) else (lambda _s: profile)
    n = len(prof_at(s0))
    steps = max(2, int((s1 - s0) / ds) + 1)
    for i in range(steps):
        s = s0 + (s1 - s0) * i / (steps - 1)
        for d, z in prof_at(s):
            x, y, _ = offset(center, s, d_base + d)
            verts.append((x, y, z + z_shift))
    segs = n if closed else n - 1
    for i in range(steps - 1):
        a, b = i * n, (i + 1) * n
        for k in range(segs):
            k2 = (k + 1) % n
            faces.append((a + k, a + k2, b + k2, b + k))
    return make_obj(name, verts, faces, mat)


def box_verts(cx, cy, cz, lx, ly, lz, th, base):
    """8 corners of an oriented box (length lx along heading th) and its 6 faces."""
    c, s = math.cos(th), math.sin(th)
    v = []
    for dz in (-lz / 2, lz / 2):
        for dx, dy in ((-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2)):
            v.append((cx + dx * c - dy * s, cy + dx * s + dy * c, cz + dz))
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, [tuple(base + i for i in face) for face in f]


class Batch:
    """Collects many boxes into one mesh object (fast to build and render)."""

    def __init__(self):
        self.v, self.f = [], []

    def box(self, *a):
        v, f = box_verts(*a, len(self.v))
        self.v += v
        self.f += f

    def wheel(self, cx, cy, cz, radius, width, th, n=14):
        """Cylinder whose axis runs across the track (heading th + 90 deg)."""
        base = len(self.v)
        ax, ay = -math.sin(th), math.cos(th)          # lateral axis
        fx, fy = math.cos(th), math.sin(th)           # forward
        for side in (-width / 2, width / 2):
            for k in range(n):
                a = 2 * math.pi * k / n
                px = cx + ax * side + fx * radius * math.cos(a)
                py = cy + ay * side + fy * radius * math.cos(a)
                self.v.append((px, py, cz + radius * math.sin(a)))
        for k in range(n):
            k2 = (k + 1) % n
            self.f.append((base + k, base + k2, base + n + k2, base + n + k))
        self.f.append(tuple(base + k for k in range(n)))
        self.f.append(tuple(base + n + k for k in reversed(range(n))))

    def build(self, name, mat):
        if self.v:
            make_obj(name, self.v, self.f, mat)


# ----------------------------------------------------------------------------- materials
def material(name, color, rough=0.8, metal=0.0, noise=None, bump=0.0, emission=None, stones=None):
    """Principled material; optional noise colour variation (noise=(scale, color2)) and bump."""
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emission is not None:
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (*emission, 1)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 8.0
    if noise:
        scale, color2 = noise
        coord = nt.nodes.new("ShaderNodeTexCoord")
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = 8.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*color, 1)
        ramp.color_ramp.elements[1].color = (*color2, 1)
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        if bump > 0:
            bmp = nt.nodes.new("ShaderNodeBump")
            bmp.inputs["Strength"].default_value = bump
            height = tex.outputs["Fac"]
            if stones:  # cellular pattern: individual stones
                vor = nt.nodes.new("ShaderNodeTexVoronoi")
                vor.inputs["Scale"].default_value = stones
                nt.links.new(coord.outputs["Object"], vor.inputs["Vector"])
                height = vor.outputs["Distance"]
            nt.links.new(height, bmp.inputs["Height"])
            nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return m


M_BALLAST = material("ballast", (0.30, 0.29, 0.28), 0.95, noise=(45.0, (0.58, 0.56, 0.53)), bump=1.0, stones=28.0)
M_SLEEPER = material("sleeper", (0.55, 0.54, 0.52), 0.9, noise=(6.0, (0.42, 0.41, 0.40)), bump=0.15)
M_TIMBER = material("timber", (0.22, 0.15, 0.10), 0.9, noise=(5.0, (0.15, 0.10, 0.07)), bump=0.2)
M_RAIL = material("rail_head", (0.72, 0.72, 0.74), 0.22, 1.0)      # polished running surface
M_RUST = material("rail_rust", (0.36, 0.20, 0.11), 0.8, 0.3, noise=(20.0, (0.25, 0.14, 0.08)), bump=0.1)
M_CLIP = material("fastening", (0.10, 0.10, 0.11), 0.6, 0.6)
M_MACHINE = material("point_machine", (0.42, 0.44, 0.46), 0.5, 0.5)
M_GROUND = material("grass", (0.16, 0.25, 0.08), 1.0, noise=(2.0, (0.30, 0.33, 0.14)), bump=0.3)
M_TRUNK = material("trunk", (0.18, 0.12, 0.08), 0.9)
M_LEAF = material("leaves", (0.07, 0.17, 0.05), 0.9, noise=(3.0, (0.12, 0.25, 0.07)))
M_BODY = material("loco_body", (0.62, 0.08, 0.06), 0.45, 0.3)
M_STRIPE = material("loco_stripe", (0.85, 0.85, 0.82), 0.5, 0.2)
M_ROOF = material("loco_roof", (0.35, 0.36, 0.38), 0.6, 0.4)
M_DARK = material("loco_dark", (0.04, 0.04, 0.05), 0.3, 0.3)
M_METAL = material("wheel", (0.20, 0.20, 0.21), 0.5, 0.8)
M_LAMP = material("lamp", (1.0, 0.95, 0.8), 0.2, emission=(1.0, 0.95, 0.8))
M_SHIRT = material("hi_vis", (0.95, 0.55, 0.05), 0.7)
M_SKIN = material("skin", (0.55, 0.38, 0.28), 0.7)
M_TROUSER = material("trouser", (0.08, 0.10, 0.18), 0.8)

# ----------------------------------------------------------------------------- scene
scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)


def rail(name, center, d, s0, s1, ds=1.0, d_base=0.0):
    """A rail with foot, web and head along `center` at lateral offset d."""
    if s1 - s0 < 0.05:
        return
    for part, (w, z0, z1, m) in {"foot": (FOOT_W, RAIL_BASE, RAIL_BASE + FOOT_H, M_RUST),
                                 "web": (WEB_W, RAIL_BASE + FOOT_H, RAIL_BASE + WEB_TOP, M_RUST),
                                 "head": (HEAD_W, RAIL_BASE + WEB_TOP, RAIL_TOP, M_RAIL)}.items():
        prof = [(d - w / 2, z0), (d + w / 2, z0), (d + w / 2, z1), (d - w / 2, z1)]
        sweep(f"{name}_{part}", center, prof, True, s0, s1, ds, d_base, m)


def inside(s, segs):
    return any(a <= s <= b for a, b in segs)


def build_track(name, center, d0, s0, s1, z_shift=0.0, rails=None, sleeper_skip=()):
    """Ballast, sleepers with fastenings and rails. rails: {side: [(s_a, s_b), ...]} (default: full length)."""
    rails = rails or {-1: [(s0, s1)], 1: [(s0, s1)]}
    sweep(f"{name}_ballast", center, [(-2.5, 0.0), (-1.75, BALLAST_TOP), (1.75, BALLAST_TOP), (2.5, 0.0)],
          False, s0, s1, 2.0, d0, M_BALLAST, z_shift)
    for side, segs in rails.items():
        for k, (a, b) in enumerate(segs):
            rail(f"{name}_rail{side}_{k}", center, side * RAIL_OFF, a, b, 1.0, d0)
    sleepers, clips = Batch(), Batch()
    s = s0 + 0.3
    while s < s1:
        if not inside(s, sleeper_skip):
            x, y, th = offset(center, s, d0)
            sleepers.box(x, y, (SLEEPER_BOT + SLEEPER_TOP) / 2 + z_shift, 0.26, 2.6, SLEEPER_TOP - SLEEPER_BOT, th)
            for side, segs in rails.items():
                if inside(s, segs):
                    for dd in (-0.11, 0.11):
                        cx, cy, _ = offset(center, s, d0 + side * RAIL_OFF + dd)
                        clips.box(cx, cy, RAIL_BASE + 0.02, 0.10, 0.07, 0.05, th)
        s += 0.6
    sleepers.build(f"{name}_sleepers", M_SLEEPER)
    clips.build(f"{name}_clips", M_CLIP)


def solve_frog():
    """Where the diverging route's left rail crosses the straight route's right rail:
    returns (s on the main line, u on the branch)."""
    lo, hi = 5.0, 80.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if offset(branch_center, mid, RAIL_OFF)[1] > -RAIL_OFF:
            lo = mid
        else:
            hi = mid
    u = (lo + hi) / 2
    return offset(branch_center, u, RAIL_OFF)[0], u


if SWITCH:
    S_FROG, U_FROG = solve_frog()
    throat, nose = 1.0, 0.3          # rail ends either side of the theoretical crossing point
    zone = (S_SWITCH - 2.0, S_FROG + 12.0)
    # straight route: left stock rail runs through; its right rail is stock rail -> switch blade
    # -> closure rail -> frog nose -> plain rail
    build_track("main", main_center, 0.0, -20.0, TRACK_LEN, sleeper_skip=[zone],
                rails={1: [(-20.0, TRACK_LEN)],
                       -1: [(-20.0, S_SWITCH), (S_SWITCH + BLADE_LEN, S_FROG - throat), (S_FROG + nose, TRACK_LEN)]})
    # diverging route: right rail is the curved stock rail; left rail is blade -> closure -> nose
    u_end = TRACK_LEN - S_SWITCH
    build_track("branch", branch_center, 0.0, 0.0, u_end, z_shift=-0.006, sleeper_skip=[(0.0, zone[1] - S_SWITCH)],
                rails={-1: [(0.0, u_end)], 1: [(BLADE_LEN, U_FROG - throat), (U_FROG + nose, u_end)]})

    # switch blades, tapered from a thin tip at the toe to full head width; set to the
    # diverging route the LEFT blade lies against the left stock rail, set straight the RIGHT one.
    def blade_profile(side, closed):
        def prof(u):
            w = 0.015 + (HEAD_W - 0.015) * min(1.0, u / 6.0)
            opening = 0.0 if closed else 0.11 * (1.0 - u / BLADE_LEN)  # open at the tip, fixed at the heel
            d = side * (RAIL_OFF - HEAD_W / 2 - w / 2) - side * opening
            return [(d - w / 2, RAIL_BASE), (d + w / 2, RAIL_BASE), (d + w / 2, RAIL_TOP - 0.004),
                    (d - w / 2, RAIL_TOP - 0.004)]
        return prof

    for side in (-1, 1):
        closed = (side == 1) == ROUTE_DIVERGE
        along = branch_center if side == 1 else (lambda u: main_center(S_SWITCH + u))
        sweep(f"blade{side}", along, blade_profile(side, closed), True, 0.0, BLADE_LEN, 0.25, 0.0, M_RAIL)

    def plain(name, center, d, a, b):
        prof = [(d - HEAD_W / 2, RAIL_BASE), (d + HEAD_W / 2, RAIL_BASE),
                (d + HEAD_W / 2, RAIL_TOP - 0.006), (d - HEAD_W / 2, RAIL_TOP - 0.006)]
        sweep(name, center, prof, True, a, b, 0.5, 0.0, M_RUST)

    gap = FLANGEWAY + HEAD_W
    # wing rails beside the two nose rails, check rails opposite the frog on the outer rails
    plain("wing_straight", main_center, -RAIL_OFF + gap, S_FROG - 0.6, S_FROG + 2.5)
    plain("wing_diverging", branch_center, RAIL_OFF - gap, U_FROG - 0.6, U_FROG + 2.5)
    plain("check_straight", main_center, RAIL_OFF - gap, S_FROG - 3.0, S_FROG + 3.0)
    plain("check_diverging", branch_center, -RAIL_OFF + gap, U_FROG - 3.0, U_FROG + 3.0)

    # long wooden switch timbers carrying both routes, with fastenings
    timbers, tclips = Batch(), Batch()
    s = zone[0]
    while s < zone[1]:
        by = branch_center(max(s - S_SWITCH, 0.0))[1] if s > S_SWITCH else 0.0
        y_l, y_r = 1.45, min(-1.45, by - 1.45)
        timbers.box(s, (y_l + y_r) / 2, (SLEEPER_BOT + SLEEPER_TOP) / 2, 0.26, y_l - y_r, SLEEPER_TOP - SLEEPER_BOT, 0.0)
        for y in (RAIL_OFF, -RAIL_OFF, by - RAIL_OFF):
            for dd in (-0.11, 0.11):
                tclips.box(s, y + dd, RAIL_BASE + 0.02, 0.10, 0.07, 0.05, 0.0)
        s += 0.6
    timbers.build("switch_timbers", M_TIMBER)
    tclips.build("switch_clips", M_CLIP)

    # point machine beside the toe, with its drive rod to the blades
    pm = Batch()
    pm.box(S_SWITCH + 0.9, RAIL_OFF + 1.25, RAIL_BASE + 0.12, 1.4, 0.55, 0.45, 0.0)
    pm.box(S_SWITCH + 0.6, 0.35, RAIL_BASE + 0.03, 0.05, 2.6, 0.04, 0.0)
    pm.build("point_machine", M_MACHINE)
else:
    build_track("main", main_center, 0.0, -20.0, TRACK_LEN)
build_track("second", main_center, SECOND_TRACK, -20.0, TRACK_LEN)

# ground
g = 1500.0
make_obj("ground", [(-g, -g, -0.02), (g, -g, -0.02), (g, g, -0.02), (-g, g, -0.02)], [(0, 1, 2, 3)], M_GROUND)

# trees, kept clear of the tracks, the inside of the curve and the diverging track
trunks, crowns = Batch(), Batch()
for _ in range(420):
    s = random.uniform(-20, TRACK_LEN)
    side = random.choice((-1, 1))
    if not SWITCH and side > 0 and s > L_STRAIGHT - 30:
        side = -1
    if SWITCH and side < 0 and s > S_SWITCH - 20:
        continue
    lo = 9 + (abs(SECOND_TRACK) if (side > 0) == (SECOND_TRACK > 0) else 0)
    d = side * random.uniform(lo, lo + 50)
    x, y, th = offset(main_center, s, d)
    h = random.uniform(6, 14)
    trunks.box(x, y, h * 0.2, 0.4, 0.4, h * 0.4, th)
    w = random.uniform(2.5, 5.0)
    crowns.box(x, y, h * 0.7, w, w, h * 0.6, th + random.uniform(0, 1))
trunks.build("trunks", M_TRUNK)
crowns.build("crowns", M_LEAF)


def build_vehicle(center, s_rear):
    """A locomotive-like rail vehicle whose rear (camera-facing) end is at s_rear."""
    L, W = 18.0, 3.0
    z_floor = RAIL_TOP + 1.15
    body, stripe, dark, roof, metal, lamps = Batch(), Batch(), Batch(), Batch(), Batch(), Batch()
    xm, ym, th = offset(center, s_rear + L / 2, 0.0)
    body.box(xm, ym, z_floor + 1.5, L, W, 3.0, th)                      # body shell
    stripe.box(xm, ym, z_floor + 0.9, L + 0.02, W + 0.02, 0.35, th)       # livery band
    roof.box(xm, ym, z_floor + 3.15, L - 1.0, W - 0.5, 0.3, th)           # roof equipment
    dark.box(xm, ym, RAIL_TOP + 0.85, L - 1.0, W - 0.5, 0.6, th)          # underframe
    for k in range(-3, 4):                                               # side windows
        for side in (-1, 1):
            x, y, _ = offset(center, s_rear + L / 2 + k * 2.2, side * (W / 2 + 0.01))
            dark.box(x, y, z_floor + 2.2, 1.4, 0.02, 0.8, th)
    xr, yr, _ = offset(center, s_rear - 0.01, 0.0)
    dark.box(xr, yr, z_floor + 2.15, 0.03, W * 0.8, 0.9, th)             # windscreen
    for side in (-1, 1):
        x, y, _ = offset(center, s_rear - 0.02, side * 0.95)
        lamps.box(x, y, z_floor + 0.45, 0.03, 0.28, 0.22, th)            # headlights
        x, y, _ = offset(center, s_rear - 0.3, side * 0.9)
        metal.box(x, y, RAIL_TOP + 1.05, 0.6, 0.35, 0.35, th)             # buffers
    for bogie in (2.8, L - 2.8):                                         # two bogies, two axles each
        for ax in (-1.1, 1.1):
            for side in (-1, 1):
                x, y, th2 = offset(center, s_rear + bogie + ax, side * (RAIL_OFF - 0.05))
                metal.wheel(x, y, RAIL_TOP + 0.46, 0.46, 0.14, th2)
    for b, n, m in ((body, "body", M_BODY), (stripe, "stripe", M_STRIPE), (roof, "roof", M_ROOF),
                    (dark, "dark", M_DARK), (metal, "metal", M_METAL), (lamps, "lamps", M_LAMP)):
        b.build(f"vehicle_{n}", m)
    return {"length": L, "width": W, "height": round(z_floor + 3.3 - RAIL_TOP, 2)}


# obstacle
if SWITCH:
    s_obs = args.obstacle_s or (S_SWITCH + 60.0)
    obs = {"scenario": args.scenario, "s": s_obs, "track": "main (straight)",
           "route": "diverging" if ROUTE_DIVERGE else "straight"}
    obs.update(build_vehicle(main_center, s_obs))
    in_route = not ROUTE_DIVERGE
elif args.scenario == "vehicle":
    s_obs = args.obstacle_s or 300.0
    obs = {"scenario": args.scenario, "s": s_obs}
    obs.update(build_vehicle(main_center, s_obs))
    in_route = True
else:
    s_obs = args.obstacle_s or 300.0
    obs = {"scenario": args.scenario, "s": s_obs, "height": 1.76}
    x, y, th = offset(main_center, s_obs, 0.0)
    for i, (w, dpt, h, zc, m) in enumerate([(0.45, 0.30, 0.85, RAIL_TOP + 0.42, M_TROUSER),
                                            (0.5, 0.32, 0.65, RAIL_TOP + 1.17, M_SHIRT),
                                            (0.24, 0.24, 0.26, RAIL_TOP + 1.63, M_SKIN)]):
        v, f = box_verts(x, y, zc, dpt, w, h, th, 0)
        make_obj(f"person_{i}", v, f, m)
    in_route = True
obs["in_route"] = in_route

# lighting and sky
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
sun.data.energy = 4.0
sun.rotation_euler = (math.radians(50), 0, math.radians(-30))
scene.collection.objects.link(sun)
world = bpy.data.worlds.new("sky")
scene.world = world
try:
    world.use_nodes = True
except Exception:
    pass
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value = (0.55, 0.70, 0.90, 1)
bg.inputs[1].default_value = 1.0

# camera
cam_data = bpy.data.cameras.new("cab_cam")
cam_data.lens = args.lens
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.clip_start = 0.1
cam_data.clip_end = 3000
cam = bpy.data.objects.new("cab_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# render settings
r = scene.render
if args.engine == "cycles":
    r.engine = "CYCLES"
    scene.cycles.samples = args.samples
else:
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            r.engine = eng
            break
        except TypeError:
            continue
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = args.samples
r.resolution_x, r.resolution_y, r.resolution_percentage = args.width, args.height, 100
r.image_settings.file_format = "JPEG"
r.image_settings.quality = 92
r.use_file_extension = False

# ----------------------------------------------------------------------------- drive
f_px = args.lens / 36.0 * args.width
end_s = (S_SWITCH + 170.0) if (SWITCH and ROUTE_DIVERGE) else (s_obs - args.stop_gap)
step = args.speed / args.fps
n_frames = int(end_s / step) + 1
expected = "BRAKE in time" if in_route else "no WARNING or BRAKE (vehicle is not on the train's route)"
meta = {
    "width": args.width, "height": args.height, "fps": args.fps, "speed_mps": args.speed,
    "f_px": f_px, "cx": args.width / 2, "cy": args.height / 2,
    "cam_height_above_rail_m": args.cam_height, "pitch_deg": args.pitch,
    "gauge_m": GAUGE, "obstacle": obs, "n_frames": n_frames, "expected": expected,
    "layout": "switch" if SWITCH else "curve",
    "switch_s_m": S_SWITCH if SWITCH else None,
    "turnout_side": "right" if SWITCH else None,
}
with open(os.path.join(OUT, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)

gt = open(os.path.join(OUT, "gt.csv"), "w", newline="")
w = csv.writer(gt)
w.writerow(["frame", "time_s", "camera_s_m", "true_distance_m", "speed_mps", "in_route"])
ox, oy, _ = offset(main_center, s_obs, 0.0)
t0 = time.time()
print(f"Rendering {n_frames} frames ({args.scenario}, {args.speed} m/s, {args.fps} fps) to {FRAMES}", flush=True)
for i in range(n_frames):
    s = i * step
    x, y, th = route_center(s)
    cam.location = (x, y, RAIL_TOP + args.cam_height)
    cam.rotation_euler = (math.radians(90 - args.pitch), 0.0, th - math.pi / 2)
    final = os.path.join(FRAMES, f"{i:06d}.jpg")
    tmp = final + ".part"
    r.filepath = tmp
    bpy.ops.render.render(write_still=True)
    os.replace(tmp, final)  # atomic: a live reader never sees a half-written frame
    dist = math.hypot(ox - x, oy - y)
    ahead = (ox - x) * math.cos(th) + (oy - y) * math.sin(th) > 0
    w.writerow([i, f"{i / args.fps:.2f}", f"{s:.2f}", f"{dist:.2f}" if ahead else "", args.speed, int(in_route)])
    gt.flush()
    if (i + 1) % 10 == 0 or i == 0:
        el = time.time() - t0
        print(f"  frame {i + 1}/{n_frames}  {el / (i + 1):.2f} s/frame  distance {dist:.0f} m", flush=True)
gt.close()
with open(os.path.join(OUT, "done.txt"), "w") as fh:
    fh.write("done\n")
print(f"Done in {time.time() - t0:.0f} s -> {OUT}", flush=True)
