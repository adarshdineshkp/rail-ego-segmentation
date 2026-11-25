"""Blender script: build a procedural railway scene and render a forward cab-camera drive.

Run with Blender (no GUI needed):
    blender -b --factory-startup --python sim/build_scene.py -- --out sim_out/run1

Scene: a straight section that turns into a left-hand curve, the ego track plus a parallel
track 4.5 m to the right, ballast, sleepers, rails, grass, trees, and (scenario "vehicle")
a rail vehicle standing on the ego track in the curve, or (scenario "person") a person
standing between the rails. The camera rides on the ego track at a fixed height and speed.

Writes to --out:
  frames/000000.jpg ...   rendered frames (each written atomically, so a live reader is safe)
  gt.csv                  frame, time_s, camera_s_m, true_distance_m, speed_mps
  meta.json               camera intrinsics/pose, speed, scenario, obstacle position
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
ap.add_argument("--scenario", choices=["vehicle", "person"], default="vehicle")
ap.add_argument("--speed", type=float, default=10.0, help="train speed, m/s (10 m/s = 36 km/h)")
ap.add_argument("--fps", type=float, default=10.0, help="frames per simulated second")
ap.add_argument("--obstacle-s", type=float, default=300.0, help="obstacle position along the track, m")
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
OUT = os.path.abspath(args.out)
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True)
for f in ("done.txt", "gt.csv"):
    if os.path.exists(os.path.join(OUT, f)):
        os.remove(os.path.join(OUT, f))
for f in os.listdir(FRAMES):  # frames from an earlier, longer run would confuse a live reader
    if f.endswith((".jpg", ".part")):
        os.remove(os.path.join(FRAMES, f))

# ----------------------------------------------------------------------------- geometry
L_STRAIGHT = 90.0      # straight section, m
RADIUS = 450.0         # left-hand curve radius, m
TRACK_LEN = 520.0
GAUGE = 1.435          # inner faces of the rails
RAIL_W, RAIL_H = 0.07, 0.16
BALLAST_TOP = 0.30
SLEEPER_TOP = 0.44
RAIL_TOP = SLEEPER_TOP + RAIL_H
RAIL_OFF = GAUGE / 2 + RAIL_W / 2
SECOND_TRACK = -4.5    # lateral offset of the neighbouring track (right side)


def centerline(s):
    """(x, y, heading) of the ego-track centre line at arc length s."""
    if s <= L_STRAIGHT:
        return s, 0.0, 0.0
    phi = (s - L_STRAIGHT) / RADIUS
    return L_STRAIGHT + RADIUS * math.sin(phi), RADIUS * (1 - math.cos(phi)), phi


def offset_point(s, d):
    """Point at arc length s, lateral offset d (positive = left)."""
    x, y, th = centerline(s)
    return x - d * math.sin(th), y + d * math.cos(th), th


def sweep(name, profile, closed, s0, s1, ds, d_base, mat):
    """Extrude a 2-D cross-section [(d, z), ...] along the track."""
    verts, faces = [], []
    n = len(profile)
    steps = max(2, int((s1 - s0) / ds) + 1)
    for i in range(steps):
        s = s0 + (s1 - s0) * i / (steps - 1)
        for d, z in profile:
            x, y, _ = offset_point(s, d_base + d)
            verts.append((x, y, z))
    segs = n if closed else n - 1
    for i in range(steps - 1):
        a, b = i * n, (i + 1) * n
        for k in range(segs):
            k2 = (k + 1) % n
            faces.append((a + k, a + k2, b + k2, b + k))
    return make_obj(name, verts, faces, mat)


def make_obj(name, verts, faces, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if mat:
        obj.data.materials.append(mat)
    return obj


def box_verts(cx, cy, cz, lx, ly, lz, th, base):
    """8 corners of an oriented box (length lx along heading th) and its 6 faces."""
    c, s = math.cos(th), math.sin(th)
    v = []
    for dz in (-lz / 2, lz / 2):
        for dx, dy in ((-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2)):
            v.append((cx + dx * c - dy * s, cy + dx * s + dy * c, cz + dz))
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, [tuple(base + i for i in face) for face in f]


# ----------------------------------------------------------------------------- materials
def material(name, color, rough=0.8, metal=0.0, noise=None, bump=0.0, emission=None):
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
            nt.links.new(tex.outputs["Fac"], bmp.inputs["Height"])
            nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return m


M_BALLAST = material("ballast", (0.33, 0.31, 0.29), 0.95, noise=(60.0, (0.55, 0.53, 0.50)), bump=0.6)
M_SLEEPER = material("sleeper", (0.42, 0.40, 0.38), 0.9, noise=(8.0, (0.30, 0.28, 0.26)), bump=0.2)
M_RAIL = material("rail", (0.55, 0.52, 0.50), 0.35, 0.9)
M_GROUND = material("grass", (0.16, 0.25, 0.08), 1.0, noise=(2.0, (0.30, 0.33, 0.14)), bump=0.3)
M_TRUNK = material("trunk", (0.18, 0.12, 0.08), 0.9)
M_LEAF = material("leaves", (0.07, 0.17, 0.05), 0.9, noise=(3.0, (0.12, 0.25, 0.07)))
M_BODY = material("loco_body", (0.75, 0.45, 0.05), 0.5, 0.3)
M_DARK = material("loco_dark", (0.05, 0.05, 0.06), 0.4, 0.2)
M_LAMP = material("lamp", (1.0, 0.95, 0.8), 0.2, emission=(1.0, 0.95, 0.8))
M_SHIRT = material("hi_vis", (0.95, 0.55, 0.05), 0.7)
M_SKIN = material("skin", (0.55, 0.38, 0.28), 0.7)
M_TROUSER = material("trouser", (0.08, 0.10, 0.18), 0.8)

# ----------------------------------------------------------------------------- scene
scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)


def build_track(name, d0, s0, s1):
    sweep(f"{name}_ballast", [(-2.1, 0.0), (-1.55, BALLAST_TOP), (1.55, BALLAST_TOP), (2.1, 0.0)],
          False, s0, s1, 2.0, d0, M_BALLAST)
    for side in (-1, 1):
        r = side * RAIL_OFF
        prof = [(r - RAIL_W / 2, SLEEPER_TOP), (r + RAIL_W / 2, SLEEPER_TOP),
                (r + RAIL_W / 2, RAIL_TOP), (r - RAIL_W / 2, RAIL_TOP)]
        sweep(f"{name}_rail{side}", prof, True, s0, s1, 1.0, d0, M_RAIL)
    verts, faces = [], []
    s = s0 + 0.3
    while s < s1:
        x, y, th = offset_point(s, d0)
        v, f = box_verts(x, y, (BALLAST_TOP + SLEEPER_TOP) / 2 + 0.01, 0.25, 2.6,
                         SLEEPER_TOP - BALLAST_TOP, th, len(verts))
        verts += v
        faces += f
        s += 0.62
    make_obj(f"{name}_sleepers", verts, faces, M_SLEEPER)


build_track("ego", 0.0, -20.0, TRACK_LEN)
build_track("second", SECOND_TRACK, -20.0, TRACK_LEN)

# ground
g = 1500.0
make_obj("ground", [(-g, -g, -0.02), (g, -g, -0.02), (g, g, -0.02), (-g, g, -0.02)], [(0, 1, 2, 3)], M_GROUND)

# trees, kept clear of both tracks
tv, tf, lv, lf = [], [], [], []
for _ in range(420):
    s = random.uniform(-20, TRACK_LEN)
    side = random.choice((-1, 1))
    if side > 0 and s > L_STRAIGHT - 30:
        side = -1  # keep the inside of the curve open so the line of sight is not blocked
    d = side * random.uniform(9, 60) + (SECOND_TRACK if side < 0 else 0)
    x, y, th = offset_point(s, d)
    h = random.uniform(6, 14)
    v, f = box_verts(x, y, h * 0.2, 0.4, 0.4, h * 0.4, th, len(tv))
    tv += v
    tf += f
    w = random.uniform(2.5, 5.0)
    v, f = box_verts(x, y, h * 0.4 + h * 0.3, w, w, h * 0.6, th + random.uniform(0, 1), len(lv))
    lv += v
    lf += f
make_obj("trunks", tv, tf, M_TRUNK)
make_obj("crowns", lv, lf, M_LEAF)

# obstacle
obs = {"scenario": args.scenario, "s": args.obstacle_s}
if args.scenario == "vehicle":
    L, W, H = 16.0, 3.0, 3.3
    x, y, th = offset_point(args.obstacle_s + L / 2, 0.0)
    z0 = RAIL_TOP + 0.9
    verts, faces = box_verts(x, y, z0 + H / 2, L, W, H, th, 0)
    make_obj("vehicle_body", verts, faces, M_BODY)
    v2, f2 = box_verts(x, y, RAIL_TOP + 0.45, L - 2, W - 0.4, 0.9, th, 0)  # bogies / underframe
    make_obj("vehicle_under", v2, f2, M_DARK)
    # rear face details: dark window band and two lamps facing the camera
    rx, ry, _ = offset_point(args.obstacle_s - 0.02, 0.0)
    v3, f3 = box_verts(rx, ry, z0 + H * 0.72, 0.05, W * 0.8, 0.8, th, 0)
    make_obj("vehicle_window", v3, f3, M_DARK)
    for side in (-1, 1):
        lx, ly, _ = offset_point(args.obstacle_s - 0.03, side * 0.9)
        v4, f4 = box_verts(lx, ly, z0 + 0.5, 0.05, 0.3, 0.3, th, 0)
        make_obj(f"vehicle_lamp{side}", v4, f4, M_LAMP)
    obs.update(length=L, width=W, height=H)
else:
    x, y, th = offset_point(args.obstacle_s, 0.0)
    parts = [(0.45, 0.30, 0.85, RAIL_TOP + 0.42, M_TROUSER),   # legs
             (0.5, 0.32, 0.65, RAIL_TOP + 1.17, M_SHIRT),      # torso
             (0.24, 0.24, 0.26, RAIL_TOP + 1.63, M_SKIN)]      # head
    for i, (w, dpt, h, zc, m) in enumerate(parts):
        v, f = box_verts(x, y, zc, dpt, w, h, th, 0)
        make_obj(f"person_{i}", v, f, m)
    obs.update(height=1.76)

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
end_s = args.obstacle_s - args.stop_gap
step = args.speed / args.fps
n_frames = int(end_s / step) + 1
meta = {
    "width": args.width, "height": args.height, "fps": args.fps, "speed_mps": args.speed,
    "f_px": f_px, "cx": args.width / 2, "cy": args.height / 2,
    "cam_height_above_rail_m": args.cam_height, "pitch_deg": args.pitch,
    "gauge_m": GAUGE, "obstacle": obs, "n_frames": n_frames,
    "track": {"straight_m": L_STRAIGHT, "radius_m": RADIUS, "second_track_offset_m": SECOND_TRACK},
}
with open(os.path.join(OUT, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)

gt = open(os.path.join(OUT, "gt.csv"), "w", newline="")
w = csv.writer(gt)
w.writerow(["frame", "time_s", "camera_s_m", "true_distance_m", "speed_mps"])
ox, oy, _ = offset_point(args.obstacle_s, 0.0)
t0 = time.time()
print(f"Rendering {n_frames} frames ({args.scenario}, {args.speed} m/s, {args.fps} fps) to {FRAMES}")
for i in range(n_frames):
    s = i * step
    x, y, th = offset_point(s, 0.0)
    cam.location = (x, y, RAIL_TOP + args.cam_height)
    cam.rotation_euler = (math.radians(90 - args.pitch), 0.0, th - math.pi / 2)
    final = os.path.join(FRAMES, f"{i:06d}.jpg")
    tmp = final + ".part"
    r.filepath = tmp
    bpy.ops.render.render(write_still=True)
    os.replace(tmp, final)  # atomic: a live reader never sees a half-written frame
    dist = math.hypot(ox - x, oy - y)
    w.writerow([i, f"{i / args.fps:.2f}", f"{s:.2f}", f"{dist:.2f}", args.speed])
    gt.flush()
    if (i + 1) % 10 == 0 or i == 0:
        el = time.time() - t0
        print(f"  frame {i + 1}/{n_frames}  {el / (i + 1):.2f} s/frame  distance {dist:.0f} m")
gt.close()
with open(os.path.join(OUT, "done.txt"), "w") as fh:
    fh.write("done\n")
print(f"Done in {time.time() - t0:.0f} s -> {OUT}")
