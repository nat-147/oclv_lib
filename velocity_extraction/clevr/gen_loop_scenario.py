# -*- coding: utf-8 -*-
"""CLEVR 시나리오 A-loop: 시나리오 A(S1) 배치 + ContinuousShiftingSpritesloops(contshiftingsprites_occlusion.yaml)의 폐곡선 경로

  blender -b -P gen_loop_scenario.py -- --out OUT/S1L --start 0 --num 4 --seed 0 [--size 128] [--amodal 1] [--dry 1]

경로 (velocity_extraction/data.py generate_closed_path와 같은 방식)
  1) 걸음 K개(--walk_len, 기본 10): 걸음마다 x, y 이동량을 ±max_shift에서 균등하게 뽑음
  2) --repeat_twice 1이면 걸음마다 두 번씩 반복 (2K걸음)
  3) 같은 걸음들을 부호만 바꿔 이어 붙임 → 출발점으로 돌아오는 폐곡선. 프레임 수 = 4K+1 (repeat 0이면 2K+1). 기본 41
  4) 경로 평균을 0에 맞추고, x·y 범위가 모두 (1 − 1.5·r_norm)보다 작을 때만 채택
  원본 값은 화면 너비 = 1 단위. 바닥 평면에서 × --world_scale(기본 5)을 월드 단위로 씀 → 걸음당 최대 0.5, 경로 범위 < 1.925
배치 (시나리오 A의 traj_S1과 같은 방식)
  - 모든 객체가 같은 경로를 함께 이동 (same velocity). 원 대신 랜덤 폐곡선이라는 점만 A와 다름
  - 객체마다 경로 평균 위치를 반경 --disk(기본 2.2) 원판 안에서 면적 균등으로 뽑음
  - 바닥 평면 거리 > --gap·(r_i + r_j) (기본 1.5, A와 같음). 3D 충돌(외접구)도 모든 프레임에서 확인
가려짐 (원본 occlusion_ratio 대신 visible_ratio 기준)
  - 렌더링 전에 객체 실루엣을 투영해(볼록 도형이라 투영 꼭짓점의 볼록 껍질 = 실루엣) 카메라에서 먼 객체부터 칠하고
    객체별·프레임별 보이는 비율을 추정
  - --vis_min v  : 모든 객체가 모든 프레임에서 추정 보이는 비율 ≥ v 일 때만 채택 (기본 0.3. 0이면 끔)
  - --vis_need v : 어떤 객체가 어떤 프레임에서 추정 보이는 비율 < v 인 순간이 있어야 채택 = 가려짐을 반드시 포함 (기본 0 = 끔)
화면 (모델 입력 크기로 바로 렌더링)
  - --size 128: SIZE×SIZE로 렌더링. 카메라 위치·방향은 CLEVR 원본과 같고 초점거리만 --lens(기본 66.67mm = Blender 기본 50mm × 320/240)
    → 원본 320×240 화면의 가운데 240×240(행 0:240, 열 40:280)과 같은 화각. crop·resize 불필요
  - 객체 중심이 화면 안(여백 5%), --in_frame 1.0: 객체 실루엣이 이 비율 이상 화면 안 (1 = 가장자리에서 잘리지 않음)
  - 배치 중심은 화면 중심 아래 바닥 점
  - 렌더링 후 --amodal 1이면 실제 visible_ratio를 frames에 저장(gen_3d_scenarios.py와 같음). 추정값은 visible_est로 같이 저장
객체 수: --counts "2:0.05,3:0.20,4:0.25,5:0.25,6:0.25" (기본. 합으로 나눠 정규화)
나머지(모양·재질·색·크기·카메라·조명·바닥 고정·mask·amodal·meta.json 형식)는 gen_3d_scenarios.py와 같음.
meta.json의 trajectory에 경로(path_xy, 평균 0), 걸음별 이동량(velocity_xy, 월드 단위, 길이 프레임수−1)을 저장.
"""
import bpy, sys, os, json, math, time, shutil
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def arg(name, default, typ=str): return typ(argv[argv.index(name)+1]) if name in argv else default
OUT = arg("--out", "OUT/S1L"); START = arg("--start", 0, int); NUM = arg("--num", 2, int)
SEED = arg("--seed", 0, int); NOBJ = arg("--nobj", 0, int); AMODAL = arg("--amodal", 1, int); DEVICE = arg("--device", "CUDA")
KEEP_AMODAL = arg("--keep_amodal", 0, int); DRY = arg("--dry", 0, int)
SIZE = arg("--size", 128, int); LENS = arg("--lens", 0.0, float); SPP = arg("--spp", 64, int)   # LENS 0 = 기본 카메라 초점거리(50mm) × 320/240
RES = (SIZE, SIZE)
# 경로: contshiftingsprites_occlusion.yaml 값 (화면 너비 = 1 단위)
WALK = arg("--walk_len", 10, int); REPEAT = arg("--repeat_twice", 1, int)
R_NORM = arg("--r_norm", 0.41, float); MAX_SHIFT = arg("--max_shift", 0.1, float)
WS = arg("--world_scale", 5.0, float); MAX_TRIES = arg("--max_tries", 800, int)
# 배치: 시나리오 A 값 (월드 단위)
DISK = arg("--disk", 2.2, float); GAP = arg("--gap", 1.5, float)
IN_FRAME = arg("--in_frame", 1.0, float)   # 모든 객체가 모든 프레임에서 실루엣 면적의 이 비율 이상이 화면 안에 있어야 채택 (1 = 잘리지 않음, 0 = 끔)
# 가려짐: 추정 visible ratio 기준
VIS_MIN = arg("--vis_min", 0.3, float); VIS_NEED = arg("--vis_need", 0.0, float)
_cnt = [kv.split(":") for kv in arg("--counts", "2:0.05,3:0.20,4:0.25,5:0.25,6:0.25").split(",")]
COUNTS = np.array([int(k) for k, _ in _cnt]); PROBS = np.array([float(p) for _, p in _cnt]); PROBS /= PROBS.sum()
NF = 4*WALK + 1 if REPEAT else 2*WALK + 1
RMIN, RMAX = 0.36, 0.66
CAM_LOC = Vector((7.48, -6.51, 5.34))

# ---------------- 장면 (gen_3d_scenarios.py와 같음)
scn = bpy.context.scene
for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
scn.render.engine = "CYCLES"
pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = DEVICE; pr.get_devices()
for d in pr.devices: d.use = (d.type == DEVICE)
scn.cycles.device = "GPU"; scn.cycles.samples = SPP; scn.cycles.use_denoising = True
scn.render.resolution_x, scn.render.resolution_y = RES; scn.render.resolution_percentage = 100
scn.render.use_persistent_data = True; scn.view_settings.view_transform = "Standard"
scn.render.image_settings.file_format = "PNG"; scn.render.image_settings.color_mode = "RGB"
bpy.ops.mesh.primitive_plane_add(size=60); ground = bpy.context.object
gm = bpy.data.materials.new("ground"); gm.use_nodes = True
gb = gm.node_tree.nodes["Principled BSDF"]; gb.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1); gb.inputs["Roughness"].default_value = 0.9
ground.data.materials.append(gm)
bpy.ops.object.camera_add(location=CAM_LOC); cam = bpy.context.object
cam.rotation_euler = (math.radians(63.6), 0, math.radians(46.7)); scn.camera = cam
LIGHTS = [((4.5, -4.5, 8.0), 500), ((-5.0, -3.0, 5.0), 130), ((-1.0, 6.0, 6.0), 200)]
for loc, e in LIGHTS:
    bpy.ops.object.light_add(type="AREA", location=loc); lt = bpy.context.object; lt.data.energy = e; lt.data.size = 3.0
    lt.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
LENS = LENS or cam.data.lens*320/240   # 원본 320×240 화면의 가운데 240×240과 같은 화각
cam.data.lens = LENS
bpy.context.view_layer.update()
_f = cam.data.lens / cam.data.sensor_width * max(RES)
PP = (RES[0]/2 - cam.data.shift_x*max(RES), RES[1]/2 + cam.data.shift_y*max(RES))   # 주점(픽셀, 왼쪽 위 원점)
W2C = np.array([list(r) for r in cam.matrix_world.inverted()])
CAM_META = {"location": list(CAM_LOC), "rotation_euler_deg": [63.6, 0.0, 46.7], "lens_mm": cam.data.lens, "sensor_width_mm": cam.data.sensor_width,
            "sensor_fit": cam.data.sensor_fit, "K_pixels": [[_f, 0, PP[0]], [0, _f, PP[1]], [0, 0, 1]],
            "world_to_camera": W2C.tolist(),
            "note": "카메라 좌표계는 Blender 규약(-Z가 시선 방향, +Y가 위). 화면 u,v는 0~1, 왼쪽 위 원점"}
LIGHT_META = [{"type": "AREA", "location": list(l), "energy": e, "size": 3.0} for l, e in LIGHTS]
RENDER_META = {"engine": "CYCLES", "samples": SPP, "denoise": True, "resolution": list(RES), "view_transform": "Standard", "blender": bpy.app.version_string}
scn.view_layers[0].use_pass_object_index = True
scn.use_nodes = True; nt = scn.node_tree
for n_ in list(nt.nodes): nt.nodes.remove(n_)
rl = nt.nodes.new("CompositorNodeRLayers"); comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
div = nt.nodes.new("CompositorNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = 255.0
fo = nt.nodes.new("CompositorNodeOutputFile"); fo.format.file_format = "PNG"; fo.format.color_mode = "BW"; fo.format.color_depth = "8"
nt.links.new(rl.outputs["IndexOB"], div.inputs[0]); nt.links.new(div.outputs[0], fo.inputs[0])

def make_material(name, rgb, metal):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Metallic"].default_value = 1.0 if metal else 0.0
    b.inputs["Roughness"].default_value = 0.2 if metal else 0.75; return m

def uv_of(p):
    v = world_to_camera_view(scn, cam, Vector(p)); return float(v.x), float(1 - v.y)

BOUND3 = {0: 1.0, 1: math.sqrt(3), 2: math.sqrt(2)}   # 외접구 반지름 / r

# ---------------- 실루엣으로 visible ratio 추정
def project_px(P):
    """월드 점 [..., 3] → (픽셀 u(오른쪽+), v(아래+), 깊이)"""
    c = P @ W2C[:3, :3].T + W2C[:3, 3]; dep = -c[..., 2]
    return PP[0] + _f*c[..., 0]/dep, PP[1] - _f*c[..., 1]/dep, dep
_t = np.array([0.7, -1.3, 0.5]); _u, _v, _ = project_px(_t); _uv = uv_of(_t)
assert abs(_u/RES[0] - _uv[0]) < 1e-4 and abs(_v/RES[1] - _uv[1]) < 1e-4, "투영식이 Blender 카메라와 다름"

_g = np.pi*(3 - math.sqrt(5)); _k = np.arange(160); _zz = 1 - 2*(_k + 0.5)/160; _rr = np.sqrt(1 - _zz**2)
UNIT_PTS = {0: np.stack([_rr*np.cos(_g*_k), _rr*np.sin(_g*_k), _zz], 1),                                         # 구: 표면 점
            1: np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float),                    # 정육면체: 꼭짓점
            2: np.concatenate([np.stack([np.cos(a), np.sin(a), np.full(64, z)], 1)
                               for a in [np.linspace(0, 2*np.pi, 64, endpoint=False)] for z in (-1, 1)])}           # 원기둥: 위아래 원

def hull(pts):
    """볼록 껍질 (monotone chain), 반시계 방향 꼭짓점 [m, 2]"""
    pts = sorted(map(tuple, pts))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= 0: h.pop()
            h.append(p)
        return h[:-1]
    return np.array(half(pts) + half(pts[::-1]))

PAD = max(RES)   # 화면 밖으로 나간 실루엣도 면적에 넣으려고 여백을 둔 캔버스에 칠함

def fill(poly):
    """볼록 다각형 안의 픽셀(중심 기준) → (행 범위, 열 범위, bool 배열). 범위는 여백 PAD를 더한 캔버스 좌표"""
    x0, x1 = max(int(math.floor(poly[:, 0].min())), -PAD), min(int(poly[:, 0].max()) + 1, RES[0] + PAD)
    y0, y1 = max(int(math.floor(poly[:, 1].min())), -PAD), min(int(poly[:, 1].max()) + 1, RES[1] + PAD)
    if x0 >= x1 or y0 >= y1: return None
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    ins = np.ones(X.shape, bool)
    for a, b in zip(poly, np.roll(poly, -1, 0)):   # 화면 좌표(v 아래+)에서 monotone chain 결과는 시계방향 → 부호 반대로 판정
        ins &= (b[0]-a[0])*(Y-a[1]) - (b[1]-a[1])*(X-a[0]) >= 0
    return slice(y0 + PAD, y1 + PAD), slice(x0 + PAD, x1 + PAD), ins

def visible_est(pos, r, sh, rz):
    """pos [NF, n, 3] → (추정 보이는 비율 [NF, n], 화면 안에 든 실루엣 비율 [NF, n]). 객체 중심 깊이 순서로 먼 것부터 칠함.
    보이는 비율은 화면 안 픽셀만 셈 (amodal 계산과 같은 기준), 화면 안 비율은 화면 밖으로 나간 부분까지 전체 실루엣 기준"""
    n = pos.shape[1]; vis = np.ones((NF, n)); inc = np.ones((NF, n))
    local = []
    for k in range(n):
        c, s = math.cos(math.radians(rz[k])), math.sin(math.radians(rz[k]))
        local.append(UNIT_PTS[int(sh[k])]*r[k] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T)
    for f in range(NF):
        lab = np.zeros((RES[1] + 2*PAD, RES[0] + 2*PAD), np.int16); amodal = np.zeros(n)
        _, _, dep = project_px(pos[f]); order = np.argsort(-dep)
        for k in order:
            u, v, _ = project_px(local[k] + pos[f, k]); fl = fill(hull(np.stack([u, v], 1)))
            if fl is None: inc[f, k] = 0; continue
            ys, xs, ins = fl; lab[ys, xs][ins] = k + 1
            Y, X = np.mgrid[ys, xs] - PAD
            amodal[k] = ((Y >= 0) & (Y < RES[1]) & (X >= 0) & (X < RES[0]))[ins].sum(); inc[f, k] = amodal[k]/max(ins.sum(), 1)
        cnt = np.bincount(lab[PAD:PAD + RES[1], PAD:PAD + RES[0]].ravel(), minlength=n + 1)[1:]
        vis[f] = np.where(amodal > 0, cnt/np.maximum(amodal, 1), 0)
    return vis, inc

# ---------------- 경로·배치
def closed_path(rng):
    """generate_closed_path: 반환 (경로 [NF, 2], 걸음 [NF-1, 2]), 월드 단위, 경로 평균 = 0"""
    for _ in range(MAX_TRIES):
        v = rng.uniform(-MAX_SHIFT, MAX_SHIFT, size=(WALK, 2))
        if REPEAT: v = np.repeat(v, 2, axis=0)
        v = np.concatenate([v, -v], axis=0)
        p = np.zeros((len(v) + 1, 2)); p[1:] = np.cumsum(v, axis=0)
        p -= p.mean(0)
        if all(p.max(0) - p.min(0) < 1 - 1.5*R_NORM):
            return p*WS, v*WS
    return None, None

def floor_point(px, py, z0=0.5):
    """화면 픽셀 (px, py)를 지나는 시선이 높이 z0 평면과 만나는 점 (x, y). 배치 중심으로 씀"""
    R = W2C[:3, :3]; d = R.T @ np.array([(px - PP[0])/_f, -(py - PP[1])/_f, -1.0]); o = np.array(CAM_LOC)
    t = (z0 - o[2])/d[2]; return (o + t*d)[:2]
CENTER = floor_point(RES[0]/2, RES[1]/2)

def place(rng, n, r, path):
    """traj_S1 배치: 객체마다 경로 평균 위치를 반경 DISK 원판 안(면적 균등)에서, 바닥 거리 > GAP·(r_i + r_j). 반환 pos [NF, n, 3]"""
    base = []
    for i in range(n):
        for _ in range(5000):
            a, rr = rng.uniform(0, 2*np.pi), DISK*math.sqrt(rng.uniform())
            p = CENTER + np.array([rr*math.cos(a), rr*math.sin(a)])
            if all(np.linalg.norm(p - q) > GAP*(r[i] + r[j]) for j, q in enumerate(base)): base.append(p); break
        else: return None
    pos = np.zeros((NF, n, 3))
    for i in range(n): pos[:, i, :2] = base[i] + path; pos[:, i, 2] = r[i]
    return pos

def collide(pos, r, bnd):
    n = pos.shape[1]
    return any(np.min(np.linalg.norm(pos[:, i] - pos[:, j], axis=1)) <= r[i]*bnd[i] + r[j]*bnd[j] for i in range(n) for j in range(i))

os.makedirs(OUT, exist_ok=True); t0 = time.time(); done = 0
for si in range(START, START + NUM):
    rng = np.random.default_rng(SEED * 1_000_003 + si)
    n = NOBJ if NOBJ > 0 else int(rng.choice(COUNTS, p=PROBS))
    for attempt in range(500):
        r = rng.uniform(RMIN, RMAX, n); sh = rng.integers(0, 3, n); rz = rng.uniform(0, 360, n); bnd = [BOUND3[int(x)] for x in sh]
        path, vel = closed_path(rng)
        if path is None: continue
        pos = place(rng, n, r, path)
        if pos is None or collide(pos, r, bnd): continue
        uv = np.array([[uv_of(pos[f, i]) for i in range(n)] for f in range(NF)])
        if (uv < 0.05).any() or (uv > 0.95).any(): continue
        vest, inc = visible_est(pos, r, sh, rz)
        if inc.min() < IN_FRAME - 1e-9: continue
        if vest.min() < VIS_MIN: continue
        if VIS_NEED > 0 and vest.min() >= VIS_NEED: continue
        break
    else:
        print(f"@@SKIP {si}"); continue
    if DRY:
        step = np.linalg.norm(np.diff(uv, axis=0)*[RES[0], RES[1]], axis=2)*(128/SIZE)   # 128 기준 px
        print(f"@@DRY S1L {si} n={n} attempts={attempt+1} step_px={step.mean():.2f} vis_est mean={vest.mean():.3f} min={vest.min():.3f}", flush=True)
        continue
    for o in [o for o in bpy.data.objects if o.pass_index > 0]: bpy.data.objects.remove(o, do_unlink=True)
    objs, specs = [], []
    for k in range(n):
        col = rng.uniform(0.1, 0.9, 3).tolist(); metal = int(rng.integers(2)); rzk = float(rz[k])
        if sh[k] == 0: bpy.ops.mesh.primitive_uv_sphere_add(radius=r[k], segments=48, ring_count=24); bpy.ops.object.shade_smooth()
        elif sh[k] == 1: bpy.ops.mesh.primitive_cube_add(size=2*r[k])
        else: bpy.ops.mesh.primitive_cylinder_add(radius=r[k], depth=2*r[k], vertices=64)
        o = bpy.context.object; o.pass_index = k+1; o.rotation_euler = (0, 0, math.radians(rzk))
        o.data.materials.append(make_material(f"obj{si}_{k}", col, metal == 1)); objs.append(o)
        specs.append({"radius": float(r[k]), "shape": int(sh[k]), "color": col, "material": metal, "rot_z": rzk})
        for f in range(NF): o.location = tuple(pos[f, k]); o.keyframe_insert("location", frame=f+1)
    scn.frame_start, scn.frame_end = 1, NF
    d = os.path.join(OUT, f"seq_{si:05d}"); os.makedirs(d, exist_ok=True)
    scn.render.filepath = os.path.join(d, "rgb_"); fo.base_path = d; fo.file_slots[0].path = "mask_"
    bpy.ops.render.render(animation=True)
    frames = [[{"xyz": pos[f, i].round(4).tolist(), "u": uv[f, i, 0], "v": uv[f, i, 1], "visible_est": round(float(vest[f, i]), 4)}
               for i in range(n)] for f in range(NF)]
    if AMODAL:
        scn.cycles.samples = 1; scn.cycles.use_denoising = False
        for l in [l for l in nt.links if l.to_node == comp]: nt.links.remove(l)
        for k, o in enumerate(objs, 1):
            for o2 in objs: o2.hide_render = (o2 is not o)
            fo.base_path = os.path.join(d, f"amodal_{k}"); scn.render.filepath = os.path.join(d, f"amodal_{k}", "_rgb_")
            bpy.ops.render.render(animation=True)
        for o2 in objs: o2.hide_render = False
        nt.links.new(rl.outputs["Image"], comp.inputs["Image"]); scn.cycles.samples = SPP; scn.cycles.use_denoising = True
        fo.base_path = d
        def load(p):
            im = bpy.data.images.load(p); im.colorspace_settings.name = "Non-Color"
            a = np.round(np.array(im.pixels[:]).reshape(RES[1], RES[0], 4)[..., 0]*255); bpy.data.images.remove(im); return a
        for f in range(NF):
            vis = load(os.path.join(d, f"mask_{f+1:04d}.png"))
            for k in range(1, n+1):
                am = (load(os.path.join(d, f"amodal_{k}", f"mask_{f+1:04d}.png")) == k).sum()
                frames[f][k-1]["visible_ratio"] = round(float((vis == k).sum() / am), 4) if am else None
        if not KEEP_AMODAL:
            for k in range(1, n+1): shutil.rmtree(os.path.join(d, f"amodal_{k}"), ignore_errors=True)
    json.dump({"scenario": "S1L", "hop": 0, "flat": 1, "seed": SEED, "sequence_index": si,
               "generator": {"scenario": "S1L", "script": "gen_loop_scenario.py", "seed": SEED, "flat": 1, "num_frames": NF,
                             "walk_len": WALK, "repeat_twice": REPEAT, "r_norm": R_NORM, "max_shift": MAX_SHIFT, "world_scale": WS,
                             "disk": DISK, "gap": GAP, "size": SIZE, "lens_mm": LENS, "in_frame": IN_FRAME, "center": CENTER.tolist(), "vis_min": VIS_MIN, "vis_need": VIS_NEED,
                             "object_counts": COUNTS.tolist(), "object_count_probs": PROBS.tolist(), "object_radius_range": [RMIN, RMAX]},
               "camera": CAM_META, "lights": LIGHT_META, "render": RENDER_META, "object_number": n, "attempts": attempt+1, "image_size": [RES[1], RES[0]],
               "shape_names": ["sphere", "cube", "cylinder"],
               "trajectory": {"path_xy": path.round(5).tolist(), "velocity_xy": vel.round(5).tolist(), "same_velocity": True},
               "objects": specs, "frames": frames},
              open(os.path.join(d, "meta.json"), "w"), indent=1, default=float)
    done += 1; print(f"@@SEQ S1L {si} n={n} attempts={attempt+1} {(time.time()-t0)/done:.1f}s/seq", flush=True)
print(f"@@DONE S1L {done}", flush=True)
