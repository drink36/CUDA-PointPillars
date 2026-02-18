import argparse
import numpy as np
from pathlib import Path
import open3d as o3d


# ----------------------------
# KITTI: velodyne bin (N,4) float32: x y z intensity
# ----------------------------
def load_kitti_velodyne_bin(bin_path: str):
    p = Path(bin_path)
    raw = np.fromfile(str(p), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Not KITTI velodyne bin (N,4). floats={raw.size}")
    pts = raw.reshape(-1, 4)
    xyz = pts[:, :3]
    intensity = pts[:, 3]
    return xyz, intensity


# ----------------------------
# Simple colors
# ----------------------------
def color_for_class(name: str):
    name = name.lower()
    if name in ["car", "van", "truck"]:
        return (1.0, 0.0, 0.0)      # red
    if name in ["pedestrian", "person_sitting"]:
        return (0.0, 1.0, 0.0)      # green
    if name in ["cyclist"]:
        return (0.0, 0.6, 1.0)      # cyan
    return (1.0, 0.8, 0.0)          # orange


# ----------------------------
# Open3D helpers
# ----------------------------
def make_pcd(xyz, max_points=200000, point_color=(0.2, 0.2, 0.2)):
    if max_points and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.paint_uniform_color(list(point_color))
    return pcd


def make_box_lines(center, size, yaw, color):
    # yaw about +Z in lidar frame
    R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, float(yaw)))
    obb = o3d.geometry.OrientedBoundingBox(np.array(center, np.float64), R, np.array(size, np.float64))
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color(list(color))
    return ls


# ----------------------------
# Case A: Lidar-box txt (already in lidar coords)
# Expect at least: x y z dx dy dz yaw score class_id OR class_name
# We'll accept:
# - last token numeric => class_id (0/1/2 -> Car/Ped/Cyc default)
# - last token str => class name
# ----------------------------
DEFAULT_ID2NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

from pathlib import Path

DEFAULT_ID2NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

def load_lidar_boxes_txt_pointpillars(txt_path: str, score_thr: float):
    """
    Expect each line:
      x y z w l h rt id score
    Convert to:
      x y z dx dy dz yaw cls_name score
    where dx=l, dy=w, dz=h.
    """
    p = Path(txt_path)
    boxes = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if parts[0].lower() == "x":
            continue
        if len(parts) < 9:
            continue

        x, y, z, w, l, h, rt, cid, score = map(float, parts[:9])

        # score filter
        if score < score_thr:
            continue

        # sanity (w/l/h in meters)
        if not (0.1 < w < 20 and 0.1 < l < 30 and 0.1 < h < 10):
            continue

        # Convert to Open3D box size convention you used: (dx, dy, dz)
        dx = l
        dy = w
        dz = h
        yaw = rt

        cls_id = int(cid)
        cls_name = DEFAULT_ID2NAME.get(cls_id, f"cls_{cls_id}")

        boxes.append((x, y, z, dx, dy, dz, yaw, cls_name, float(score)))

    return boxes



# ----------------------------
# Case B: KITTI label_2 (camera coords) + calib
# We convert 3D boxes from camera to lidar
# label_2 format:
# type trunc occl alpha bbox2d(4) h w l x y z ry
# (h,w,l in meters, x,y,z in camera coords, ry rotation around Y in camera)
# ----------------------------
def read_calib(calib_path: str):
    p = Path(calib_path)
    data = {}
    for line in p.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        vals = np.array([float(x) for x in v.strip().split()], dtype=np.float64)
        data[k.strip()] = vals
    # KITTI calib essentials
    # Tr_velo_to_cam: 3x4
    Tr = data.get("Tr_velo_to_cam", None)
    R0 = data.get("R0_rect", None)
    if Tr is None or R0 is None:
        raise ValueError("calib must contain Tr_velo_to_cam and R0_rect")
    Tr = Tr.reshape(3, 4)
    R0 = R0.reshape(3, 3)

    Tr4 = np.eye(4)
    Tr4[:3, :4] = Tr
    R04 = np.eye(4)
    R04[:3, :3] = R0

    # camera_rect = R0_rect * (Tr_velo_to_cam * velo)
    # => velo = inv(Tr_velo_to_cam) * inv(R0_rect) * camera_rect
    V2C = Tr4
    R0rect = R04
    C2V = np.linalg.inv(V2C)
    R0inv = np.linalg.inv(R0rect)
    return C2V, R0inv


def load_kitti_label2(label_path: str):
    p = Path(label_path)
    objs = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        t = parts[0]
        if t.lower() == "dontcare":
            continue
        # h w l x y z ry (camera)
        h = float(parts[8]); w = float(parts[9]); l = float(parts[10])
        x = float(parts[11]); y = float(parts[12]); z = float(parts[13])
        ry = float(parts[14])
        objs.append((t, h, w, l, x, y, z, ry))
    return objs


def camera_box_to_lidar(center_cam, hwl_cam, ry_cam, C2V, R0inv):
    # center_cam in rect camera coords (x,y,z)
    # Convert to velo: velo = C2V * R0inv * [x,y,z,1]
    pt = np.array([center_cam[0], center_cam[1], center_cam[2], 1.0], dtype=np.float64)
    pt = (R0inv @ pt)
    pt = (C2V @ pt)
    center_velo = pt[:3]

    # KITTI convention:
    # camera: x right, y down, z forward
    # lidar (velo): x forward, y left, z up
    # ry is rotation around camera Y (down axis).
    # A common conversion for yaw in lidar is: yaw_lidar = -(ry_cam + pi/2)
    yaw_lidar = -(ry_cam + np.pi / 2.0)

    # sizes in lidar: dx, dy, dz correspond to l, w, h in lidar frame
    # We'll use (dx=l, dy=w, dz=h)
    dx, dy, dz = hwl_cam[2], hwl_cam[1], hwl_cam[0]  # l,w,h
    return center_velo, (dx, dy, dz), yaw_lidar

def vis_crop_360(xyz, r_max=80.0, z_min=-5.0, z_max=5.0):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    r = np.sqrt(x*x + y*y)
    # vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);
    # vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);
    # vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
    # crop front only, and also remove points that are too high/low or too far (likely noise)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (r < r_max) & (z > z_min) & (z < z_max) 
    
    return xyz[m]

# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="KITTI velodyne .bin (N,4 float32)")
    ap.add_argument("--max_points", type=int, default=2000000)
    ap.add_argument("--point_size", type=float, default=1.0)
    ap.add_argument("--front_only", action="store_true", help="keep x>0 only")
    ap.add_argument("--score_thr", type=float, default=0.5)

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--label2", help="KITTI label_2 .txt (camera coords)")
    mode.add_argument("--lidar_txt", help="lidar box txt (x y z dx dy dz yaw ... class)")

    ap.add_argument("--calib", help="KITTI calib .txt (required if using --label2)")
    args = ap.parse_args()

    xyz, intensity = load_kitti_velodyne_bin(args.bin)
    print(intensity[:10])
    if args.front_only:
        xyz = xyz[xyz[:, 0] > 0]
    xyz = vis_crop_360(xyz, r_max=80.0, z_min=-5.0, z_max=5.0)
    
    pcd = make_pcd(xyz, args.max_points, point_color=(0.15, 0.15, 0.15))
    geoms = [pcd]
    pc_range = np.array([0, -39.68, -3, 69.12, 39.68, 1], dtype=np.float32)
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    mask = (x>=pc_range[0])&(x<=pc_range[3])&(y>=pc_range[1])&(y<=pc_range[4])&(z>=pc_range[2])&(z<=pc_range[5])
    print("points before:", len(xyz), "after range:", mask.sum())

    if args.label2:
        if not args.calib:
            raise ValueError("--calib is required when using --label2")
        C2V, R0inv = read_calib(args.calib)
        objs = load_kitti_label2(args.label2)
        for (t, h, w, l, x, y, z, ry) in objs:
            center_velo, size_velo, yaw = camera_box_to_lidar((x, y, z), (h, w, l), ry, C2V, R0inv)
            geoms.append(make_box_lines(center_velo, size_velo, yaw, color_for_class(t)))
    else:
        boxes = load_lidar_boxes_txt_pointpillars(args.lidar_txt, args.score_thr)
        for (x, y, z, dx, dy, dz, yaw, cls_name, score) in boxes:
            geoms.append(make_box_lines((x, y, z), (dx, dy, dz), yaw, color_for_class(cls_name)))

    vis = o3d.visualization.Visualizer()
    vis.create_window("KITTI viz", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
    vis.add_geometry(axis)

    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_front([0.0, 0.0, 1.0])  # 從上往下看
    ctr.set_up([1.0, 0.0, 0.0])      # 固定畫面不要亂翻
    ctr.set_zoom(0.25)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
