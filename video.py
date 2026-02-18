import argparse
import numpy as np
from pathlib import Path
import open3d as o3d
import glob
import os
import cv2

def load_kitti_velodyne_bin(bin_path: str):
    p = Path(bin_path)
    raw = np.fromfile(str(p), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Not KITTI velodyne bin (N,4). floats={raw.size}")
    pts = raw.reshape(-1, 4)
    xyz = pts[:, :3]
    intensity = pts[:, 3]
    return xyz, intensity

def color_for_class(name: str):
    name = name.lower()
    if name in ["car", "van", "truck"]:
        return (1.0, 0.0, 0.0)
    if name in ["pedestrian", "person_sitting"]:
        return (0.0, 1.0, 0.0)
    if name in ["cyclist"]:
        return (0.0, 0.6, 1.0)
    return (1.0, 0.8, 0.0)

def make_pcd(xyz, max_points=2000000, point_color=(0.2, 0.2, 0.2)):
    if max_points and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.paint_uniform_color(list(point_color))
    return pcd

def make_box_lines(center, size, yaw, color):
    R = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, float(yaw)))
    obb = o3d.geometry.OrientedBoundingBox(np.array(center, np.float64), R, np.array(size, np.float64))
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color(list(color))
    return ls

DEFAULT_ID2NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

def load_lidar_boxes_txt_pointpillars(txt_path: str, score_thr: float):
    p = Path(txt_path)
    boxes = []
    if not p.exists(): return boxes 

    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        parts = ln.split()
        if parts[0].lower() == "x": continue
        if len(parts) < 9: continue

        try:
            x, y, z, w, l, h, rt, cid, score = map(float, parts[:9])
        except: continue

        if score < score_thr: continue
        
        if not (0.1 < w < 20 and 0.1 < l < 30 and 0.1 < h < 10): continue

        dx, dy, dz = l, w, h
        yaw = rt
        cls_id = int(cid)
        cls_name = DEFAULT_ID2NAME.get(cls_id, f"cls_{cls_id}")
        boxes.append((x, y, z, dx, dy, dz, yaw, cls_name, float(score)))

    return boxes

def vis_crop_360(xyz, r_max=80.0, z_min=-5.0, z_max=5.0):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    r = np.sqrt(x*x + y*y)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (r < r_max) & (z > z_min) & (z < z_max) 
    return xyz[m]

class VideoGenerator:
    def __init__(self, bin_dir, txt_dir, score_thr=0.5, output_filename="output_lidar.mp4"):
        self.bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        self.txt_dir = txt_dir
        self.score_thr = score_thr
        self.total = len(self.bin_files)
        self.output_filename = output_filename
        
        self.width = 1280
        self.height = 720
        self.fps = 3.0

        if self.total == 0:
            print(f"Error: {bin_dir} 裡面沒有 .bin 檔")
            return

        print(f"準備處理 {self.total} 幀，輸出至 {self.output_filename}")
        self.run_render()

    def run_render(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window("Rendering", width=self.width, height=self.height, visible=True)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.point_size = 1.0
        opt.line_width = 5.0 

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_filename, fourcc, self.fps, (self.width, self.height))

        geometry_list = []

        for idx, bin_path in enumerate(self.bin_files):
            for g in geometry_list:
                vis.remove_geometry(g, reset_bounding_box=False)
            geometry_list = []

            filename = os.path.basename(bin_path).replace('.bin', '')
            txt_path = os.path.join(self.txt_dir, filename + ".txt")

            print(f"[{idx+1}/{self.total}] Processing {filename} ...", end="\r")

            xyz, intensity = load_kitti_velodyne_bin(bin_path)
            xyz = vis_crop_360(xyz)
            pcd = make_pcd(xyz)
            
            reset_view = (idx == 0)
            vis.add_geometry(pcd, reset_bounding_box=reset_view)
            geometry_list.append(pcd)

            boxes = load_lidar_boxes_txt_pointpillars(txt_path, self.score_thr)
            for (x, y, z, dx, dy, dz, yaw, cls_name, score) in boxes:
                ls = make_box_lines((x, y, z), (dx, dy, dz), yaw, color_for_class(cls_name))
                vis.add_geometry(ls, reset_bounding_box=False)
                geometry_list.append(ls)

            if idx == 0:
                ctr = vis.get_view_control()
                # ctr.set_lookat([0, 0, 0])
                # ctr.set_front([0, 0.01, 1])
                # ctr.set_up([0, 1, 0])
                # ctr.set_zoom(0.4)
                
                ctr.set_lookat([0, 0, 0])
                ctr.set_front([-1.0, 0.0, 0.5]) 
                ctr.set_up([0, 0, 1])           
                ctr.set_zoom(0.3)

                vis.poll_events()
                vis.update_renderer()
            

            vis.poll_events()
            vis.update_renderer()


            image = vis.capture_screen_float_buffer(False)
            image = (np.asarray(image) * 255).astype(np.uint8)
            

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            video_writer.write(image)

        vis.destroy_window()
        video_writer.release()
        print(f"\nDone! Video saved to {self.output_filename}")

if __name__ == "__main__":

    BIN_DIR = "lidar"         
    TXT_DIR = "result0.2"     
    SCORE = 0.4               
    OUTPUT_VIDEO = "demo_result_0.4.mp4"

    VideoGenerator(BIN_DIR, TXT_DIR, SCORE, OUTPUT_VIDEO)