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
        return (1.0, 0.0, 0.0)      # red
    if name in ["pedestrian", "person_sitting"]:
        return (0.0, 1.0, 0.0)      # green
    if name in ["cyclist"]:
        return (0.0, 0.6, 1.0)      # cyan
    return (1.0, 0.8, 0.0)          # orange

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
    if not p.exists(): return boxes # 防止檔案不存在報錯

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
        
        # 你的簡單過濾
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

# ==========================================
#  這裡改成了 Class 來支援上一張/下一張
# ==========================================

class Player:
    def __init__(self, bin_dir, txt_dir, score_thr=0.5):
        self.bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        self.txt_dir = txt_dir
        self.score_thr = score_thr
        self.idx = 0
        self.total = len(self.bin_files)
        
        if self.total == 0:
            print(f"Error: {bin_dir} 裡面沒有 .bin 檔")
            return

        print(f"載入 {self.total} 幀。按 '右鍵' 下一張，'左鍵' 上一張。")

        # 使用支援按鍵的 Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("User Style Viewer", width=1280, height=720)
        
        # 設定背景為白色 (你的偏好)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1]) 
        opt.point_size = 1.0

        # 按鍵綁定
        self.vis.register_key_callback(262, self.next_frame) # Right Arrow
        self.vis.register_key_callback(263, self.prev_frame) # Left Arrow

        self.geometry_list = []
        self.update_frame(reset_view=True)  # 顯示第一幀
        
        self.vis.run()
        self.vis.destroy_window()

    def update_frame(self, reset_view=False):
        # 1. 清除上一幀的物件
        for g in self.geometry_list:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.geometry_list = []

        # 2. 讀取資料
        bin_path = self.bin_files[self.idx]
        filename = os.path.basename(bin_path).replace('.bin', '')
        txt_path = os.path.join(self.txt_dir, filename + ".txt")

        # 在終端機顯示當前狀態
        print(f"\n{'='*50}")
        print(f"Frame: {self.idx+1}/{self.total} | Score閾值: ≥{self.score_thr:.2f}")
        print(f"檔案: {filename}")
        print(f"{'='*50}")

        # 3. 建立點雲 (你的邏輯)
        xyz, intensity = load_kitti_velodyne_bin(bin_path)
        xyz = vis_crop_360(xyz)
        
        pcd = make_pcd(xyz)
        self.vis.add_geometry(pcd, reset_bounding_box=reset_view)
        self.geometry_list.append(pcd)

        # 4. 建立框框
        boxes = load_lidar_boxes_txt_pointpillars(txt_path, self.score_thr)
        print(f"偵測到 {len(boxes)} 個物件")
        
        for (x, y, z, dx, dy, dz, yaw, cls_name, score) in boxes:
            ls = make_box_lines((x, y, z), (dx, dy, dz), yaw, color_for_class(cls_name))
            self.vis.add_geometry(ls)
            self.geometry_list.append(ls)

        # 5. 設定相機視角（只在第一幀時）
        if reset_view:
            ctr = self.vis.get_view_control()
            ctr.set_lookat([0, 0, 0])       # 看向原點
            ctr.set_front([0, -0.3, -0.95]) # 相機方向（往下看）
            ctr.set_up([0, -1, 0])          # Y軸向下為上方
            ctr.set_zoom(0.3)               # 縮放程度

    def next_frame(self, vis):
        if self.idx < self.total - 1:
            self.idx += 1
            self.update_frame()
    
    def prev_frame(self, vis):
        if self.idx > 0:
            self.idx -= 1
            self.update_frame()

if __name__ == "__main__":
    BIN_DIR = "lidar" 
    TXT_DIR = "result0.2"   
    SCORE = 0.3

    Player(BIN_DIR, TXT_DIR, SCORE)