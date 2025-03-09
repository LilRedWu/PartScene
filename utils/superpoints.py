import numpy as np
from sklearn.cluster import DBSCAN

def generate_superpoints(point_cloud, eps=0.1, min_samples=10):
    coords = point_cloud[:, :3]  # assuming point_cloud is Nx6 (xyzrgb)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    num_superpoints = labels.max() + 1
    superpoints = np.zeros((len(point_cloud), num_superpoints), dtype=int)
    for i, label in enumerate(labels):
        if label != -1:  # ignore noise points
            superpoints[i, label] = 1
    
    return superpoints

import numpy as np

def calculate_view_weights(camera_pos, mask_pos, grid_size=3):
    grid_x, grid_y = int(mask_pos[0] // grid_size), int(mask_pos[1] // grid_size)
    cam_x, cam_y = int(camera_pos[0] // grid_size), int(camera_pos[1] // grid_size)
    
    if grid_x == cam_x and grid_y == cam_y:
        return 3
    elif abs(grid_x - cam_x) <= 1 and abs(grid_y - cam_y) <= 1:
        return 2
    return 1

import numpy as np
import open3d as o3d

def calculate_view_weights(camera_pos, mask_center, scene_bounds, grid_size=3):
    x_min, x_max, y_min, y_max, _, _ = scene_bounds
    grid_width = (x_max - x_min) / grid_size
    grid_height = (y_max - y_min) / grid_size
    
    cam_x, cam_y = int((camera_pos[0] - x_min) / grid_width), int((camera_pos[1] - y_min) / grid_height)
    mask_x, mask_y = int((mask_center[0] - x_min) / grid_width), int((mask_center[1] - y_min) / grid_height)
    
    dist = max(abs(cam_x - mask_x), abs(cam_y - mask_y))
    if dist == 0:
        return 3
    elif dist == 1:
        return 2
    return 1

def project_points(point_cloud, camera_pos, orientation, intrinsics):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.cross([0, 0, 1], orientation))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = camera_pos
    
    points = point_cloud[:, :3]
    points_cam = (R @ points.T + T[:3, 3:4]).T
    depth = points_cam[:, 2]
    uv = (intrinsics.intrinsic_matrix @ points_cam.T).T
    uv = uv[:, :2] / (uv[:, 2:3] + 1e-6)
    return uv, depth

def group_3d_parts(superpoints, point_cloud, views, masks, camera_positions, orientations, scene_bounds):
    intrinsics = o3d.camera.PinholeCameraIntrinsic(256, 256, 500, 500, 128, 128)
    scores = np.zeros(superpoints.shape[1])
    visibility = np.zeros((len(point_cloud), len(views)))
    weights = []
    
    for v, (view, mask, pos, ori) in enumerate(zip(views, masks, camera_positions, orientations)):
        uv, depth = project_points(point_cloud, pos, ori, intrinsics)
        mask_center = np.mean(np.argwhere(mask > 0.5), axis=0) if np.any(mask > 0.5) else np.array([128, 128])
        weight = calculate_view_weights(pos, mask_center[::-1], scene_bounds)
        weights.append(weight)
        
        for i, (u, d) in enumerate(zip(uv, depth)):
            if d > 0 and 0 <= u[0] < 256 and 0 <= u[1] < 256:
                visibility[i, v] = 1
                if mask[int(u[1]), int(u[0])] > 0.5:
                    scores += superpoints[i] * weight
    
    denom = visibility @ np.array(weights) + 1e-6
    scores = scores / denom
    
    foreground = scores > 0.5
    part_masks = superpoints[:, foreground]
    return part_masks