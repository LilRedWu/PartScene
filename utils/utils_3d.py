import numpy as np 
import torch
from typing import Tuple, Dict, List
from collections import defaultdict
import os
import open3d as o3d
NUM_PROMPTS = 1024


def load_instance_info(instance_info_dir):
    """
    Load instance information and point cloud data from the given directory.
    
    Args:
    instance_info_dir (str): Path to the directory containing the saved files.
    
    Returns:
    dict: A dictionary containing the loaded data.
    """
    # Load PyTorch tensors
    top_k_masks = torch.load(os.path.join(instance_info_dir, 'top_k_masks.pt'))
    pc_depth = torch.load(os.path.join(instance_info_dir, 'pc_depth.pt'))
    screen_coords = torch.load(os.path.join(instance_info_dir, 'screen_coords.pt'))
    
    # Load numpy array
    obj_xyz = np.load(os.path.join(instance_info_dir, 'obj_xyz.npy'))
    
    # Create a dictionary to hold all the loaded data
    
    
    
    return top_k_masks,pc_depth,screen_coords,obj_xyz

def load_prediction_data(filename):
    with open(filename, 'r') as file:
        return [{'file': parts[0], 'prediction': int(parts[1]), 'confidence': float(parts[2])}
                for line in file if len(parts := line.strip().split()) == 3]
    
def normalize_point_cloud(xyz):
    """
    Normalize the point cloud by centering it and scaling it to fit within a unit sphere.
    :param xyz: The point cloud coordinates of shape (n_points, 3).
    :return: The normalized point cloud coordinates.
    """
    # Center the points by subtracting the mean
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid

    # Scale the points to fit within a unit sphere
    furthest_distance = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
    xyz_normalized = xyz_centered / furthest_distance

    return xyz_normalized

# def binaryMaskIOU(mask1, mask2):
#     if isinstance(mask1, np.ndarray):
#         mask1 = torch.from_numpy(mask1)
#     if isinstance(mask2, np.ndarray):
#         mask2 = torch.from_numpy(mask2)
    
#     mask1_area = torch.count_nonzero(mask1)
#     mask2_area = torch.count_nonzero(mask2)
#     intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
#     union = mask1_area + mask2_area - intersection
#     iou = intersection.float() / union.float() if union > 0 else torch.tensor(0.0)
#     return iou.item()

def binaryMaskIOU(mask1, mask2):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    if isinstance(mask1, np.ndarray):
        mask1 = torch.from_numpy(mask1).to(device)
    if isinstance(mask2, np.ndarray):
        mask2 = torch.from_numpy(mask2).to(device)

    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union


def process_masks_and_calculate_iou(
    class_results: Dict, 
    num_views: int, 
    binary_masks_list: List[np.ndarray], 
    iou_threshold: float = 0, 
    normalized_threshold: float = 0.2
) -> Dict:
    target_3d_masks = {class_name: {view_idx: [] for view_idx in range(num_views)} for class_name in class_results.keys()}

    for view_idx in range(num_views):
        binary_masks = binary_masks_list[view_idx]

        for class_name, class_data in class_results.items():
            if view_idx < len(class_data['masks']):
                sam_masks = class_data['masks'][view_idx]
                view_ious = []

                for mask_idx in range(len(binary_masks)):
                    for sam_mask_idx, sam_mask in enumerate(sam_masks):
                        # Calculate IoU for masks
                        mask_iou = binaryMaskIOU(binary_masks[mask_idx], sam_mask)
                        if mask_iou > 0:
                            view_ious.append((mask_idx, mask_iou))

                # Normalize IoU scores for this view and class using sum-based method
                if view_ious:
                    total_iou = sum(iou for _, iou in view_ious)
                    
                    if total_iou > normalized_threshold:
                        normalized_ious = [(mask_idx, iou / total_iou) for mask_idx, iou in view_ious]
                        
                        # Filter normalized IoUs based on the normalized threshold
                        filtered_ious = [(mask_idx, score) for mask_idx, score in normalized_ious if score >= normalized_threshold]
                        
                        target_3d_masks[class_name][view_idx] = filtered_ious

    return target_3d_masks

 
    
def assign_labels_to_masks(class_results, target_3d_masks, num_views, N=1):
        # Dictionary to store mIoU for each mask-label combination
        mask_label_miou = defaultdict(lambda: defaultdict(float))

        # Dictionary to store how many views each 3D mask appears in for each label
        mask_occurrences = defaultdict(lambda: defaultdict(int))

        # Calculate mIoU and count occurrences
        for class_name, view_data in target_3d_masks.items():
            for view_idx, mask_list_for_view in view_data.items():
                for mask_idx, iou in mask_list_for_view:
                    mask_label_miou[mask_idx][class_name] += iou
                    mask_occurrences[mask_idx][class_name] += 1

        # Calculate average IoU (mIoU) for each mask-label combination
        for mask_idx in mask_label_miou:
            for class_name in mask_label_miou[mask_idx]:
                mask_label_miou[mask_idx][class_name] /= mask_occurrences[mask_idx][class_name]

        # Assign labels based on highest mIoU and N-view threshold
        final_predictions = {}
        for mask_idx in mask_label_miou:
            best_label = max(mask_label_miou[mask_idx], key=mask_label_miou[mask_idx].get)
            if mask_occurrences[mask_idx][best_label] >= N:
                final_predictions[mask_idx] = {
                    'label': best_label,
                    'miou': mask_label_miou[mask_idx][best_label],
                    'views': mask_occurrences[mask_idx][best_label]
                }

        print(f"Final predictions:")
        # for mask_idx, prediction in final_predictions.items():
        #     print(f"Mask {mask_idx}: Label={prediction['label']}, mIoU={prediction['miou']:.4f}, Views={prediction['views']}")

        return final_predictions

# def binaryMaskIOU(mask1, mask2):
#     if isinstance(mask1, np.ndarray):
#         mask1 = torch.from_numpy(mask1)
#     if isinstance(mask2, np.ndarray):
#         mask2 = torch.from_numpy(mask2)
    
#     mask1_area = torch.count_nonzero(mask1)
#     mask2_area = torch.count_nonzero(mask2)
#     intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
#     union = mask1_area + mask2_area - intersection
#     iou = intersection.float() / union.float() if union > 0 else torch.tensor(0.0)
#     return iou.item()


def process_masks_and_calculate_iou(
    class_results: Dict, 
    num_views: int, 
    binary_masks_list: List[np.ndarray], 
    iou_threshold: float = 0, 
    normalized_threshold: float = 0.2
) -> Dict:
    target_3d_masks = {class_name: {view_idx: [] for view_idx in range(num_views)} for class_name in class_results.keys()}

    for view_idx in range(num_views):
        binary_masks = binary_masks_list[view_idx]

        for class_name, class_data in class_results.items():
            if view_idx < len(class_data['masks']):
                sam_masks = class_data['masks'][view_idx]
                view_ious = []

                for mask_idx in range(len(binary_masks)):
                    for sam_mask_idx, sam_mask in enumerate(sam_masks):
                        # Calculate IoU for masks
                        mask_iou = binaryMaskIOU(binary_masks[mask_idx], sam_mask)
                        if mask_iou > 0:
                            view_ious.append((mask_idx, mask_iou))

                # Normalize IoU scores for this view and class using sum-based method
                if view_ious:
                    total_iou = sum(iou for _, iou in view_ious)
                    
                    normalized_ious = [(mask_idx, iou / total_iou) for mask_idx, iou in view_ious]
                        
                    # Filter normalized IoUs based on the normalized threshold
                    filtered_ious = [(mask_idx, score) for mask_idx, score in normalized_ious if score >= normalized_threshold]
                    
                    target_3d_masks[class_name][view_idx] = filtered_ious

    return target_3d_masks

def process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos):
    pcd = o3d.io.read_point_cloud(scene_path)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors) * 255

#     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))

# # Rotate point cloud

# # Rotate mesh
#     pcd = pcd.rotate(rotation_matrix, center=(0, 0, 0)) 
    
    for idx, mask_info in enumerate(mask_infos):
        mask = np.loadtxt(os.path.join(os.path.dirname(mask_info_path), mask_info['file'])).astype(bool)
        obj_xyz = normalize_point_cloud(xyz[mask])
        obj_rgb = rgb[mask]
        
        obj_xyz_tensor = torch.tensor(obj_xyz).to(device).float()
        obj_rgb_tensor = torch.tensor(obj_rgb).to(device).float()

         
        obj_pcd = Pointclouds(points=[obj_xyz_tensor], features=[obj_rgb_tensor])
        obj_xyz_tensor = obj_xyz_tensor.unsqueeze(0)
        obj_rgb_tensor = obj_rgb_tensor.unsqueeze(0)
        top_k_masks, _, _ = mask_proposal(obj_xyz_tensor, obj_rgb_tensor, NUM_PROMPTS, model)
        #instance_pcd
        img_dir, pc_depth, screen_coords, num_views, cameras = render_all_angles_pc(obj_pcd, os.path.join(output_dir, str(idx)), device)
        # save top_k_masks,pc_depth,screen_coords as pt
        # save obj_xyz as np
        # make a new directoy under the os.path.join(output_dir, str(idx)) called ins_info
        instance_info_dir = os.path.join(output_dir, str(idx), 'ins_info')
        os.makedirs(instance_info_dir, exist_ok=True)

        # Save top_k_masks, pc_depth, and screen_coords as pt files
        torch.save(top_k_masks, os.path.join(instance_info_dir, 'top_k_masks.pt'))
        torch.save(pc_depth, os.path.join(instance_info_dir, 'pc_depth.pt'))
        torch.save(screen_coords, os.path.join(instance_info_dir, 'screen_coords.pt'))

        # Save obj_xyz as numpy array
        np.save(os.path.join(instance_info_dir, 'obj_xyz.npy'), obj_xyz)

        print(f"Saved instance information and point cloud data to {instance_info_dir}")
    return top_k_masks, img_dir, pc_depth, screen_coords, num_views, cameras,obj_xyz


def project_3d_to_2d(obj_xyz: np.ndarray, 
                     top_k_masks: np.ndarray, 
                     screen_coords: np.ndarray, 
                     pc_depth: np.ndarray, 
                     img_size: Tuple[int, int] = (1024, 1024)) -> Tuple[List[np.ndarray], Dict[int, Dict[int, Tuple[float, float, float, float]]]]:
    width, height = img_size
    num_views = screen_coords.shape[0]
    
    if isinstance(top_k_masks, torch.Tensor):
        top_k_masks = top_k_masks.cpu().numpy()
    
    num_masks = top_k_masks.shape[0]
    colors = np.random.randint(0, 255, size=(num_masks, 3))
    
    mask_2d_bbox_correspondences = {view_idx: {} for view_idx in range(num_views)}
    mask2d_view_list = []

    for view_idx in range(num_views):
        depth_buffer = np.full((height, width), np.inf)
        mask_2d_color_view = np.zeros((height, width, 3), dtype=np.uint8)
        
        view_screen_coords = screen_coords[view_idx]
        view_pc_depth = pc_depth[view_idx]
        
        for mask_idx in range(num_masks):
            mask_points = obj_xyz[top_k_masks[mask_idx]]
            mask_screen_coords = view_screen_coords[top_k_masks[mask_idx]]
            mask_depth = np.linalg.norm(mask_points, axis=1)
            
            valid_points = (mask_screen_coords[:, 0] >= 0) & (mask_screen_coords[:, 0] < width) & \
                           (mask_screen_coords[:, 1] >= 0) & (mask_screen_coords[:, 1] < height)
            
            valid_screen_coords = mask_screen_coords[valid_points].astype(int)
            valid_depths = mask_depth[valid_points]
            
            y, x = valid_screen_coords[:, 1], valid_screen_coords[:, 0]
            current_depths = depth_buffer[y, x]
            update_mask = valid_depths < current_depths
            
            if np.any(update_mask):
                update_y, update_x = y[update_mask], x[update_mask]
                depth_buffer[update_y, update_x] = valid_depths[update_mask]
                mask_2d_color_view[update_y, update_x] = colors[mask_idx]
                
                min_x, min_y = np.min(valid_screen_coords[update_mask], axis=0)
                max_x, max_y = np.max(valid_screen_coords[update_mask], axis=0)
                mask_2d_bbox_correspondences[view_idx][mask_idx] = (min_x, min_y, max_x, max_y)
        
        mask2d_view_list.append(mask_2d_color_view)
    
    return mask2d_view_list, mask_2d_bbox_correspondences


def project_3d_to_2d(obj_xyz: np.ndarray, 
                     top_k_masks: np.ndarray, 
                     screen_coords: np.ndarray, 
                     pc_depth: np.ndarray, 
                     img_size: Tuple[int, int] = (1024, 1024)) -> Tuple[List[np.ndarray], Dict[int, Dict[int, Tuple[float, float, float, float]]], List[np.ndarray]]:
    width, height = img_size
    num_views = screen_coords.shape[0]
    
    if isinstance(top_k_masks, torch.Tensor):
        top_k_masks = top_k_masks.cpu().numpy()
    
    num_masks = 100  # Fixed number of masks
    colors = np.random.randint(0, 255, size=(num_masks, 3))
    
    mask_2d_bbox_correspondences = {view_idx: {} for view_idx in range(num_views)}
    mask2d_view_list = []
    binary_masks_list = []

    for view_idx in range(num_views):
        depth_buffer = np.full((height, width), np.inf)
        mask_2d_color_view = np.zeros((height, width, 3), dtype=np.uint8)
        binary_masks = np.zeros((num_masks, height, width), dtype=bool)
        
        view_screen_coords = screen_coords[view_idx]
        view_pc_depth = pc_depth[view_idx]
        
        for mask_idx in range(num_masks):
            if mask_idx < len(top_k_masks):
                mask_points = obj_xyz[top_k_masks[mask_idx]]
                mask_screen_coords = view_screen_coords[top_k_masks[mask_idx]]
                mask_depth = np.linalg.norm(mask_points, axis=1)
                
                valid_points = (mask_screen_coords[:, 0] >= 0) & (mask_screen_coords[:, 0] < width) & \
                               (mask_screen_coords[:, 1] >= 0) & (mask_screen_coords[:, 1] < height)
                
                valid_screen_coords = mask_screen_coords[valid_points].astype(int)
                valid_depths = mask_depth[valid_points]
                
                y, x = valid_screen_coords[:, 1], valid_screen_coords[:, 0]
                current_depths = depth_buffer[y, x]
                update_mask = valid_depths < current_depths
                
                if np.any(update_mask):
                    update_y, update_x = y[update_mask], x[update_mask]
                    depth_buffer[update_y, update_x] = valid_depths[update_mask]
                    mask_2d_color_view[update_y, update_x] = colors[mask_idx]
                    binary_masks[mask_idx, update_y, update_x] = True
                    
                    min_x, min_y = np.min(valid_screen_coords[update_mask], axis=0)
                    max_x, max_y = np.max(valid_screen_coords[update_mask], axis=0)
                    mask_2d_bbox_correspondences[view_idx][mask_idx] = (min_x, min_y, max_x, max_y)
        
        mask2d_view_list.append(mask_2d_color_view)
        binary_masks_list.append(binary_masks)
    
    return mask2d_view_list, mask_2d_bbox_correspondences, binary_masks_list

