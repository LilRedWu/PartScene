import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
from point_sam.build_model import build_point_sam
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pytorch3d.ops as ops
from pytorch3d.ops import sample_farthest_points
from utils.nms import apply_pointwise_nms,visualize_point_clouds_with_masks
from mask_proposal import  mask_proposal,mask_proposal_v2,batch_mask_proposal
from utils.render import render_all_angles_pc,render_single_view,project_3d_to_2d
import glob
from point_sam.build_model import build_point_sam
import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
# Use glob to access all files in the directory
import os
from utils.inference_florence import run_florence2
from PIL import Image
import cv2
import supervision as sv

import open3d as o3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def load_prediction_data(filename):
#     with open(filename, 'r') as file:
#         return [{'file': parts[0], 'prediction': int(parts[1]), 'confidence': float(parts[2])}
#                 for line in file if len(parts := line.strip().split()) == 3]

# def normalize_point_cloud(xyz):
#     centroid = np.mean(xyz, axis=0)
#     xyz_centered = xyz - centroid
#     furthest_distance = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
#     return xyz_centered / furthest_distance

# def process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos):
#     pcd = o3d.io.read_point_cloud(scene_path)
#     xyz = np.asarray(pcd.points)
#     rgb = np.asarray(pcd.colors) * 255

#     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))

# # Rotate point cloud

# # Rotate mesh
#     pcd = pcd.rotate(rotation_matrix, center=(0, 0, 0)) 
    
    for idx, mask_info in enumerate(mask_infos):
        instance_info_dir = os.path.join(output_dir, str(idx), 'ins_info')
        if os.path.exists(os.path.join(instance_info_dir, 'top_k_masks.pt')):
            continue
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
        # instance_info_dir = os.path.join(output_dir, str(idx), 'ins_info')
        os.makedirs(instance_info_dir, exist_ok=True)

        # Save top_k_masks, pc_depth, and screen_coords as pt files
        torch.save(top_k_masks, os.path.join(instance_info_dir, 'top_k_masks.pt'))
        torch.save(pc_depth, os.path.join(instance_info_dir, 'pc_depth.pt'))
        torch.save(screen_coords, os.path.join(instance_info_dir, 'screen_coords.pt'))

        # Save obj_xyz as numpy array
        np.save(os.path.join(instance_info_dir, 'obj_xyz.npy'), obj_xyz)

        print(f"Saved instance information and point cloud data to {instance_info_dir}")
    # return top_k_masks, img_dir, pc_depth, screen_coords, num_views, cameras,obj_xyz

# Main execution
if __name__ == "__main__":
    NUM_PROMPTS = 1024
    NUM_MASKS_PER_PROMPT = 3
    NMS_THRESHOLD = 0.1
    TOP_K_PROPOSALS = 250

    dataset_dir = '/home/wan/Datasets/Test_scene/part_valid'
    project_path = '/home/wan/Workplace-why/PartScene'
    final_masks_save_dir = os.path.join(project_path, 'part_scene_results')
    by_product_save_dir = 'part_scene_saved'
    ckpt_path = os.path.join(project_path, "checkpoints/model.safetensors")

    model = build_point_sam(ckpt_path, 512, 64).to(device)
    print('Model built successfully')

    for scene_id in os.listdir(final_masks_save_dir)[:]:
        # print(scene_id)
            print(scene_id)
        # if scene_id == '0013':
            # continue
            scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
            mask_info_path = os.path.join(final_masks_save_dir, scene_id, f'{scene_id}_summary.txt')
            output_dir = os.path.join(project_path, by_product_save_dir, scene_id)
            os.makedirs(output_dir, exist_ok=True)
            mask_infos = load_prediction_data(mask_info_path)

        
            process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos)
            
    for scene_id in os.listdir(final_masks_save_dir)[:]:
        # print(scene_id)
            print(scene_id)
        # if scene_id == '0013':
            # continue
            scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
            mask_info_path = os.path.join(final_masks_save_dir, scene_id, f'{scene_id}_summary.txt')
            output_dir = os.path.join(project_path, by_product_save_dir, scene_id)
            os.makedirs(output_dir, exist_ok=True)
            mask_infos = load_prediction_data(mask_info_path)

        
            process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos)

          # Remove this if you want to process all scenes


import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
from point_sam.build_model import build_point_sam
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pytorch3d.ops as ops
from pytorch3d.ops import sample_farthest_points
from utils.nms import apply_pointwise_nms,visualize_point_clouds_with_masks
from mask_proposal import  mask_proposal,mask_proposal_v2,batch_mask_proposal
from utils.render import render_all_angles_pc,render_single_view,project_3d_to_2d
import glob
from point_sam.build_model import build_point_sam
import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
# Use glob to access all files in the directory
import os
from utils.inference_florence import run_florence2
from PIL import Image
import cv2
import supervision as sv

import open3d as o3d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def load_prediction_data(filename):
#     with open(filename, 'r') as file:
#         return [{'file': parts[0], 'prediction': int(parts[1]), 'confidence': float(parts[2])}
#                 for line in file if len(parts := line.strip().split()) == 3]

# def normalize_point_cloud(xyz):
#     centroid = np.mean(xyz, axis=0)
#     xyz_centered = xyz - centroid
#     furthest_distance = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
#     return xyz_centered / furthest_distance

# def process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos):
#     pcd = o3d.io.read_point_cloud(scene_path)
#     xyz = np.asarray(pcd.points)
#     rgb = np.asarray(pcd.colors) * 255

#     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))

# # Rotate point cloud

# # Rotate mesh
#     pcd = pcd.rotate(rotation_matrix, center=(0, 0, 0)) 
    
    for idx, mask_info in enumerate(mask_infos):
        instance_info_dir = os.path.join(output_dir, str(idx), 'ins_info')
        if os.path.exists(os.path.join(instance_info_dir, 'top_k_masks.pt')):
            continue
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
        # instance_info_dir = os.path.join(output_dir, str(idx), 'ins_info')
        os.makedirs(instance_info_dir, exist_ok=True)

        # Save top_k_masks, pc_depth, and screen_coords as pt files
        torch.save(top_k_masks, os.path.join(instance_info_dir, 'top_k_masks.pt'))
        torch.save(pc_depth, os.path.join(instance_info_dir, 'pc_depth.pt'))
        torch.save(screen_coords, os.path.join(instance_info_dir, 'screen_coords.pt'))

        # Save obj_xyz as numpy array
        np.save(os.path.join(instance_info_dir, 'obj_xyz.npy'), obj_xyz)

        print(f"Saved instance information and point cloud data to {instance_info_dir}")
    # return top_k_masks, img_dir, pc_depth, screen_coords, num_views, cameras,obj_xyz

# Main execution
if __name__ == "__main__":
    NUM_PROMPTS = 1024
    NUM_MASKS_PER_PROMPT = 3
    NMS_THRESHOLD = 0.1
    TOP_K_PROPOSALS = 250

    dataset_dir = '/home/wan/Datasets/Test_scene/part_valid'
    project_path = '/home/wan/Workplace-why/PartScene'
    final_masks_save_dir = os.path.join(project_path, 'part_scene_results')
    by_product_save_dir = 'part_scene_saved'
    ckpt_path = os.path.join(project_path, "checkpoints/model.safetensors")

    model = build_point_sam(ckpt_path, 512, 64).to(device)
    print('Model built successfully')

    for scene_id in os.listdir(final_masks_save_dir)[:]:
        # print(scene_id)
            print(scene_id)
        # if scene_id == '0013':
            # continue
            scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
            mask_info_path = os.path.join(final_masks_save_dir, scene_id, f'{scene_id}_summary.txt')
            output_dir = os.path.join(project_path, by_product_save_dir, scene_id)
            os.makedirs(output_dir, exist_ok=True)
            mask_infos = load_prediction_data(mask_info_path)

        
            process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos)
            
    for scene_id in os.listdir(final_masks_save_dir)[:]:
        # print(scene_id)
            print(scene_id)
        # if scene_id == '0013':
            # continue
            scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
            mask_info_path = os.path.join(final_masks_save_dir, scene_id, f'{scene_id}_summary.txt')
            output_dir = os.path.join(project_path, by_product_save_dir, scene_id)
            os.makedirs(output_dir, exist_ok=True)
            mask_infos = load_prediction_data(mask_info_path)

        
            process_scene(scene_id, scene_path, mask_info_path, model, output_dir,mask_infos)

          # Remove this if you want to process all scenes