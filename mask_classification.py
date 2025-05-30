
from utils.utils_3d import *
import torch
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import os
import json
import sys
import re
import glob
import shutil
from third_party.torkit3d.config.config import *
from tqdm import tqdm
from utils.process import *


def save_mask_results(scene_id, part_mask_after_process, scene_pcd, ins_mask, ins, output_dir, part_label_v2):
    """
    Save mask results to the specified output directory with improved error handling.
    
    Args:
        scene_id: ID of the scene
        part_mask_after_process: Dictionary containing mask data
        scene_pcd: Point cloud data of the scene
        ins_mask: Instance mask
        ins: Instance information
        output_dir: Directory to save output
        part_label_v2: Dictionary mapping part labels to numbers
    """
    # Create necessary directories
    scene_dir = os.path.join(output_dir, f'{scene_id}')
    pred_part_mask_dir = os.path.join(scene_dir, 'pred_part_mask')
    
    try:
        os.makedirs(pred_part_mask_dir, exist_ok=True)
    except PermissionError:
        print(f"Error: Permission denied when trying to create directory: {pred_part_mask_dir}")
        print("Please check that you have write permissions for the output directory.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Output directory path: {output_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while creating directories: {str(e)}")
        sys.exit(1)
    
    summary_data = []
    base_cls = ins.split(' ')[-1].lower()

    # Get the current highest index in the pred_part_mask directory
    existing_files = glob.glob(os.path.join(pred_part_mask_dir, '*.txt'))
    numeric_files = [f for f in existing_files if re.match(r'^\d+\.txt$', os.path.basename(f))]
    start_idx = max([int(os.path.splitext(os.path.basename(f))[0]) for f in numeric_files], default=-1) + 1

    for idx, (label_key, data) in enumerate(part_mask_after_process.items(), start=start_idx):
        part_mask = data['mask']
        part_score = data['score']
        
        # Map parts to scene
        scene_part_mask = map_parts_to_scene(scene_pcd, part_mask, ins_mask)
        
        # Save individual mask file
        mask_filename = f'{idx:03d}.txt'
        mask_filepath = os.path.join(pred_part_mask_dir, mask_filename)
        
        # Convert scene_part_mask to integer numpy array
        mask_to_save = np.asarray(scene_part_mask, dtype=int)
        
        try:
            # Save the mask
            np.savetxt(mask_filepath, mask_to_save, fmt='%d')
        except (PermissionError, IOError) as e:
            print(f"Error saving mask file: {mask_filepath}")
            print(f"Error details: {str(e)}")
            sys.exit(1)
        
        # Get part label and number
        part_label = f'{base_cls}_{label_key}'
        part_label_num = part_label_v2.get(part_label)  # Using .get() for safer dictionary access
        
        # Append to summary data
        summary_data.append(f"pred_part_mask/{mask_filename} {part_label_num} {part_score:.4f}")
    
    # Save part summary file at the same level as regular summary
    part_summary_filepath = os.path.join(scene_dir, f'{scene_id}_part_summary.txt')
    
    try:
        # Append to part summary file
        with open(part_summary_filepath, 'a') as f:
            f.write('\n'.join(summary_data) + '\n')
        print(f"Part summary appended in {part_summary_filepath}")
    except Exception as e:
        print(f"Error saving part summary file: {part_summary_filepath}")
        print(f"Error details: {str(e)}")
        sys.exit(1)


def process_scene(scene_id, dataset_dir, final_masks_save_dir, by_product_save_dir, project_path, reversed_dict, cls_part_dict, part_label_v2):
    """
    Process a single scene from the dataset.
    
    Args:
        scene_id: ID of the scene to process
        dataset_dir: Path to dataset directory
        final_masks_save_dir: Directory for final mask results
        by_product_save_dir: Directory for intermediate results
        project_path: Path to project
        reversed_dict: Reversed class dictionary
        cls_part_dict: Class-part dictionary
        part_label_v2: Dictionary mapping part labels to numbers
    """
    scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
    mask_result_path = os.path.join(final_masks_save_dir, scene_id)
    output_scene_dir = os.path.join(project_path, by_product_save_dir, scene_id)
    
    # Load prediction data
    mask_infos = load_prediction_data(f'{mask_result_path}/{scene_id}_summary.txt')
    pred_part_mask_dir = os.path.join(mask_result_path, 'pred_part_mask')

    # Check if the directory exists and remove it if necessary
    if os.path.exists(pred_part_mask_dir):
        shutil.rmtree(pred_part_mask_dir)
    
    # Create the directory
    os.makedirs(pred_part_mask_dir, exist_ok=True)

    # Check if the summary file exists
    summary_file = os.path.join(mask_result_path, f'{scene_id}_part_summary.txt')
    if os.path.exists(summary_file):
        os.remove(summary_file)

    # Load scene point cloud
    scene_pcd = o3d.io.read_point_cloud(scene_path)
    
    # Process each mask
    for idx, mask in enumerate(mask_infos):
        ins_num = mask['prediction']
        mask_file = mask['file']
        instance_dir = os.path.join(output_scene_dir, str(idx))
        
        # Load instance information
        top_k_masks, pc_depth, screen_coords, obj_xyz = load_instance_info(f'{instance_dir}/ins_info')
        ins = reversed_dict[ins_num]
        prompt = cls_part_dict[ins]
        
        # Get number of views
        num_views = pc_depth.shape[0]
        
        # Load segmentation result
        result_dict = torch.load(os.path.join(f'{instance_dir}/ins_info', 'sem_seg.pt'))
        ins_mask = np.loadtxt(f'{mask_result_path}/{mask_file}').astype('bool')
        
        # Project 3D to 2D
        mask2d_view_list, mask_2d_bbox_correspondences, binary_masks_list = project_3d_to_2d(
            obj_xyz, top_k_masks, screen_coords, pc_depth)
        
        # Process masks and calculate IoU
        target_3d_masks = process_masks_and_calculate_iou(
            result_dict, num_views, binary_masks_list, 0, 0.1)
        
        # Assign labels to masks
        final_predictions = assign_labels_to_masks(
            result_dict, target_3d_masks, num_views, N=2)
        
        # Process and save mask results
        part_mask_after_process = process_mask_results(final_predictions, top_k_masks)
        save_mask_results(
            scene_id, part_mask_after_process, scene_pcd, ins_mask, ins, 
            final_masks_save_dir, part_label_v2)


def main():
    """Main function to process all scenes in the dataset."""
    # Configuration
    dataset_dir = '/home/wan/Datasets/Test_scene/part_valid'
    project_path = '/home/wan/Workplace-why/PartScene'
    output_dir = 'part_scene_results'
    by_product_save_dir = 'part_scene_saved'
    ckpt_path = os.path.join(project_path, "checkpoints/model.safetensors")
    
    # Prepare directory paths
    final_masks_save_dir = os.path.join(project_path, output_dir)
    
    # Prepare dictionaries
    reversed_dict = {value: key for key, value in cls_dict.items()}
    
    # Process each scene
    scene_ids = os.listdir(final_masks_save_dir)
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        try:
            process_scene(
                scene_id, dataset_dir, final_masks_save_dir, by_product_save_dir,
                project_path, reversed_dict, cls_part_dict, part_label_v2
            )
        except Exception as e:
            print(f"Error processing scene {scene_id}: {str(e)}")
            continue


if __name__ == "__main__":
    main()



dataset_dir = '/home/wan/Datasets/Test_scene/part_valid'
project_path = '/home/wan/Workplace-why/PartScene'
output_dir = 'part_scene_results'
output_dir = 'part_scene_results'

final_masks_save_dir = os.path.join(project_path, 'part_scene_results')
by_product_save_dir = 'part_scene_saved'
ckpt_path = os.path.join(project_path, "checkpoints/model.safetensors")
reversed_dict = {value: key for key, value in cls_dict.items()}



for scene_id in tqdm(os.listdir(final_masks_save_dir)[:]): 
        scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
        mask_result_path = os.path.join(final_masks_save_dir, scene_id)
        output_scene_dir = os.path.join(project_path, by_product_save_dir, scene_id)
        mask_infos = load_prediction_data( f'{mask_result_path}/{scene_id}_summary.txt')
        pred_part_mask_dir = os.path.join(mask_result_path, 'pred_part_mask')

                # Check if the directory exists and remove it if necessary
        if os.path.exists(pred_part_mask_dir):
                        shutil.rmtree(pred_part_mask_dir)
                # Create the directory
        os.makedirs(pred_part_mask_dir)

                # Check if the summary file exists
        summary_file = os.path.join(mask_result_path, f'{scene_id}_part_summary.txt')
        if os.path.exists(summary_file):
                # Handle the case when the summary file exists (if needed)
                os.remove(summary_file)

        for idx,mask in enumerate(mask_infos):
                ins_num = mask['prediction']
                mask_file = mask['file']
                instance_dir = os.path.join(output_scene_dir,str(idx))
                top_k_masks,pc_depth,screen_coords,obj_xyz = load_instance_info(f'{instance_dir}/ins_info')
                ins =reversed_dict[ins_num]
                prompt = cls_part_dict[ins]
                file_paths = glob.glob(os.path.join(f'{instance_dir}/rendered_img', '*'))
                points_3d =[]
                visible_pts_list = []
                # Print all the files found
                num_views = pc_depth.shape[0]
                text_input = prompt
                # load the segment result:
                result_dict = torch.load(os.path.join(f'{instance_dir}/ins_info', 'sem_seg.pt'))
                ins_mask =np.loadtxt(f'{mask_result_path}/{mask_file}').astype('bool')
                mask2d_view_list, mask_2d_bbox_correspondences, binary_masks_list = project_3d_to_2d(obj_xyz, top_k_masks, screen_coords, pc_depth)
                target_3d_masks = process_masks_and_calculate_iou(result_dict, num_views, binary_masks_list, 0,0.1)
                final_predictions = assign_labels_to_masks(result_dict, target_3d_masks, num_views, N=2)
                scene_pcd = o3d.io.read_point_cloud(f'/home/wan/Datasets/Test_scene/part_valid/{scene_id}/points_{scene_id}.ply')




                part_mask_after_process = process_mask_results(final_predictions,top_k_masks)
                save_mask_results(scene_id, part_mask_after_process, scene_pcd, ins_mask, ins, output_dir, part_label_v2)      

