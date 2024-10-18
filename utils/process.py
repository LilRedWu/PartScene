import os
import numpy as np
import open3d as o3d
import sys
import sys
import glob
import shutil
import glob
import re

def process_3d_scenes(dataset_dir, save_dir):
    scene_dir = os.path.join(dataset_dir, 'part_valid')
    gt_instance_dir = os.path.join(dataset_dir, 'mask_valid_gt')

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for scene_id in os.listdir(scene_dir):
        # Create a directory for each scene
        scene_save_dir = os.path.join(save_dir, scene_id)
        os.makedirs(scene_save_dir, exist_ok=True)
        
        # Create pred_mask directory within the scene directory
        pred_mask_dir = os.path.join(scene_save_dir, 'pred_mask')
        os.makedirs(pred_mask_dir, exist_ok=True)

        # Read the point cloud
        scene_ply = o3d.io.read_point_cloud(os.path.join(scene_dir, scene_id, f'points_{scene_id}.ply'))
        points = np.asarray(scene_ply.points)

        # Load the mask
        masks = np.loadtxt(os.path.join(gt_instance_dir, f'mask_{scene_id}.txt'))

        # Ensure masks have the same length as points
        assert len(masks) == len(points), f"Mismatch in length for scene {scene_id}"

        # Get unique labels
        unique_labels = np.unique(masks)

        # Process each unique label
        for label in unique_labels:
            if label == 0:  # Assuming 0 is background or unlabeled
                continue

            # Extract points for this label
            label_mask = masks == label
            label_points = points[label_mask]

            # Save the mask for this label
            mask_filename = os.path.join(pred_mask_dir, f'{int(label):03d}.txt')
            np.savetxt(mask_filename, label_mask.astype(int), fmt='%d')

            # Optionally, you can save the points for this label as well
            # points_filename = os.path.join(pred_mask_dir, f'{int(label):03d}_points.txt')
            # np.savetxt(points_filename, label_points)

        print(f"Processed scene {scene_id}")

    print("All scenes processed.")

# example usage
# dataset_dir = '/home/wan/Datasets/Test_scene'
# scene_dir = os.path.join(dataset_dir, 'part_valid')
# gt_instance_dir = os.path.join(dataset_dir, 'mask_valid_gt')
# save_dir = 'part_scene_results'

prompt_dict = {

    "Regular Chair":["leg","back","seat","arm"],
    "Regular Table":["leg","tabletop"]

}



def map_parts_to_scene(scene_pcd, part_mask, ins_mask):
    ins_points = np.asarray(scene_pcd.points)[ins_mask]
    binary_ins = np.zeros(len(ins_points), dtype=bool)
    binary_scene = np.zeros(len(scene_pcd.points), dtype=bool)

    # Mark the parts as True in the binary array
    binary_ins[part_mask] = True
    binary_scene[ins_mask] = binary_ins

    # The scene_part_mask is now directly equal to binary_scene
    scene_part_mask = binary_scene

    return scene_part_mask



def save_mask_results(scene_id, part_mask_after_process, scene_pcd, ins_mask, ins, output_dir, part_label_v2):
    try:
        scene_dir = os.path.join(output_dir, f'{scene_id}')
        pred_part_mask_dir = os.path.join(scene_dir, 'pred_part_mask')
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
    if numeric_files:
        highest_idx = max([int(os.path.splitext(os.path.basename(f))[0]) for f in numeric_files])
        start_idx = highest_idx + 1
    else:
        start_idx = 0

    for idx, (label_key, data) in enumerate(part_mask_after_process.items(), start=start_idx):
        part_mask = data['mask']
        part_score = data['score']
        
        # Map parts to scene
        scene_part_mask = map_parts_to_scene(scene_pcd, part_mask, ins_mask)
        
        # Save individual mask file
        mask_filename = f'{idx:03d}.txt'
        mask_filepath = os.path.join(pred_part_mask_dir, mask_filename)
        
        # Convert scene_part_mask to integer numpy array
        if isinstance(scene_part_mask, np.ndarray):
            mask_to_save = scene_part_mask.astype(int)
        else:
            mask_to_save = np.array(scene_part_mask, dtype=int)
        
        try:
            # Save the mask
            np.savetxt(mask_filepath, mask_to_save, fmt='%d')
        except PermissionError:
            print(f"Error: Permission denied when trying to save file: {mask_filepath}")
            print("Please check that you have write permissions for the output directory.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while saving mask file: {str(e)}")
            sys.exit(1)
        
        # Get part label and number
        part_label = f'{base_cls}_{label_key}'
        # print(part_label)
        part_label_num = part_label_v2.get(part_label)  # This will return None if label not found
        
        # Append to summary data
        summary_data.append(f"pred_part_mask/{mask_filename} {part_label_num} {part_score:.4f}")
    
    # Save part summary file at the same level as regular summary
    part_summary_filepath = os.path.join(scene_dir, f'{scene_id}_part_summary.txt')
    
    try:
        # Append to part summary file
        with open(part_summary_filepath, 'a') as f:
            f.write('\n'.join(summary_data) + '\n')
        print(f"Part summary appended in {part_summary_filepath}")
    except PermissionError:
        print(f"Error: Permission denied when trying to save file: {part_summary_filepath}")
        print("Please check that you have write permissions for the output directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving part summary file: {str(e)}")
        sys.exit(1)


def process_mask_results(mask_results, top_k_masks):
    merged_masks = {}
    merged_scores = {}
    mask_counts = {}
    
    for idx, result in mask_results.items():
        label = result['label']
        part_mask = top_k_masks[idx].cpu().numpy()
        score = result['miou'].item()  # Convert tensor to float
        
        if label in merged_masks:
            # Merge mask
            merged_masks[label] = np.logical_or(merged_masks[label], part_mask)
            
            # Update score (simple average)
            mask_counts[label] += 1
            merged_scores[label] += score
        else:
            # New label
            merged_masks[label] = part_mask
            merged_scores[label] = score
            mask_counts[label] = 1
    
    # Calculate average scores
    for label in merged_scores:
        merged_scores[label] /= mask_counts[label]
    
    # Combine results into a single dictionary
    merged_results = {
        label: {
            'mask': mask,
            'score': merged_scores[label]
        } for label, mask in merged_masks.items()
    }
    
    return merged_results
