import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
from point_sam.build_model import build_point_sam
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import glob
import numpy as np
import torch
# Use glob to access all files in the directory
from transformers import AutoProcessor, AutoModelForCausalLM
import random
import os
from utils.inference_florence import run_florence2
from PIL import Image
import cv2
import supervision as sv
from third_party.Ground_SAM.sam2.build_sam import build_sam2
from third_party.Ground_SAM.sam2.sam2_image_predictor import SAM2ImagePredictor
from third_party.Ground_SAM.mask_proposal_2d import segment2d
import open3d as o3d
from utils.utils_3d import * 
import json 
from third_party.torkit3d.config.config import * 


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



import os 

FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "third_party/Ground_SAM/checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)
reversed_dict = {value: key for key, value in cls_dict.items()}



dataset_dir = '/home/wan/Datasets/Test_scene/part_valid'
project_path = '/home/wan/Workplace-why/Part-SAM'
final_masks_save_dir = os.path.join(project_path, 'part_scene_results')
by_product_save_dir = 'part_scene_saved'
ckpt_path = os.path.join(project_path, "checkpoints/model.safetensors")
for scene_id in tqdm(os.listdir(final_masks_save_dir)[:]):
        scene_path = os.path.join(dataset_dir, scene_id, f'points_{scene_id}.ply')
        # mask_info_path = os.path.join(final_masks_save_dir, scene_id, f'{scene_id}_summary.txt')
        # output_scene_dir = os.path.join(project_path, by_product_save_dir, scene_id)
        mask_infos = load_prediction_data(mask_info_path)
        print(scene_id)

        for idx,mask in enumerate(mask_infos):
                ins_num = mask['prediction']
                instance_dir = os.path.join(output_scene_dir,str(idx))
                top_k_masks,pc_depth,screen_coords,obj_xyz = load_instance_info(f'{instance_dir}/ins_info')
                # continue
                ins =reversed_dict[ins_num]
                prompt = cls_part_dict[ins]
                file_paths = glob.glob(os.path.join(f'{instance_dir}/rendered_img', '*'))
                points_3d =[]
                visible_pts_list = []
                # Print all the files found
                num_views = pc_depth.shape[0]
                task_prompt = "<OPEN_VOCABULARY_DETECTION>"
                text_input = prompt
                # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__(
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                # if os.path.exists('')
                sem_seg_path = os.path.join(f'{instance_dir}/ins_info', 'sem_seg.pt')
                if os.path.exists(sem_seg_path):
                       continue
                result_dict = segment2d(num_views = num_views ,save_dir=instance_dir,text_input=text_input,task_prompt=task_prompt,florence2_model=florence2_model,florence2_processor=florence2_processor,sam2_predictor= sam2_predictor)
                torch.save(result_dict, sem_seg_path)
                