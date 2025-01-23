import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
from point_sam.build_model import build_point_sam
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pytorch3d.ops as ops
from pytorch3d.ops import sample_farthest_points
from point_sam.build_model import build_point_sam
import numpy as np
import torch
from scripts.utils import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.io import load_obj
from nms import apply_pointwise_nms,visualize_point_clouds_with_masks


# Constants
# NUM_PROMPTS = 10
NUM_PROMPTS = 1024
NUM_MASKS_PER_PROMPT = 3
NMS_THRESHOLD = 0.3
TOP_K_PROPOSALS = 100



def mask_proposal(xyz,rgb,num_masks,model):
    # Initialize prompt sampling with FPS
    batch_size = xyz.shape[0]
    lengths = torch.tensor([xyz.shape[1]] * batch_size).cuda()

    # Sample prompts using FPS (Farthest Point Sampling)
    prompt_coords, _ = sample_farthest_points(xyz, lengths=lengths, K=num_masks)
    prompt_labels = torch.ones((1, num_masks), dtype=torch.long).cuda()

    # Generate masks: 3 masks per prompt
    all_masks = []
    all_scores = []
    all_featas = []

    # Set point cloud for the model (downsampled)
    model.set_pointcloud(xyz, rgb)

    # Generate masks for each prompt
    with torch.no_grad():
        for i in tqdm(range(num_masks)):
            prompt = prompt_coords[:, i:i+1, :]  # (1, 1, 3)
            label = prompt_labels[:, i:i+1]      # (1, 1)
            masks, scores, featas = model.predict_masks(prompt, label)  # masks (batch_size, 3, num_points)
            
            all_masks.append(masks.squeeze(0))  # Collect masks (3, N_down)
            all_scores.append(scores)  # Collect scores for NMS
            all_featas.append(featas)  # Collect features for analysis

    # Convert the masks and scores to tensors for easier processing
    all_masks = torch.cat(all_masks, dim=0)  # Shape: (NUM_PROMPTS * 3, num_points)
    all_scores = torch.cat(all_scores)  # Shape: (NUM_PROMPTS * 3,)

    selected_masks, selected_scores = apply_pointwise_nms(all_masks, all_scores,xyz, threshold=NMS_THRESHOLD)

    # Truncate proposals to top K
    top_k_indices = torch.argsort(selected_scores, descending=True)[:TOP_K_PROPOSALS]
    top_k_masks = selected_masks[top_k_indices]
    top_k_scores = selected_scores[top_k_indices]

    return top_k_masks,top_k_indices,top_k_scores


def mask_proposal(xyz,rgb,num_masks,model):
    # Initialize prompt sampling with FPS
    batch_size = xyz.shape[0]
    lengths = torch.tensor([xyz.shape[1]] * batch_size).cuda()

    # Sample prompts using FPS (Farthest Point Sampling)
    prompt_coords, _ = sample_farthest_points(xyz, lengths=lengths, K=num_masks)
    prompt_labels = torch.ones((1, num_masks), dtype=torch.long).cuda()

    # Generate masks: 3 masks per prompt
    all_masks = []
    all_scores = []
    all_featas = []

    # Set point cloud for the model (downsampled)
    model.set_pointcloud(xyz, rgb)

    # Generate masks for each prompt
    with torch.no_grad():
        for i in tqdm(range(num_masks)):
            prompt = prompt_coords[:, i:i+1, :]  # (1, 1, 3)
            label = prompt_labels[:, i:i+1]      # (1, 1)
            masks, scores, featas = model.predict_masks(prompt, label)  # masks (batch_size, 3, num_points)
            
            all_masks.append(masks.squeeze(0))  # Collect masks (3, N_down)
            all_scores.append(scores)  # Collect scores for NMS
            all_featas.append(featas)  # Collect features for analysis

    # Convert the masks and scores to tensors for easier processing
    all_masks = torch.cat(all_masks, dim=0)  # Shape: (NUM_PROMPTS * 3, num_points)
    all_scores = torch.cat(all_scores)  # Shape: (NUM_PROMPTS * 3,)

    selected_masks, selected_scores = apply_pointwise_nms(all_masks, all_scores,xyz, threshold=NMS_THRESHOLD)

    # Truncate proposals to top K
    top_k_indices = torch.argsort(selected_scores, descending=True)[:TOP_K_PROPOSALS]
    top_k_masks = selected_masks[top_k_indices]
    top_k_scores = selected_scores[top_k_indices]

    return top_k_masks,top_k_indices,top_k_scores

def batch_mask_proposal(xyz, rgb, num_masks, ckpt_path="checkpoints/model.safetensors", nms_threshold=0.3, top_k_proposals=10):
    model = build_point_sam(ckpt_path, 512, 64).cuda()  # (ckpt_path, num_centers, KNN size)
    print('Build Model Success')

    # Initialize prompt sampling with FPS
    batch_size, num_points, _ = xyz.shape
    lengths = torch.full((batch_size,), num_points, dtype=torch.long, device='cuda')

    # Sample prompts using FPS (Farthest Point Sampling)
    prompt_coords, _ = sample_farthest_points(xyz, lengths=lengths, K=num_masks)
    prompt_labels = torch.ones((batch_size, num_masks), dtype=torch.long, device='cuda')

    # Generate masks: 3 masks per prompt
    all_masks = []
    all_scores = []
    all_featas = []

    # Set point cloud for the model (downsampled)
    model.set_pointcloud(xyz, rgb)

    # Generate masks for each prompt
    with torch.no_grad():
        for i in tqdm(range(num_masks)):
            prompt = prompt_coords[:, i:i+1, :]  # (batch_size, 1, 3)
            label = prompt_labels[:, i:i+1]      # (batch_size, 1)
            masks, scores, featas = model.predict_masks(prompt, label)  # masks (batch_size, 3, num_points)
            
            all_masks.append(masks)  # Collect masks (batch_size, 3, num_points)
            all_scores.append(scores)  # Collect scores for NMS
            all_featas.append(featas)  # Collect features for analysis

    # Convert the masks and scores to tensors for easier processing
    all_masks = torch.cat(all_masks, dim=1)  # Shape: (batch_size, NUM_PROMPTS * 3, num_points)
    all_scores = torch.cat(all_scores, dim=1)  # Shape: (batch_size, NUM_PROMPTS * 3)

    # Apply NMS and select top K proposals for each point cloud in the batch
    top_k_masks_list = []
    top_k_indices_list = []
    top_k_scores_list = []

    for b in range(batch_size):
        selected_masks, selected_scores = apply_pointwise_nms(all_masks[b], all_scores[b], xyz[b], threshold=nms_threshold)

        # Truncate proposals to top K
        top_k_indices = torch.argsort(selected_scores, descending=True)[:top_k_proposals]
        top_k_masks = selected_masks[top_k_indices]
        top_k_scores = selected_scores[top_k_indices]

        top_k_masks_list.append(top_k_masks)
        top_k_indices_list.append(top_k_indices)
        top_k_scores_list.append(top_k_scores)

    # Pad the results to handle different numbers of masks per point cloud
    # top_k_masks_padded = pad_sequence(top_k_masks_list, batch_first=True, padding_value=0)
    # top_k_indices_padded = pad_sequence(top_k_indices_list, batch_first=True, padding_value=-1)
    # top_k_scores_padded = pad_sequence(top_k_scores_list, batch_first=True, padding_value=0)

    return top_k_masks_list, top_k_indices_list, top_k_scores_list



def mask_proposal_v2(xyz, rgb, num_masks, ckpt_path="checkpoints/model.safetensors", N=1):
    model = build_point_sam(ckpt_path, 512, 64).cuda()  # (ckpt_path, num_centers, KNN size)
    print('Build Model Success')

    # Initialize prompt sampling with FPS
    batch_size = xyz.shape[0]
    lengths = torch.tensor([xyz.shape[1]] * batch_size).cuda()

    # Sample prompts using FPS (Farthest Point Sampling)
    prompt_coords, _ = sample_farthest_points(xyz, lengths=lengths, K=num_masks)
    
    # Prepare positive and negative labels
    prompt_labels_pos = torch.ones((1, num_masks), dtype=torch.long).cuda()  # Positive labels (1)
    prompt_labels_neg = torch.zeros((1, N), dtype=torch.long).cuda()  # Negative labels (0)

    # Generate masks: 3 masks per prompt
    all_masks = []
    all_scores = []
    all_featas = []

    # Set point cloud for the model (downsampled)
    model.set_pointcloud(xyz, rgb)

    # Generate masks for each prompt
    with torch.no_grad():
        for i in tqdm(range(num_masks)):
            # Positive label for the current point
            prompt_pos = prompt_coords[:, i:i+1, :]  # (1, 1, 3)
            label_pos = prompt_labels_pos[:, i:i+1]  # (1, 1) - Positive label

            # Compute distances between the current point and other FPS points
            dists = torch.norm(prompt_coords - prompt_coords[:, i:i+1, :], dim=-1)  # (1, num_masks)

            # Select the N farthest points as negative samples
            farthest_indices = torch.argsort(dists, descending=True)[0, 1:N+1]  # Indices of N farthest points (excluding the current one)
            prompt_neg = prompt_coords[:, farthest_indices, :]  # Negative prompts (1, N, 3)
            label_neg = prompt_labels_neg  # Negative labels (1, N)

            # Generate positive mask
            masks_pos, scores_pos, featas_pos = model.predict_masks(prompt_pos, label_pos)
            all_masks.append(masks_pos.squeeze(0))  # Collect masks (3, N_down)
            all_scores.append(scores_pos)  # Collect scores for NMS
            all_featas.append(featas_pos)  # Collect features for analysis

            # Generate negative masks for the N farthest points
            for j in range(N):
                prompt_neg_j = prompt_neg[:, j:j+1, :]
                label_neg_j = label_neg[:, j:j+1]
                masks_neg, scores_neg, featas_neg = model.predict_masks(prompt_neg_j, label_neg_j)

                all_masks.append(masks_neg.squeeze(0))  # Collect masks (3, N_down)
                all_scores.append(scores_neg)  # Collect scores for NMS
                all_featas.append(featas_neg)  # Collect features for analysis

    # Convert the masks and scores to tensors for easier processing
    all_masks = torch.cat(all_masks, dim=0)  # Shape: (NUM_PROMPTS * 3, num_points)
    all_scores = torch.cat(all_scores)  # Shape: (NUM_PROMPTS * 3,)

    # Apply NMS to select the best masks
    selected_masks, selected_scores = apply_pointwise_nms(all_masks, all_scores, xyz, threshold=NMS_THRESHOLD)

    # Truncate proposals to top K
    top_k_indices = torch.argsort(selected_scores, descending=True)[:TOP_K_PROPOSALS]
    top_k_masks = selected_masks[top_k_indices]
    top_k_scores = selected_scores[top_k_indices]

    return top_k_masks, top_k_indices, top_k_scores
