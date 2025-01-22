
import torch
import open3d as o3d
import numpy as np
import torch





# 3D IoU calculation
def iou(box_a, box_b):
    """
    Compute the IoU (Intersection over Union) between two 3D boxes.
    
    box_a and box_b are lists or tensors where:
    - box_a[0], box_a[1], box_a[2] are the x, y, z coordinates of the bottom-left corner.
    - box_a[3], box_a[4] are the width and height of the box in 2D.
    - box_a[5] is the length of the box in the z direction.
    
    Similarly for box_b.
    """
    # Calculate the top-right corner coordinates of both boxes in 2D
    box_a_top_right_corner = [box_a[0] + box_a[3], box_a[1] + box_a[4]]
    box_b_top_right_corner = [box_b[0] + box_b[3], box_b[1] + box_b[4]]

    # Compute the 2D area of both boxes (width * height)
    box_a_area = box_a[3] * box_a[4]
    box_b_area = box_b[3] * box_b[4]

    # Compute the intersection coordinates in 2D
    xi = max(box_a[0], box_b[0])
    yi = max(box_a[1], box_b[1])

    # Compute the top-right corner of the intersection in 2D
    corner_x_i = min(box_a_top_right_corner[0], box_b_top_right_corner[0])
    corner_y_i = min(box_a_top_right_corner[1], box_b_top_right_corner[1])

    # Compute the intersection area in 2D
    intersection_area = max(0, corner_x_i - xi) * max(0, corner_y_i - yi)

    # Now compute the intersection in the z (length) direction
    intersection_l_min = max(box_a[2], box_b[2])
    intersection_l_max = min(box_a[2] + box_a[5], box_b[2] + box_b[5])
    intersection_length = max(0, intersection_l_max - intersection_l_min)  # Ensure non-negative length

    # Compute the volume of the intersection
    intersection_volume = intersection_area * intersection_length

    # Compute the volume of both boxes (area * length)
    box_a_volume = box_a_area * box_a[5]
    box_b_volume = box_b_area * box_b[5]

    # IoU is the intersection volume divided by the union volume
    union_volume = box_a_volume + box_b_volume - intersection_volume

    # To avoid division by zero, add a small epsilon (1e-5)
    iou_value = intersection_volume / (union_volume + 1e-5)

    return iou_value


# Function to generate 3D bounding boxes from masks and xyz coordinates
# Function to generate 3D bounding boxes from masks and xyz coordinates
def generate_bounding_boxes_from_masks_and_xyz(masks, xyz):
    """
    Generate 3D bounding boxes from binary masks and corresponding xyz coordinates.
    :param masks: Tensor of shape (num_masks, num_points) with binary mask values.
    :param xyz: Tensor of shape (1, num_points, 3) with the actual 3D coordinates of each point.
    :return: boxes: Tensor of shape (num_masks, 6) containing [x_min, y_min, z_min, width, height, length] for each mask.
    """
    num_masks, num_points = masks.shape
    boxes = []

    for i in range(num_masks):
        # Get the masked xyz points directly using boolean indexing
        masked_xyz = xyz[0][masks[i]]  # Extract points for mask i from xyz
        if masked_xyz.shape[0] == 0:
            # If no points belong to the mask, create a zero bounding box
            box = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32)
        else:
            # Compute min and max coordinates for the bounding box
            x_min, y_min, z_min = masked_xyz.min(dim=0).values
            x_max, y_max, z_max = masked_xyz.max(dim=0).values

            # Calculate width, height, and length of the bounding box
            width = x_max - x_min
            height = y_max - y_min
            length = z_max - z_min

            # Bounding box format: [x_min, y_min, z_min, width, height, length]
            box = torch.tensor([x_min, y_min, z_min, width, height, length], dtype=torch.float32)

        boxes.append(box)

    # Stack all boxes into a single tensor
    boxes = torch.stack(boxes)

    return boxes


    return boxes
# Post-processing: Apply point-wise NMS to remove duplicate proposals
def apply_pointwise_nms(masks, scores, xyz, threshold):
    """
    Apply Non-Maximum Suppression (NMS) on point-wise masks using 3D IoU.
    :param masks: Tensor of shape (num_masks, num_points) with binary mask values
    :param scores: Tensor of shape (num_masks, *) with scores for each mask. This can have multiple elements.
    :param xyz: Tensor of shape (1, num_points, 3) with the actual 3D coordinates of each point.
    :param threshold: IoU threshold for suppression
    :return: selected_masks, selected_scores
    """
    # If scores have more than one dimension, reduce it (e.g., by taking the max score per mask)
    if scores.dim() > 1:
        scores = scores.max(dim=1).values  # Use the max score across the second dimension

    # Sort masks by their scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]

    # Generate bounding boxes from masks and xyz coordinates
    boxes = generate_bounding_boxes_from_masks_and_xyz(masks, xyz)

    selected_masks = []
    selected_scores = []
    suppressed_indices = set()

    for i in range(masks.shape[0]):
        if i in suppressed_indices:
            continue

        mask_i = masks[i]
        score_i = scores[i].item()  # Convert tensor to scalar
        box_i = boxes[i]

        for j in range(i + 1, masks.shape[0]):
            if j in suppressed_indices:
                continue
            mask_j = masks[j]
            box_j = boxes[j]

            # Call the 3D IoU function
            iou_value = iou(box_i, box_j)
            
            if iou_value > threshold:
                suppressed_indices.add(j)  # Suppress mask_j if IoU is greater than the threshold
        
        # Keep the current mask and score if it's not suppressed
        selected_masks.append(mask_i)
        selected_scores.append(score_i)

    # Convert the selected masks and scores to tensors
    selected_masks = torch.stack(selected_masks) if selected_masks else torch.empty((0, masks.shape[1]))
    selected_scores = torch.tensor(selected_scores)

    return selected_masks, selected_scores



# Post-processing: Apply point-wise NMS to remove duplicate proposals
def apply_pointwise_nms(masks, scores, xyz, threshold):
    """
    Apply Non-Maximum Suppression (NMS) on point-wise masks using 3D IoU.
    :param masks: Tensor of shape (num_masks, num_points) with binary mask values
    :param scores: Tensor of shape (num_masks, *) with scores for each mask. This can have multiple elements.
    :param xyz: Tensor of shape (1, num_points, 3) with the actual 3D coordinates of each point.
    :param threshold: IoU threshold for suppression
    :return: selected_masks, selected_scores
    """
    # If scores have more than one dimension, reduce it (e.g., by taking the max score per mask)
    if scores.dim() > 1:
        scores = scores.max(dim=1).values  # Use the max score across the second dimension

    # Sort masks by their scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]

    # Generate bounding boxes from masks and xyz coordinates
    boxes = generate_bounding_boxes_from_masks_and_xyz(masks, xyz)

    selected_masks = []
    selected_scores = []
    suppressed_indices = set()

    for i in range(masks.shape[0]):
        if i in suppressed_indices:
            continue

        mask_i = masks[i]
        score_i = scores[i].item()  # Convert tensor to scalar
        box_i = boxes[i]

        for j in range(i + 1, masks.shape[0]):
            if j in suppressed_indices:
                continue
            mask_j = masks[j]
            box_j = boxes[j]

            # Call the 3D IoU function
            iou_value = iou(box_i, box_j)
            
            if iou_value > threshold:
                suppressed_indices.add(j)  # Suppress mask_j if IoU is greater than the threshold
        
        # Keep the current mask and score if it's not suppressed
        selected_masks.append(mask_i)
        selected_scores.append(score_i)

    # Convert the selected masks and scores to tensors
    selected_masks = torch.stack(selected_masks) if selected_masks else torch.empty((0, masks.shape[1]))
    selected_scores = torch.tensor(selected_scores)

    return selected_masks, selected_scores




def visualize_point_clouds_with_masks(xyz, masks, mask_colors=None):
    """
    Visualize point cloud with multiple masks using Open3D, where each mask is shown as a separate point cloud.
    :param xyz: Point cloud coordinates (num_points, 3).
    :param masks: Binary masks (num_masks, num_points) indicating which points belong to which mask.
    :param mask_colors: List of colors for each mask. If None, random colors will be generated.
    """
    num_masks = masks.shape[0]
    num_points = xyz.shape[0]

    if isinstance(masks,torch.Tensor):
        masks = masks.cpu().numpy()
    # Ensure xyz is a numpy array of dtype float64 and correct shape

    xyz = np.asarray(xyz, dtype=np.float64)  # Ensure it's a numpy array of float64 type

    # Generate random colors for each mask if not provided
    if mask_colors is None:
        mask_colors = np.random.rand(num_masks, 3)  # Random RGB colors for each mask

    # List to store individual point clouds
    point_clouds = []

    # Create a point cloud for each mask
    for i in range(num_masks):

        print(i)
        print(xyz,xyz.shape)
        # Extract the points corresponding to the current mask
        mask = masks[i]
        masked_xyz = xyz[mask]  # Get the points corresponding to this mask


        print(masked_xyz)
        # Create an Open3D point cloud for this mask
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(masked_xyz)

        # Set the color for all points in this point cloud
        colors = np.tile(mask_colors[i], (masked_xyz.shape[0], 1))  # Apply the same color to all points
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Append the colored point cloud to the list
        point_clouds.append(point_cloud)

    # Visualize all point clouds together
    o3d.visualization.draw_plotly(point_clouds)
    
    return point_clouds