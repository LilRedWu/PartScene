import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
from typing import Tuple, Dict, List

from PIL import Image

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,   
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor

)

# add path for demo utils functions 
import sys
import os

def render_single_view(pc, view, device, background_color=(1,0,0), resolution=1024, camera_distance=3, 
                        point_size=0.005, points_per_pixel=1, bin_size=0, znear=0.01):
    R, T = look_at_view_transform(camera_distance, view[0], view[1])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear)

    raster_settings = PointsRasterizationSettings(
        image_size=resolution, 
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    compositor=NormWeightedCompositor(background_color=background_color)
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=compositor
    )
    img = renderer(pc)
    depth = rasterizer(pc).idx
    screen_coords = cameras.transform_points_screen(pc._points_list[0], image_size=(resolution, resolution))
    return img, depth, screen_coords,cameras


def render_single_view_mesh(mesh, view, device, background_color=(1,1,1), resolution=1024, camera_distance=2.2, 
                        point_size=0.005, points_per_pixel=1, bin_size=0, znear=0.01):
    R, T = look_at_view_transform(camera_distance, view[0], view[1])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear)

    raster_settings =  RasterizationSettings(
    image_size=resolution, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
    
    

    mesh_rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    img = renderer(mesh)
    depth = mesh_rasterizer(mesh).pix_to_face
    # screen_coords = cameras.transform_points_screen(pc._points_list[0], image_size=(resolution, resolution))
    return img, depth, '' ,cameras

def render_all_angles(pc,mesh,save_dir, device):
    #pc = io.load_pointcloud(pc_file, device=device)

    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)
    indices = [0, 4, 7, 1, 5, 2, 8, 6, 3, 9]

    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240]]

    pc_depth_list = []
    screen_coords_list = []

    for i, view in enumerate(views):

        pc_img, pc_depth, screen_coords, cameras = render_single_view(pc = pc,view = view,device=device)
        mesh_img, mesh_depth, _, cameras = render_single_view_mesh(mesh = mesh.to(device),view =view,device=device)

        plt.imsave(os.path.join(img_dir, f"{i}.png"), mesh_img[0, ..., :3].cpu().numpy() * 0.99999)
        pc_depth_list.append(pc_depth)
        screen_coords_list.append(screen_coords)

    pc_depth = torch.cat(pc_depth_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]
    # np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    # np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())
    return img_dir, pc_depth.cpu().numpy(), screen_coords.cpu().numpy(), len(views)


def render_all_angles(pc,mesh,save_dir, device):
    #pc = io.load_pointcloud(pc_file, device=device)

    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)
    indices = [0, 4, 7, 1, 5, 2, 8, 6, 3, 9]

    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240]]

    pc_depth_list = []
    screen_coords_list = []

    for i, view in enumerate(views):

        pc_img, pc_depth, screen_coords, cameras = render_single_view(pc = pc,view = view,device=device)
        mesh_img, mesh_depth, _, cameras = render_single_view_mesh(mesh = mesh.to(device),view =view,device=device)

        plt.imsave(os.path.join(img_dir, f"{i}.png"), mesh_img[0, ..., :3].cpu().numpy() * 0.99999)
        pc_depth_list.append(pc_depth)
        screen_coords_list.append(screen_coords)

    pc_depth = torch.cat(pc_depth_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]
    # np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    # np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())
    return img_dir, pc_depth.cpu().numpy(), screen_coords.cpu().numpy(), len(views)



def render_all_angles(pc,mesh,save_dir, device):
    #pc = io.load_pointcloud(pc_file, device=device)

    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)
    indices = [0, 4, 7, 1, 5, 2, 8, 6, 3, 9]

    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240]]

    pc_depth_list = []
    screen_coords_list = []

    for i, view in enumerate(views):

        pc_img, pc_depth, screen_coords, cameras = render_single_view(pc = pc,view = view,device=device)
        mesh_img, mesh_depth, _, cameras = render_single_view_mesh(mesh = mesh.to(device),view =view,device=device)

        plt.imsave(os.path.join(img_dir, f"{i}.png"), mesh_img[0, ..., :3].cpu().numpy() * 0.99999)
        pc_depth_list.append(pc_depth)
        screen_coords_list.append(screen_coords)

    pc_depth = torch.cat(pc_depth_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]
    # np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    # np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())
    return img_dir, pc_depth.cpu().numpy(), screen_coords.cpu().numpy(), len(views)


def render_all_angles_pc(pc,save_dir, device):
    #pc = io.load_pointcloud(pc_file, device=device)

    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)
    indices = [0, 4, 7, 1, 5, 2, 8, 6, 3, 9]

    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    # views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240]]

    pc_depth_list = []
    screen_coords_list = []
    cameras_list = []
    for i, view in enumerate(views):

        pc_img, pc_depth, screen_coords, cameras = render_single_view(pc = pc,view = view,device=device)
        img_np = pc_img[0, ..., :3].cpu().numpy() * 0.99999  # First batch, RGB channels only
        img_np_uint8 = (img_np).astype(np.uint8)  # Convert float [0, 1] to uint8 [0, 255]

        plt.imsave(os.path.join(img_dir, f"{i}.png"), img_np_uint8)
        pc_depth_list.append(pc_depth)
        screen_coords_list.append(screen_coords)
        cameras_list.append(cameras)

    pc_depth = torch.cat(pc_depth_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]
    # np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    # np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())
    return img_dir, pc_depth.cpu().numpy(), screen_coords.cpu().numpy(), len(views), cameras_list


def render_all_angles_pc(pc,save_dir, device):
    #pc = io.load_pointcloud(pc_file, device=device)

    img_dir = os.path.join(save_dir, "rendered_img")
    os.makedirs(img_dir, exist_ok=True)
    indices = [0, 4, 7, 1, 5, 2, 8, 6, 3, 9]

    views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240], [-20, 60], [-20, 180], [-20, 300]]
    # views = [[10, 0], [10, 90], [10, 180], [10, 270], [40, 0], [40, 120], [40, 240]]

    pc_depth_list = []
    screen_coords_list = []
    cameras_list = []
    for i, view in enumerate(views):

        pc_img, pc_depth, screen_coords, cameras = render_single_view(pc = pc,view = view,device=device)
        img_np = pc_img[0, ..., :3].cpu().numpy() * 0.99999  # First batch, RGB channels only
        img_np_uint8 = (img_np).astype(np.uint8)  # Convert float [0, 1] to uint8 [0, 255]

        plt.imsave(os.path.join(img_dir, f"{i}.png"), img_np_uint8)
        pc_depth_list.append(pc_depth)
        screen_coords_list.append(screen_coords)
        cameras_list.append(cameras)

    pc_depth = torch.cat(pc_depth_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views),-1, 3)[...,:2]
    # np.save(f"{save_dir}/idx.npy", pc_idx.cpu().numpy())
    # np.save(f"{save_dir}/coor.npy", screen_coords.cpu().numpy())
    return img_dir, pc_depth.cpu().numpy(), screen_coords.cpu().numpy(), len(views), cameras_list


def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


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

