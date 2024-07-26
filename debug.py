import os
import glob
import json
import numpy as np
import copy
import random
import pdb
import trimesh
from utils.trans import *
from utils.utils import *
from PIL import Image
import shutil
import torch


def downsample_mesh(mesh, target_num_vertices):
    # Use Trimesh's built-in simplification (decimation) method
    # The function simplifies the mesh to a target number of faces, not vertices
    # So we need to estimate the number of faces that would approximately give us the target number of vertices
    target_num_faces = int(target_num_vertices * (len(mesh.faces) / len(mesh.vertices)))
    
    # Simplify the mesh
    simplified_mesh = mesh.simplify_quadric_decimation(target_num_faces)
    
    # breakpoint()
    # Return the simplified mesh
    return simplified_mesh




def load_parts2obj_uv(obj_id, parts, label_to_index):
    obj_base_dir = f'/home/lidosan/Datasets/datasets--ShapeNet--PartNet-archive/blobs/data_v0/{obj_id}/objs'
    
    combined_mesh_list = []
    combined_vertices = []
    combined_labels = []
    obbs_dict = {}
    count_dict = {}
    unique_labels = {label for label, _ in parts}

    label_counts = {label: 0 for label in unique_labels}
    unique_labels_mesh_dict = {label: [] for label in unique_labels}

    for label, objs in parts:
        label_number = label_to_index.get(label)
        if label_number is None:
            print(f"Label {label} not found in label_to_index")
            continue
        
        for obj in objs:
            mesh_path = os.path.join(obj_base_dir, f"{obj}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                unique_node_name = f"{label}_{label_counts[label]}"
                unique_node_name = unique_node_name.split('/')[0] + ':' + unique_node_name.split('/')[-1]
                obbs_dict[unique_node_name] = mesh.bounds
                count_dict[unique_node_name] = mesh.vertices.shape[0]
                unique_labels_mesh_dict[label].append(mesh)
                label_counts[label] += 1

    for label, meshes in unique_labels_mesh_dict.items():
        label_number = label_to_index.get(label)
        if label_number is None:
            print(f"Label {label} not found in label_to_index")
            continue
            
        combined_mesh = trimesh.util.concatenate(meshes)
        if len(combined_mesh.vertices) > 10000:
            save_as_uv_mesh(downsample_mesh(combined_mesh, 10000), 'temp.obj')
        else:
            save_as_uv_mesh(combined_mesh, 'temp.obj')
        
        combined_mesh_uv = trimesh.load('temp.obj')
        if len(combined_mesh.vertices) > 1024:
            trimesh.grouping.merge_vertices(combined_mesh_uv, merge_tex=True)

        vertices = combined_mesh_uv.vertices
        num_vertices = vertices.shape[0]
        labels = np.full((num_vertices, 1), label_number)

        combined_vertices.append(vertices)
        combined_labels.append(labels)
        combined_mesh_list.append(combined_mesh_uv)
        
    if combined_vertices:
        combined_vertices = np.vstack(combined_vertices)
        combined_labels = np.vstack(combined_labels)
        point_cloud = np.hstack((combined_vertices, combined_labels))
    else:
        point_cloud = np.array([])

    final_mesh = trimesh.util.concatenate(combined_mesh_list)
    print(label,len(final_mesh.vertices), point_cloud.shape)
    
    return point_cloud, label_to_index, final_mesh, obbs_dict

import re
# Assuming the following functions are defined:
# load_json, extract_parts, load_parts2obj_uv

obj_ids = ["46531", "12159", "2554", "2355", "18233", "18277", "22981", "45187","9067","3512","3810","15719","10207","6490"]
target_dir = '/home/lidosan/Datasets/PartNet_complete'

label_to_index = load_json('label2idx.json')


def clean_label(label):
    """Remove specific substrings from the label."""
    cleaned_label = re.sub(r'\(other\)/other_leaf|\(other\)|leaf', '', label)
    return cleaned_label.strip()

def normalize_label(label):
    """Convert the label to the desired format and clean it."""
    replacements = {
        " ": "_",
        "/other_": "",
        "sitting_furniture": "chair",
        "pot_body": "body",
        "-": "_",
        "/cylinder": "",
        "/decoration": "",
        "star_shape": "star",
        "sofa_style_arm": "arm_sofa_style",
        "stuffs_contained_in_the_pot":"containing_things"
    }
    
    cleaned_label = clean_label(label).lower()

    for old, new in replacements.items():
        cleaned_label = cleaned_label.replace(old, new)

    if "chair/chair_base/" in cleaned_label:
        cleaned_label = "chair/chair_base"
    elif "storage_furniture/cabinet/cabinet_frame" in cleaned_label and "bar" in cleaned_label:
        # merge the storage_furniture_bar
        cleaned_label = "storage_furniture/cabinet/cabinet_frame"



    return cleaned_label



def extract_parts_v2(data, parent_label=''):
    parts = []
    if isinstance(data, dict):
        if 'text' in data:
            label = parent_label + '/' + data['text'] if parent_label else data['text']
            label = normalize_label(label)
            if 'objs' in data:
                parts.append((label, data['objs']))
            if 'children' in data:
                for child in data['children']:
                    parts.extend(extract_parts_v2(child, label))
    elif isinstance(data, list):
        for item in data:
            parts.extend(extract_parts_v2(item, parent_label))
    return parts



for obj_id in obj_ids:
    number_to_label = {}
    obj_base_dir = f'/home/lidosan/Datasets/datasets--ShapeNet--PartNet-archive/blobs/data_v0/{obj_id}'
    
    obj_structure = load_json(os.path.join(obj_base_dir, 'result.json'))
    model_cat = load_json(os.path.join(obj_base_dir, 'meta.json'))['model_cat']
    parts_render_img = os.path.join(obj_base_dir, 'parts_render', '0.png')
    # Extract parts with labels and object files
    parts = extract_parts_v2(obj_structure)
    # # Load parts and create the mesh
    points, number_to_label, combined_mesh,obbs = load_parts2obj_uv(obj_id, parts, label_to_index)
    
    