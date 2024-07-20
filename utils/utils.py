import os
import glob
import json
import numpy as np
import copy
import random
import pdb
import os
import numpy as np
import trimesh
import xatlas
# Load JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def generate_color_palette(num_colors):
    np.random.seed(42)
    return np.random.rand(num_colors, 3)  # Generate colors in range [0, 1]

def update_number_to_label(existing_dict, new_labels):
    start_index = len(existing_dict)
    for label in new_labels:
        if label not in existing_dict.values():
            existing_dict[start_index] = label
            start_index += 1
    return existing_dict


def extract_parts(data, parent_label=''):
    parts = []
    if isinstance(data, dict):
        if 'text' in data:
            label = parent_label + ' -> ' + data['text'] if parent_label else data['text']
            if 'objs' in data:
                parts.append((label, data['objs']))
            if 'children' in data:
                for child in data['children']:
                    parts.extend(extract_parts(child, label))
    elif isinstance(data, list):
        for item in data:
            parts.extend(extract_parts(item, parent_label))
    return parts


def load_and_color_meshes_instance(obj_id, parts, number_to_label=None):
    if number_to_label is None:
        number_to_label = {}

    obj_base_dir = f'/home/lidosan/Datasets/datasets--ShapeNet--PartNet-archive/blobs/data_v0/{obj_id}/objs'
    combined_mesh = trimesh.Scene()
    combined_vertices = []
    combined_labels = []

    # Flatten the list of parts to get all unique labels and object IDs
    all_objects = [(label, obj) for label, objs in parts for obj in objs]
    unique_labels = {f"{label}_{obj}" for label, obj in all_objects}
    
    # Update the global number_to_label dictionary
    number_to_label = update_number_to_label(number_to_label, unique_labels)
    
    color_palette = generate_color_palette(len(number_to_label))
    label_to_color = {number_to_label[i]: (color_palette[i] * 255).astype(np.uint8) for i in number_to_label.keys()}

    for label, objs in parts:
        for obj in objs:
            identifier = f"{label}_{obj}"
            color = label_to_color[identifier]
            label_number = list(number_to_label.keys())[list(number_to_label.values()).index(identifier)]
            mesh_path = os.path.join(obj_base_dir, f"{obj}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                vertices = mesh.vertices
                num_vertices = vertices.shape[0]
                labels = np.full((num_vertices, 1), label_number)

                combined_vertices.append(vertices)
                combined_labels.append(labels)

                # Assign a color to the mesh
                mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
                # Add mesh to the scene
                combined_mesh.add_geometry(mesh)
            else:
                print(f"Mesh file {mesh_path} not found!")

    if combined_vertices:
        combined_vertices = np.vstack(combined_vertices)
        combined_labels = np.vstack(combined_labels)
        point_cloud = np.hstack((combined_vertices, combined_labels))
    else:
        point_cloud = np.array([])

    return point_cloud, number_to_label, combined_mesh

def update_number_to_label(existing_dict, new_labels):
    start_index = len(existing_dict)
    print(start_index)
    for label in new_labels:
        if label not in existing_dict.values():
            existing_dict[start_index] = label
            start_index += 1
    return existing_dict

def load_and_color_meshes_semantics(obj_id, parts, number_to_label=None):
    # if len(number_to_label)==0 :
    #     number_to_label = {}

    obj_base_dir = f'/home/lidosan/Datasets/datasets--ShapeNet--PartNet-archive/blobs/data_v0/{obj_id}/objs'
    combined_mesh = trimesh.Scene()
    combined_vertices = []
    combined_labels = []

    # Get all unique labels
    unique_labels = {label for label, _ in parts}
    
    # Update the number_to_label dictionary
    number_to_label = update_number_to_label(number_to_label, unique_labels)
    
    color_palette = generate_color_palette(len(number_to_label))
    label_to_color = {label: color_palette[i] for i, label in enumerate(unique_labels)}

    label_counts = {label: 0 for label in unique_labels}

    for label, objs in parts:
        label_number = list(number_to_label.keys())[list(number_to_label.values()).index(label)]
        color = label_to_color[label]
        # label = label.split('->')[-1]
        for obj in objs:
            mesh_path = os.path.join(obj_base_dir, f"{obj}.obj")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(mesh_path)
                vertices = mesh.vertices
                num_vertices = vertices.shape[0]
                labels = np.full((num_vertices, 1), label_number)

                combined_vertices.append(vertices)
                combined_labels.append(labels)
                
                # Assign the color to the mesh
                mesh.visual.vertex_colors = np.tile(color, (mesh.vertices.shape[0], 1))
                
                # Create a unique node name
                unique_node_name = f"{label}_{label_counts[label]}"
                label_counts[label] += 1
                
                mesh.__name__ = unique_node_name.split('->')[0]+'_'+unique_node_name.split('->')[-1]
                # print(mesh.__name__)

                # Add mesh to the scene
                combined_mesh.add_geometry(mesh,node_name=unique_node_name.split('->')[-1],geom_name=unique_node_name.split('->')[-1])
            else:
                print(f"Mesh file {mesh_path} not found!")

    if combined_vertices:
        combined_vertices = np.vstack(combined_vertices)
        combined_labels = np.vstack(combined_labels)
        point_cloud = np.hstack((combined_vertices, combined_labels))
    else:
        point_cloud = np.array([])

    return point_cloud, number_to_label, combined_mesh


def generate_scene_objects(id_list): 
    point_clouds = {}
    mesh_dict = {}
    label_to_number = {}
    cat_to_id = {}
    for obj_id in id_list:
        # print(obj_id)
        obj_base_dir = '/home/lidosan/Datasets/datasets--ShapeNet--PartNet-archive/blobs/data_v0/{}'.format(obj_id)
        obj_structure = load_json(os.path.join(obj_base_dir, 'result.json'))
        obj_infromation = load_json(os.path.join(obj_base_dir, 'meta.json'))
        # Extract parts with labels and object files
        parts = extract_parts(obj_structure)

        # Load first object
        point_cloud, label_to_number, combined_mesh = load_and_color_meshes_semantics(obj_id, parts, label_to_number)

        mesh_dict[obj_id] = combined_mesh
        point_clouds[obj_id] = point_cloud
        cat_to_id[obj_id] = obj_infromation['model_cat']
    return point_clouds, mesh_dict, label_to_number,cat_to_id



def generate_part_scene(instances):
    combined_scene = trimesh.Scene()
    central_points_dict = {}
    cat_list = []
    for ins_obj in instances.values():
            cat_count = 0
            mesh = ins_obj.mesh
            category= f'{ins_obj.category}_{cat_count}'
            while True:
                if category in cat_list:
                    cat_count+=1
                    category= f'{ins_obj.category}_{cat_count}'
                else:
                    break
                cat_list.append(f'{category}')
            
            for node_name in mesh.graph.nodes_geometry:
                transform, geometry_name = mesh.graph[node_name]
                geometry = mesh.geometry[geometry_name]
                
                combined_scene.add_geometry(geometry, transform=transform)
                transformed_central_point = np.dot(transform, np.append(geometry.centroid, 1))[:3]
                central_points_dict[f"{category}:{geometry_name}"] = transformed_central_point
    return combined_scene,central_points_dict


def save_as_uv_mesh(mesh,output_path):
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    # Trimesh needs a material to export uv coordinates and always creates a *.mtl file.
    # Alternatively, we can use the `export` helper function to export the mesh as obj.
    xatlas.export(f"{output_path}", mesh.vertices[vmapping], indices, uvs)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def random_objects_generate(surface_top_list,all_obj_dict):
    selected_category = random.choice(surface_top_list)
    selected_obj_id = random.choice(list(all_obj_dict[selected_category].keys()))
    return selected_category,selected_obj_id