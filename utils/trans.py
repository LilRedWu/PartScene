import os
import glob
import json
import numpy as np
import copy
import random
import pdb
import trimesh



# Define helper transformation functions
def create_scale_matrix(scale_factor):
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    return scale_matrix

def create_rotation_matrix(axis, angle_degrees):
    rotation_axes = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    return trimesh.transformations.rotation_matrix(np.radians(angle_degrees), rotation_axes[axis])

def create_translation_matrix(translation):
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    return translation_matrix

def basic_transformation(mesh, scale_factor):
    rotation_matrix_x = create_rotation_matrix('x', 90)
    mesh.apply_transform(rotation_matrix_x)
    translation_to_ground = create_translation_matrix([0, 0, -mesh.bounds[0][2]])
    mesh.apply_transform(translation_to_ground)
    scale_matrix = create_scale_matrix(scale_factor)
    mesh.apply_transform(scale_matrix)
    return mesh


# Define the function to place objects on surfaces
def place_object_on(source_object, target_object, object_surface):
    surface_bounds = object_surface.bounds
    surface_bounds_min_x, surface_bounds_min_y = surface_bounds[0][0], surface_bounds[0][1]
    surface_bounds_max_x, surface_bounds_max_y = surface_bounds[1][0], surface_bounds[1][1]
    surface_height = surface_bounds[1][2]
    
    source_object_ = copy.deepcopy(source_object)
    # print('Initial bounds of the source object:', source_object.bounds)
    
    retry_count = 0
    while True:
        retry_count+=1
        if retry_count>5:
                raise Exception(f"Too much retry")
        random_x = random.uniform(surface_bounds_min_x, surface_bounds_max_x)
        random_y = random.uniform(surface_bounds_min_y, surface_bounds_max_y)
        place_at = [random_x, random_y, surface_height + 0.0001]

        angles = [180, 150, 120, 90, 60, 30, 0, -30, -60, -90, -120, -150, -180]
        random_rotation = random.choice(angles)

        source_object_.position = place_at
        source_object_.rotation = random_rotation
        source_object_.apply_transformation()

        if target_object.overlap_detection(source_object_.mesh):
            # print('Overlap detected, regenerating position...')
            source_object_ = copy.deepcopy(source_object)
            continue

        overlap = False
        for obj in target_object.objects_on_surface.values():
            if obj.overlap_detection(source_object_.mesh):
                overlap = True
                source_object_ = copy.deepcopy(source_object)
                break
        

        
        if not overlap:
            source_object.position = place_at
            source_object.rotation = random_rotation
            source_object.apply_transformation()
            target_object.objects_on_surface[source_object.category] = source_object
            # print('Final bounds of the source object:', source_object.position)

            break

    
# Define the InstanceObject class
class InstanceObject:
    def __init__(self, obj_id, category, instance_mesh, scale_factor, orientation=np.array([0, 1, 0])):
        self.obj_id = obj_id
        self.category = category
        self.mesh = basic_transformation(instance_mesh, scale_factor)
        self.scale_factor = scale_factor
        self.orientation = orientation

        # Initialize position and transformation attributes
        self.position = np.zeros(3)
        self.rotation_matrix = np.eye(4)
        self.translation_matrix = np.eye(4)

        # Extract mesh details
        self.contains = list(self.mesh.geometry.keys())
        self.central_points = {}
        self.bounds = self.mesh.bounds
        self.height = self.bounds[1][2] - self.bounds[0][2]
        self.part_list = []  # store all the parts information

        # Additional attributes
        self.rotation = 0
        self.transformation = None
        self.objects_on_surface = {}  # Objects on the surfaces


    def apply_transformation(self):
        final_translation_matrix = create_translation_matrix(self.position)
        center_translation_to_origin = create_translation_matrix(-self.mesh.centroid)
        center_translation_back = create_translation_matrix(self.mesh.centroid)
        final_transform = np.dot(
            center_translation_back,
            np.dot(create_rotation_matrix('z', self.rotation), center_translation_to_origin)
        )
        final_transform = np.dot(final_translation_matrix, final_transform)
        self.mesh.apply_transform(final_transform)
        self.transformation = final_transform
        return final_transform

    def get_surface_mesh(self):
        surface_nodename_map = {
            'Bed': ' Mattress_0',
            'Table': ' Board_0'
        }
        surface_nodename = surface_nodename_map[self.category]
        transform = self.mesh.graph[surface_nodename][0]
        top_mesh = copy.deepcopy(self.mesh.geometry[surface_nodename])
        top_mesh.apply_transform(transform)
        return top_mesh

    def overlap_detection(self, other_mesh):
        other_bbox = other_mesh.bounds
        for node_name in self.mesh.graph.nodes_geometry:
            geometry = self.mesh.geometry[node_name]
            transform = self.mesh.graph[node_name][0]
            geometry_copy = copy.deepcopy(geometry)
            geometry_copy.apply_transform(transform)
            self_bbox = geometry_copy.bounds
            intersects = not (
                self_bbox[1][0] < other_bbox[0][0] or self_bbox[0][0] > other_bbox[1][0] or  # x-axis
                self_bbox[1][1] < other_bbox[0][1] or self_bbox[0][1] > other_bbox[1][1] or  # y-axis
                self_bbox[1][2] < other_bbox[0][2] or self_bbox[0][2] > other_bbox[1][2]    # z-axis
            )
            if intersects:
                print('Overlap')
                return True
        return False


