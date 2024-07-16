import os
import json
import numpy as np
import copy
import random
import trimesh
from utils.utils import *
from utils.trans  import InstanceObject, place_object_on


surface_top_list = ['Mug','Hat','Bowl','Lamp','Vase','Laptop','Keyboard','Bag','Scissors','Display','Knife','Bottle','Earphone']

obejct_with_top_surface_list = ['StorageFurniture','Refrigerator', 'Table','Bed', 'Dishwasher']

obejct_without_top_surface_list =['TrashCan','Chair','Door']




cat_factors = {
    'Mug': (0.2, 0.5),        # Mugs typically vary a bit in size but are generally small.
    'Hat': (0.2, 0.5),        # Hats have some variation but are generally smaller.
    'Bowl': (0.2, 0.5),       # Bowls can range from small to medium sizes.
    'Lamp': (0.6, 0.9),       # Lamps can range from small desk lamps to larger ones.
    'Vase': (0.4, 0.8),       # Vases can vary significantly in size but are usually not as large as laptops.
    'Laptop': (0.8, 1.0),     # Laptops have a narrow size range and are relatively larger.
    'Keyboard': (0.8, 1.0),   # Keyboards also have a narrow size range similar to laptops.
    'Bag': (0.5, 1.0),        # Bags can vary significantly in size.
    'Scissors': (0.2, 0.5),   # Scissors typically have a limited size range and are generally small.
    'Display': (0.7, 1.0),    # Displays usually have a narrow size range and are relatively larger.
    'Knife': (0.2, 0.5),      # Knives can vary from small to medium sizes.
    'Bottle': (0.2, 0.5),     # Bottles can vary significantly in size.
    'Earphone': (0.2, 0.5)    # Earphones are generally very small.
}



config = load_json('scene_layout.json')
all_obj_dict = load_json('obj_correspondence.json')
large_id_list = [obj['obj_id'] for obj in config['large_objects']]

large_objects_contain = {}
tiny_obj_id_list = []
obj_num = 3
for obj_id in [obj['obj_id'] for obj in config['large_objects']]:
    tiny_objects = []
    for i in range(obj_num):
        tiny_obj_cat, tiny_obj_id= random_objects_generate(surface_top_list,all_obj_dict)
        tiny_obj_info = {
            "obj_id":tiny_obj_id,
            "category":tiny_obj_cat,
            "scale_factor":random.uniform(cat_factors[tiny_obj_cat][0],cat_factors[tiny_obj_cat][1])
        }
        tiny_objects.append(tiny_obj_info)
        tiny_obj_id_list.append(tiny_obj_id)
    large_objects_contain[obj_id] = tiny_objects
    
id_list = large_id_list+tiny_obj_id_list
point_clouds, mesh_dict, label_to_number, cat_to_id = generate_scene_objects(id_list)


# Initialize instances
instances = {}
cat_list =[]
import uuid

# Without clock


    
    
for large_object in config['large_objects']:
    lid = large_object['obj_id']
    mesh = copy.deepcopy(mesh_dict[lid])
    label_count = 0
    instance_id = str(uuid.uuid4())
    instance = InstanceObject(lid, cat_to_id[lid], mesh, large_object['scale_factor'])
    instance.position = large_object['position']
    instance.rotation = large_object['rotation']
    instance.apply_transformation()
    instances[f'{instance.category}:{instance_id}'] = instance

    large_object_surface_mesh = instance.get_surface_mesh()

    for tiny_object in large_objects_contain[lid]:
        print('Tiny',tiny_object['obj_id'])
        mesh = copy.deepcopy(mesh_dict[tiny_object['obj_id']])
        instance_id = str(uuid.uuid4())
        tiny_instance = InstanceObject(tiny_object['obj_id'], cat_to_id[tiny_object['obj_id']], mesh, tiny_object['scale_factor'])
        place_object_on(tiny_instance, instance, large_object_surface_mesh)
        instances[f'{tiny_instance.category}:{instance_id}'] = tiny_instance



import trimesh.scene

objects_mseh_list = [instance.mesh for instance in instances.values()]
tiny_combined_scene = trimesh.Scene()
# Instance level scene
instance_level_scene = trimesh.scene.scene.append_scenes(objects_mseh_list)

# Part level scene

part_level_scene,central_points_dict = generate_part_scene(instances)

# part_level_scene.export('scene_example_1.obj')

print('Done')
#TODO random generate the small instance on the large objects
#TODO fix the bug of disappearing parts