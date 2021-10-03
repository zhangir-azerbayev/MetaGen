import os
import csv
import math
import random
import json
import math
from random import randrange
from tqdm import tqdm
random.seed(18)

import numpy as np
from scipy.spatial.distance import cdist as cdist

from pathlib import Path
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.output_data import OutputData, Bounds, Images, Collision, EnvironmentCollision

num_videos = 200
num_frames = 20
dimensions = [13, 13]


data = []


# Trajectory generators
def CircleTrajectory(num_frames, radius):
    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]

    for angle in angles:
        camera = {"x": radius * math.cos(angle), "y": 3, "z": radius * math.sin(angle)}
        lookat = {"x": -radius * math.cos(angle), "y": 0, "z": -radius * math.sin(angle)}
        yield camera, lookat


def gp_noise_trajectory(num_frames, radius, camera_var, camera_l,
        lookat_var, lookat_l, dimensions):
    def rbf_kernel(xa, xb, var, l):
        sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean') / l**2
        return var * np.exp(sq_norm)

    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]
    ts = np.expand_dims(np.linspace(0, 10, num_frames), 1)
    camera_cov = rbf_kernel(ts, ts, camera_var, camera_l)
    lookat_cov = rbf_kernel(ts, ts, lookat_var, lookat_l)

    camera_dx = np.random.multivariate_normal(mean=np.zeros(num_frames),
            cov=camera_cov, size=2)

    lookat_dx = np.random.multivariate_normal(mean=np.zeros(num_frames),
            cov=lookat_cov, size=3)

    for t in range(num_frames):
        camera_x = radius * math.cos(angles[t]) + camera_dx[0, t],
        camera_z = radius * math.sin(angles[t]) + camera_dx[1, t]

        camera = {"x": np.clip(camera_x, -dimensions[0]/2+1, dimensions[0]/2-1).item(),
                  "y": 1.5,
                  "z": np.clip(camera_z, -dimensions[1]/2+1, dimensions[1]/2-1).item()}
        lookat = {"x": lookat_dx[0, t],
                  "y": lookat_dx[2, t],
                  "z": lookat_dx[1, t]}

        yield camera, lookat

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



# Creates objects dictionary
lib = ModelLibrarian(library="models_core.json")
"""
objects = dict({})
search_terms = ["chair", "sofa"]

for search_term in search_terms:
    records = lib.search_records(search_term)
    objects[search_term] = [record.name for record in records]
"""

#objects = dict({'table': ['dining_room_table'], 'microwave': ['appliance-ge-profile-microwave'], 'backpack': ['b04_backpack'], 'bowl': ['round_bowl_small_beech'], 'sofa': ['sayonara_sofa']})
#objects = dict({'table': ['dining_room_table'], 'chair': ['blue_club_chair'], 'bowl': ['round_bowl_small_beech'], 'sofa': ['sayonara_sofa']})
objects = dict({'chair': ['blue_club_chair'], 'bowl': ['round_bowl_small_beech']})


for video in tqdm(range(num_videos)):
    c = Controller(launch_build=True)
    c.communicate({"$type": "set_render_quality", "render_quality": 3})
    c.communicate({"$type": "simulate_physics", "value": True})
    c.communicate({"$type": "set_time_step", "time_step": 1})

    print("video {}".format(video))
    # Loads scene, creates empty room

    resp = c.communicate([{"$type": "load_scene",
                           "scene_name": "ProcGenScene"},
                          TDWUtils.create_empty_room(*dimensions)])
    print("created empty scene")
    # Populates scene with objects (placement is uniform random)
    num_objects = randrange(1, 4)
    print("num_objects: ", num_objects)

    #initialize a tracker for where object get placed
    placed_objects_x = []
    placed_objects_z = []

    labels = []
    category_names = []
    placement_height = 0
    for obj_id in range(num_objects):
        x = random.uniform(-(dimensions[0]/2 - 3.5), dimensions[0]/2 - 3.5)
        z = random.uniform(-(dimensions[1]/2 - 3.5), dimensions[1]/2 - 3.5)

        j = 1 #indexes through the objects that have already been placed and accepted
        while j <= len(placed_objects_x):
            print("in while loop")
            if dist(placed_objects_x[j-1], x, placed_objects_z[j-1], z) < 3: #if too close, resample. j-1 is for zero-indexing
                x = random.uniform(-(dimensions[0]/2 - 3.5), dimensions[0]/2 - 3.5)
                z = random.uniform(-(dimensions[1]/2 - 3.5), dimensions[1]/2 - 3.5)
                j = 1 #restart loop over objects already placed
            else:
                j = j + 1

        placed_objects_x.append(x)
        placed_objects_z.append(z)

        angle = random.uniform(0, 360)

        coco_name = random.choice(list(objects.keys()))
        category_names.append(coco_name)
        tdw_name = random.choice(objects[coco_name])

        resp = c.communicate([c.get_add_object(model_name = tdw_name, object_id = obj_id),
                              {"$type": "teleport_object", "id": obj_id, "position": {"x": x, "y": placement_height, "z": z}},
                              {"$type": "rotate_object_to_euler_angles", "euler_angles": {"x": 0, "y": angle, "z": 0},
                                  "id": obj_id}])

        print("placed object {}".format(obj_id))
        print(tdw_name)
        print(x, z)
        print('#'*20)


    for obj_id in range(num_objects):
        resp = c.communicate({"$type": "set_kinematic_state", "id": obj_id, "is_kinematic": True, "use_gravity": False})

    # saves labels
    resp = c.communicate({"$type": "send_bounds", "ids": list(range(num_objects))})
    for r in resp:
        r_id = OutputData.get_data_type_id(r)
        if r_id == 'boun':
            b = Bounds(r)

    print('saving labels')
    print('number of responses: ', len(resp))
    for index in range(num_objects): #order coming out of response might not match order of object ids
        obj = b.get_id(index)
        position = b.get_center(index)
        labels.append({"category_name": category_names[obj], "position": position})

    c.communicate({"$type": "simulate_physics", "value": False})
    # Creates frames
    radius = dimensions[0]/2 - 2
    angles = [2 * math.pi * i / num_frames for i in range(num_frames)]

    avatar_id = "a"

    resp = c.communicate([{"$type": "create_avatar",
                           "type": "A_Img_Caps_Kinematic",
                           "avatar_id": avatar_id},
                          {"$type": "set_pass_masks",
                           "avatar_id": avatar_id,
                           "pass_masks": ["_img"]}])
    print("created avatar")

    views = []
    radius = dimensions[0]/2 - 1
    trajectories = gp_noise_trajectory(num_frames, radius, 0.35, 2.65, 0.2, 2.2,
            dimensions) #variance in camera, length in camera, var of lookat, length of lookat
    for frame, (camera, lookat) in zip(range(num_frames), trajectories):
        resp = c.communicate([{"$type": "teleport_avatar_to",
                               "position": camera,
                               "avatar_id": avatar_id},
                              {"$type": "look_at_position",
                               "avatar_id": avatar_id,
                               "position": lookat }])

        resp = c.communicate({"$type": "send_images", "frequency": "once"})
        views.append({"camera": camera, "lookat": lookat})

        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "imag":
                img = Images(r)

        TDWUtils.save_images(img, filename=f"{video:03}_{frame:03}")

    # saves scene to data list
    data.append({"views": views, "labels": labels, "path": f"{video:03}"})
    # Terminates build
    c.communicate({"$type": "terminate"})
    del c

with open("unlabelled_data.json", "w") as f:
    json.dump(data, f)
