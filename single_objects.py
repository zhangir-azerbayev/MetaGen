import os
import csv
import math
import random
import json
from random import randrange
from tqdm import tqdm
random.seed(17)

from pathlib import Path
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.output_data import OutputData, Bounds, Images, Collision, EnvironmentCollision

num_frames = 50
dimensions = [10, 10]
data = []


# Creates objects dictionary
lib = ModelLibrarian(library="models_core.json")

objects = dict({})
search_terms = ["chair", "sofa"]

for search_term in search_terms:
    records = lib.search_records(search_term)
    objects[search_term] = [record.name for record in records]


print(objects)

video = 0
for search_term in objects:
    for name in objects[search_term]:
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
        print(name)

        labels = []

        # Places object in the center of the scene 
        resp = c.communicate([c.get_add_object(model_name = name, object_id = 0),
                              {"$type": "teleport_object", "id": 0, "position": {"x": 0, "y": 0, "z": 0}}])

        print("placed object")


        resp = c.communicate({"$type": "send_bounds", "id": 0})
        b = Bounds(resp[0])

        position = b.get_center(0)
        labels.append({"category_name": search_term, "tdw_name": name, "position": position})


        # Creates avatar 
        avatar_id = "a"
        resp = c.communicate([{"$type": "create_avatar",
                               "type": "A_Img_Caps_Kinematic",
                               "avatar_id": avatar_id},
                              {"$type": "look_at",
                               "avatar_id": avatar_id,
                               "object_id": 0},
                              {"$type": "set_pass_masks",
                               "avatar_id": avatar_id,
                               "pass_masks": ["_img"]}])

        radius = 3
        angles = [2*math.pi * i/num_frames for i in range(num_frames)]
    
        views = []
        for frame in range(num_frames):
            angle = angles[frame]

            camera = {"x": radius * math.cos(angle), "y": 2.5, "z": radius * math.sin(angle)}

            resp = c.communicate([{"$type": "teleport_avatar_to", 
                                   "position": camera, 
                                   "avatar_id": avatar_id}, 
                                  {"$type": "look_at", 
                                   "avatar_id": avatar_id, 
                                   "object_id": 0}])

            resp = c.communicate({"$type": "send_images", "frequency": "once"})
            views.append({"camera": camera, "lookat": position})

            for r in resp: 
                r_id = OutputData.get_data_type_id(r)
                if r_id == "imag": 
                    img = Images(r)

            TDWUtils.save_images(img, filename=f"{video:03}_{frame:03}_{name}")

        data.append({"views": views, "labels": labels, "path":f"{video:03}"})

        # Terminates build
        c.communicate({"$type": "terminate"})
        del c
        video+=1

with open("unlabelled_data.json", "w") as f: 
    json.dump(data, f)



