#run from shell with exec(open("ORB_project3/MetaGen/make_NN_demos.py").read())

#make gif from command line with convert -delay 50 *.jpg(n) out.gif

import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math
import os

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

COCO_CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
    'spoon',
    'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush')

    #office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]
office_subset = ["chair", "couch"]

threshold = 0.11
num_videos = 10
num_frames = 200

f = open("scratch_work_07_16_21/08_20/data_labelled.json",)
dict = json.load(f)

for v in range(num_videos):
    if not os.path.exists(f"scratch_work_07_16_21/08_20/{v}"):
        os.mkdir(f"scratch_work_07_16_21/08_20/{v}")
        os.mkdir(f"scratch_work_07_16_21/08_20/{v}/NN_{threshold}")

    for x in range(num_frames):
            #img64 = dict[v]["views"][x]["image"]
            #im_bytes = base64.b64decode(img64)
            #im_file = BytesIO(im_bytes)
        im_path = dict[v]["path"]
            #img = Image.open("SceneNet/pySceneNetRGBD/data/train/" + im_path + "/photo/" + str(25*x) + ".jpg")
        img = Image.open(f"metagen-data/dist/img_{im_path}_{x:03}.png")
        centers = dict[v]["views"][x]["detections"]["center"]
        for i in range(len(centers)):
            center = dict[v]["views"][x]["detections"]["center"][i]
            label = dict[v]["views"][x]["detections"]['labels'][i]
            prob = dict[v]["views"][x]["detections"]['scores'][i]
            word_label = COCO_CLASSES[label]
            if word_label in office_subset and prob > threshold:
                draw = ImageDraw.Draw(img)
                text = word_label + " " + str(truncate(prob, 2))
                text_location = [center[0] + 2, center[1]]
                draw.ellipse([center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1], 'red')
                draw.text(xy=text_location, text=text, fill='white')

        name = (f"scratch_work_07_16_21/08_20/{v}/NN_{threshold}/{x:03}.png")
        #name = ("scratch_work_07_16_21/08_20/" + str(v) + "/NN_" + str(threshold) + "/" + str(x) + ".png")
        print(name)
        img.save(name)
