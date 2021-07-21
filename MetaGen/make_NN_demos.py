#run from shell with exec(open("ORB_project3/MetaGen/show_images.py").read())

#make gif from command line with convert -delay 50 *.jpg(n) out.gif

import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math

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

f = open("scratch_work_07_16_21/tiny_set_detections.json",)
dict = json.load(f)

v = 1

for x in range(300):
    img64 = dict[v]["views"][x]["image"]
    im_bytes = base64.b64decode(img64)
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file)

    centers = dict[v]["views"][x]["detections"]["center"]
    for i in range(len(centers)):
        center = dict[v]["views"][x]["detections"]["center"][i]
        label = dict[v]["views"][x]["detections"]['labels'][i]
        prob = dict[v]["views"][x]["detections"]['scores'][i]
        if prob > 0.5:
            draw = ImageDraw.Draw(img)
            text = str(COCO_CLASSES[label]) + " " + str(truncate(prob, 2))
            text_location = [center[0] + 2, center[1]]
            draw.ellipse([center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1], 'red')
            draw.text(xy=text_location, text=text, fill='white')

        name = "scratch_work_07_16_21/" + str(v) + "/"+ str(x) +".jpg"
        img.save(name)
