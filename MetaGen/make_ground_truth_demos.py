#run from shell with exec(open("ORB_project3/MetaGen/show_images.py").read())

#make gif from command line with convert -delay 50 *.jpg(n) out.gif

import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math

office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]

f = open("ORB_project3/Data/output_with_gt.json",)
dict = json.load(f)

v = 2

for x in range(300):
    #img64 = dict[v]["views"][x]["image"]
    #im_bytes = base64.b64decode(img64)
    #im_file = BytesIO(im_bytes)
    im_path = dict[v]["path"]
    img = Image.open("SceneNet/pySceneNetRGBD/data/train/" + im_path + "/photo/" + str(25*x) + ".jpg")

    centers = dict[v]["views"][x]["ground_truth"]["centers"]
    for i in range(len(centers)):
        center = dict[v]["views"][x]["ground_truth"]["centers"][i]
        label = dict[v]["views"][x]["ground_truth"]['labels'][i]
        draw = ImageDraw.Draw(img)
        text = str(office_subset[label-1])#-1 gets us from julia's indexing to python's
        text_location = [center[0] + 2, center[1]]
        draw.ellipse([center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1], 'red')
        draw.text(xy=text_location, text=text, fill='white')

    name = "scratch_work_07_16_21/08_09/" + str(v) + "/ground_truth/"+ str(x) +".jpg"
    img.save(name)
