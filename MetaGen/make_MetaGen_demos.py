#run from shell with exec(open("ORB_project3/MetaGen/show_images.py").read())

#make gif from command line with convert -delay 50 *.jpg(n) out.gif

import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math

office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]

f = open("ORB_project3/Data/output.json",)
dict = json.load(f)

v = 31

for x in range(300):
    im_path = dict[v]["path"]
    img = Image.open("SceneNet/pySceneNetRGBD/data/train/" + im_path + "/photo/" + str(25*x) + ".jpg")

    centers = dict[v]["views"][x]["inferences"]["centers"]
    for i in range(len(centers)):
        center = dict[v]["views"][x]["inferences"]["centers"][i]
        label = dict[v]["views"][x]["inferences"]['labels'][i]
        draw = ImageDraw.Draw(img)
        text = str(office_subset[label-1])
        text_location = [center[0] + 2, center[1]]
        draw.ellipse([center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1], 'red')
        draw.text(xy=text_location, text=text, fill='white')

    name = "scratch_work_07_16_21/08_09/" + str(v) + "/MetaGen/"+ str(x) +".jpg"

    img.save(name)
