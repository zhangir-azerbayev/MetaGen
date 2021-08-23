#run from shell with exec(open("ORB_project3/MetaGen/show_images.py").read())

#make gif from command line with convert -delay 50 *.jpg(n) out.gif

import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import math
import os

office_subset = ["chair", "keyboard", "laptop", "dining table", "potted plant", "cell phone", "bottle"]

num_videos = 10
num_frames = 200

f = open("scratch_work_07_16_21/08_20/output.json",)

dict = json.load(f)

for v in range(num_videos):
    #os.mkdir(f"scratch_work_07_16_21/08_20/{v}/MetaGen")

    for x in range(num_frames):
        im_path = dict[v]["path"]
        img = Image.open(f"metagen-data/dist/img_{im_path}_{x:03}.png")

        centers = dict[v]["views"][x]["inferences"]["centers"]
        for i in range(len(centers)):
            center = dict[v]["views"][x]["inferences"]["centers"][i]
            label = dict[v]["views"][x]["inferences"]['labels'][i]
            draw = ImageDraw.Draw(img)
            text = str(office_subset[label-1])
            text_location = [center[0] + 2, center[1]]
            draw.ellipse([center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1], 'red')
            draw.text(xy=text_location, text=text, fill='white')

        name = (f"scratch_work_07_16_21/08_20/{v}/MetaGen/{x:03}.png")
        img.save(name)
