from models.retinanet import RetinaNet
from models.detr import DeTr
from models.faster_rcnn import FasterRcnn
import torch 
import torchvision.transforms as transforms 
import json
import os
import io
from PIL import Image 

from tqdm import tqdm 

import sys 

device = 'cuda'

if sys.argv[1] == "retinanet": 
    model = RetinaNet()
elif sys.argv[1] == "detr": 
    model = DeTr()
elif sys.argv[1] == "faster_rcnn":
    model = FasterRcnn()
else: 
    raise Exception("invalid argument")

data = json.load(open('unlabelled_data/unlabelled_data.json',"rb"))

for i, scene in enumerate(tqdm(data)): 
    print(f"i {i}")
    # Loads images and converts to pytorch tensor
    torch_imgs = []
    for frame_num, frame in enumerate(scene["views"]): 
        with open(f"images/img_{scene['path']}_{frame_num:03}.png", "rb") as f: 
            img = f.read()
        pil_img = Image.open(io.BytesIO(img)).convert('RGB')
        torch_img = transforms.Compose([transforms.ToTensor()])(pil_img).to(device)
        torch_imgs.append(torch_img)

    # Does forward propagation in batches of 2 
    detections = []
    
    with torch.no_grad(): 
        for f in range(0, len(scene["views"]), 2):
            print("here")
            detections = detections + model.forward(torch_imgs[f:f+2])

    # Formats detections and adds to data array
    for j, frame in enumerate(scene["views"]): 
        # Restrict to threshold of .0
        threshold = .0 
        mask = detections[j]["scores"] > threshold 
        box_mask = torch.stack((mask, mask, mask, mask), dim=1)
        detections[j]['boxes'] = torch.reshape(torch.masked_select(detections[j]['boxes'], box_mask), (-1, 4))
        detections[j]['labels'] = torch.masked_select(detections[j]['labels'], mask)
        detections[j]['scores'] = torch.masked_select(detections[j]['scores'], mask)
        # Format 
        boxes = detections[j].pop("boxes")
        top_left = boxes[:, 0:2]
        bottom_right = boxes[:, 2:4]
        detections[j]["center"] = (top_left+bottom_right)/2
        detections[j]["top_left"] = top_left
        detections[j]["bottom_right"] = bottom_right
        for key in detections[j]:
            if isinstance(detections[j][key], torch.Tensor): 
                detections[j][key] = detections[j][key].tolist()
        data[i]["views"][j]["detections"] = detections[j]
    




if os.path.exists("data_labelled_{}.json"):
    os.remove("data_labelled_{}.json")

with open("data_labelled_{}.json".format(sys.argv[1]), 'a') as f:
    json.dump(data, f)
    

