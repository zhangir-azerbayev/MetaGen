from models.retinanet import RetinaNet
import torch 
import torchvision.transforms as transforms 

import json
import io
import base64
from PIL import Image 

from tqdm import tqdm 

import sys 

device = 'cuda'

model = RetinaNet()

batch = sys.argv[1]

data = json.load(open('data_unlabelled/' + sys.argv[1] + '_data.json',"rb"))

for i, scene in enumerate(tqdm(data)): 
	# Loads images and converts to pytorch tensor
	torch_imgs = []
	for j, frame in enumerate(scene["views"]): 
		frame_num = j * 25 
		with open("/home/zaa7/scratch60/pySceneNetRGBD/data/train/" + scene["path"] + '/photo/' + str(frame_num) + '.jpg', "rb") as f: 
			img = f.read()
		pil_img = Image.open(io.BytesIO(img)).convert('RGB')
		torch_img = transforms.Compose([transforms.ToTensor()])(pil_img).to(device)
		torch_imgs.append(torch_img)

	# Does forward propagation in batches of 25
	detections = []
	
	with torch.no_grad(): 
		for f in range(0, len(scene["views"]), 25):
			detections = detections + model.forward(torch_imgs[f:f+25])

	# Formats detections and adds to data arra y
	for j, frame in enumerate(scene["views"]): 
		# Restrict to threshold of .25
		threshold = .25 
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

with open('data_labelled/' + sys.argv[1] + '_data_labelled.json', 'a') as f:
	json.dump(data, f)
	

