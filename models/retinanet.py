from visual_system import VisualSystem
import torch
import torchvision.models as models
import torchvision.transforms as transforms

class RetinaNet(VisualSystem): 
	
	def __init__(self): 
		self.device = 'cuda'
		self.model = models.detection.retinanet_resnet50_fpn(pretrained=False).to(self.device)
		self.model.load_state_dict(torch.load('pretrained_state_dict'))
		self.model.eval()
		self.transform = transforms.Compose([transforms.ToTensor()])
		
		self.key = [
    			'__background__', 'person', 'bicycle', 'car', 'motorcycle', 
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
			'toothbrush'
			]

	def forward(self, tensors): 
		gpu_tensors = [tensor.to(self.device) for tensor in tensors]
		with torch.no_grad():
			outputs = self.model(gpu_tensors)
		return outputs

def main(): 
	retnet = models.detection.retinanet_resnet50_fpn(pretrained=True)
	torch.save(retnet.state_dict(), './retinanet_state_dict')

if __name__=="__main__": 
	main()
			
		
