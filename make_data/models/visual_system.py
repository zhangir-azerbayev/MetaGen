"""
A generic API for artifical visual systems. 
Takes pytorch tensors for forward propagation. This might be a problem
"""
import torch
import torchvision.transforms as transforms 
from PIL import Image, ImageDraw, ImageFont

class VisualSystem: 
	def __init__(self): 
		self.model = None
		self.key = None
		self.transform = None

	def forward(self, tensors): 
		"""
		tensors: List of [C, H, W] pytorch tensors
		out: list of dictionaries with keys 'boxes, 'scores', 'labels`
		Note: outputs are on gpu
		"""
		return 
	def im_forward(self, images):
		"""
		images: list of images
		out: list of dictionaries with keys 'boxes', 'scores', 'labels'
		"""
		tensors = []
		for image in images:
			tensors.append(self.transform(image))

		return self.forward(tensors)
		
	def im_annotate(self, images):
		"""
		image: list of images
		out: list of annotated images 
		"""
		outputs = self.im_forward(images)
		annotated_images = images
		for n in range(len(images)): 
			draw = ImageDraw.Draw(annotated_images[n])
			for i in range(len(outputs[n]['boxes'])):
				box = outputs[n]['boxes'][i].tolist()
				prob = outputs[n]['scores'][i].item()
				label = outputs[n]['labels'][i].item()
				text = self.key[label] + str(prob)

				draw.rectangle(xy=box, outline='red')
				text_location = [box[0] + 2, box[1]]
				draw.text(xy=text_location, text=text, fill='white')
		return annotated_images
	
	def im_annotate_save(self, images):
		annotated_images = self.im_annotate(images)
		for i in range(len(annotated_images)): 
			path = 'annotated_images/' + str(i) + '.jpg'
			annotated_images[i].save(path)

	def vid_annotate(self, videos): 
		""" implement later """

	def vid_annotate_save(self, videos): 
		""" implement later """
