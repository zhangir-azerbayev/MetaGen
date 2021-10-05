from VisualSystem import VisualSystem
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.ops as ops
import numpy as np

COCO_CLASSES = np.array(['__background__', 'person', 'bicycle', 'car', 'motorcycle', 
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
            'toothbrush'])
 

def filter_classes(outputs):
    classes = ["potted plant", "tv", "umbrella", "bowl", "chair"]
    labels0 = np.array(outputs[0]["labels"].cpu())
    word_labels0 = COCO_CLASSES[labels0.astype(int)]
    temp0 = [(word_label in classes) for word_label in word_labels0]
    indices0 = np.where(temp0)
    out0 = {"boxes": outputs[0]["boxes"][indices0], "scores": outputs[0]["scores"][indices0], "labels": outputs[0]["labels"][indices0]}
    labels1 = np.array(outputs[1]["labels"].cpu())
    word_labels1 = COCO_CLASSES[labels1.astype(int)]
    temp1 = [(word_label in classes) for word_label in word_labels1]
    indices1 = np.where(temp1)
    out1 = {"boxes": outputs[1]["boxes"][indices1], "scores": outputs[1]["scores"][indices1], "labels": outputs[1]["labels"][indices1]}
    return [out0, out1]


class RetinaNet(VisualSystem): 
    
    def __init__(self): 
        self.device = 'cuda'
        self.model = models.detection.retinanet_resnet50_fpn(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load('retinanet_state_dict'))
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
        print(f"tensors {tensors}")
        gpu_tensors = [tensor.to(self.device) for tensor in tensors]
        with torch.no_grad():
            outputs = self.model(gpu_tensors)
        outputs = filter_classes(outputs)
        #perform nms
        index_0 = ops.nms(outputs[0]["boxes"], outputs[0]["scores"], 0.4)
        print(f"index_0 {index_0}")
        print(f"boxes {outputs[0]['boxes']}")
        print(f"boxes[index_0] {outputs[0]['boxes'][index_0]}")
        new_dict_0 = {"boxes": outputs[0]["boxes"][index_0], "scores": outputs[0]["scores"][index_0], "labels": outputs[0]["labels"][index_0]}
        index_1 = ops.nms(outputs[1]["boxes"], outputs[1]["scores"], 0.4)
        new_dict_1 = {"boxes": outputs[1]["boxes"][index_1], "scores": outputs[1]["scores"][index_1], "labels": outputs[1]["labels"][index_1]}
        results = [new_dict_0, new_dict_1]
        print(f"results {results}")
        return results
