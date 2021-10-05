from VisualSystem import VisualSystem 
import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms 
import torch.nn.functional as F

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def post_process(outputs, target_sizes):
    """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    return results

class DeTr(VisualSystem): 
    def __init__(self): 
        self.device='cuda'
        self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101', pretrained=True).to(self.device)
        #self.model.load_state_dict(torch.load('detr_state_dict'))
        self.model.eval()

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
            target_sizes = torch.tensor([[256, 256], [256, 256]])
            gpu_target_sizes = target_sizes.to(self.device)
            results = post_process(outputs, gpu_target_sizes)   
            #outputs = post_processing(outputs["pred_logits"], outputs["pred_boxes"])           
        print(f"results {results}")
        return results

