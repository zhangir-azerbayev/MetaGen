import torch 
import torchvision.models as models
retnet = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
torch.save(retnet.state_dict(), './faster_rcnn_state_dict')
