import torch 
import torchvision.models as models
retnet = models.detection.retinanet_resnet50_fpn(pretrained=True)
torch.save(retnet.state_dict(), './retinanet_state_dict')
