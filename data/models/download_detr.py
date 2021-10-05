import torch 

model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

torch.save(model.state_dict(), './detr_state_dict')
