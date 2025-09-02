import timm
import torch
temp_model = timm.create_model('mobilenetv2_120d', features_only=True)
print("Feature channels:", [x.shape[1] for x in temp_model(torch.randn(1,3,224,224))])