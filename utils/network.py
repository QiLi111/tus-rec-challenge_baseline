
import torch
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch

def build_model(opt,in_frames, pred_dim):

    if opt.model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.features[0][0].out_channels, 
            kernel_size  = model.features[0][0].kernel_size, 
            stride       = model.features[0][0].stride, 
            padding      = model.features[0][0].padding, 
            bias         = model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features   = model.classifier[1].in_features,
            out_features  = pred_dim
        )
        # print(model.classifier[1].in_features)
        # print(model)


    # elif opt.model_name[:6] == "resnet":
    #     model = resnet101()
    #     model.conv1 = torch.nn.Conv2d(
    #         in_channels  = in_frames, 
    #         out_channels = model.conv1.out_channels,
    #         kernel_size  = model.conv1.kernel_size,
    #         stride       = model.conv1.stride,
    #         padding      = model.conv1.padding,
    #         bias         = model.conv1.bias
    #     )
    #     model.fc = torch.nn.Linear(
    #         in_features   = model.fc.in_features,
    #         out_features  = pred_dim
    #     )


    else:
        raise("Unknown model.")
    
    return model

