from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
# from torchvision.models.vgg import model_urls


def init_weights(modules):
    """Weight initialization."""
    for module in modules:
        if isinstance(module, nn.Conv2d):
            init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.01)
            module.bias.data.zero_()


class VGG16DBN(nn.Module):

    """VGG16 Configuration D with Batch Normalization."""

    def __init__(self, pretrained=False):
        """TODO: to be defined.

        :pretrained: TODO

        """
        nn.Module.__init__(self)

        # self._pretrained = pretrained
        # model_urls['vgg16_bn'] =\
        #     model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg16dbn_features = models.vgg16_bn(pretrained=pretrained).features
        self._slice1 = nn.Sequential()
        self._slice2 = nn.Sequential()
        self._slice3 = nn.Sequential()
        self._slice4 = nn.Sequential()
        self._slice5 = nn.Sequential()
        for index in range(12):  # conv2_2
            self._slice1.add_module(str(index), vgg16dbn_features[index])
        for index in range(12, 22):  # conv3_3
            self._slice2.add_module(str(index), vgg16dbn_features[index])
        for index in range(22, 32):  # conv4_3
            self._slice3.add_module(str(index), vgg16dbn_features[index])
        for index in range(32, 42):  # conv5_3
            self._slice4.add_module(str(index), vgg16dbn_features[index])

        self._slice5 = nn.Sequential(
            vgg16dbn_features[42],
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self._slice1.modules())
            init_weights(self._slice2.modules())
            init_weights(self._slice3.modules())
            init_weights(self._slice4.modules())
        # no pretrained parameters for fc6 and fc7
        init_weights(self._slice5.modules())

    def forward(self, in_tensor):
        """Downscale path."""
        mid_tensor = self._slice1(in_tensor)
        relu2_2 = mid_tensor
        mid_tensor = self._slice2(mid_tensor)
        relu3_3 = mid_tensor
        mid_tensor = self._slice3(mid_tensor)
        relu4_3 = mid_tensor
        mid_tensor = self._slice4(mid_tensor)
        relu5_3 = mid_tensor
        mid_tensor = self._slice5(mid_tensor)
        fc7 = mid_tensor
        VGGOutputs =\
            namedtuple('VGGOutputs',
                       ['fc7', 'relu5_3', 'relu4_3', 'relu3_3', 'relu2_2'])
        out = VGGOutputs(fc7, relu5_3, relu4_3, relu3_3, relu2_2)
        return out


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    model = VGG16DBN(pretrained=False)
    in_tensor = torch.empty(1, 3, 1024, 1472)
    out_tensor = model(in_tensor)
    print(out_tensor.relu2_2.shape)
    print(out_tensor.relu3_3.shape)
    print(out_tensor.relu4_3.shape)
    print(out_tensor.relu5_3.shape)
    print(out_tensor.fc7.shape)

    print(out_tensor.relu2_2.dtype)
    print(out_tensor.relu3_3.dtype)
    print(out_tensor.relu4_3.dtype)
    print(out_tensor.relu5_3.dtype)
    print(out_tensor.fc7.dtype)

    print(out_tensor.relu2_2.device)
    print(out_tensor.relu3_3.device)
    print(out_tensor.relu4_3.device)
    print(out_tensor.relu5_3.device)
    print(out_tensor.fc7.device)


if __name__ == "__main__":
    main()
