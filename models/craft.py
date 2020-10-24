from pathlib import Path
current_file_path = Path(__file__).resolve()
import sys
sys.path.append(str(current_file_path.parent.parent))
from models.backbone import VGG16DBN, init_weights

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    """Docstring for DoubleConv. """

    def __init__(self, in_ch, mid_ch, out_ch):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        self._conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, in_tensor):
        """TODO: Docstring for forward.

        :in_tensor: TODO
        :returns: TODO

        """
        return self._conv(in_tensor)


class CRAFT(nn.Module):

    """Docstring for CRAFT. """

    def __init__(self, pretrained=False):
        """TODO: to be defined. """
        nn.Module.__init__(self)

        self._backbone = VGG16DBN(pretrained=pretrained)

        self._upconv1 = DoubleConv(1024, 512, 256)
        self._upconv2 = DoubleConv(512, 256, 128)
        self._upconv3 = DoubleConv(256, 128, 64)
        self._upconv4 = DoubleConv(128, 64, 32)

        num_classes = 2
        self._conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        init_weights(self._upconv1.modules())
        init_weights(self._upconv2.modules())
        init_weights(self._upconv3.modules())
        init_weights(self._upconv4.modules())
        init_weights(self._conv_cls.modules())

    def forward(self, in_ten):
        """ Backbone """
        sources = self._backbone(in_ten)

        """ Unet """
        mid_ten = torch.cat([sources[0], sources[1]], dim=1)
        mid_ten = self._upconv1(mid_ten)

        mid_ten = F.interpolate(mid_ten, size=sources[2].size()[2:],
                                mode='bilinear', align_corners=False)
        mid_ten = torch.cat([mid_ten, sources[2]], dim=1)
        mid_ten = self._upconv2(mid_ten)

        mid_ten = F.interpolate(mid_ten, size=sources[3].size()[2:],
                                mode='bilinear', align_corners=False)
        mid_ten = torch.cat([mid_ten, sources[3]], dim=1)
        mid_ten = self._upconv3(mid_ten)

        mid_ten = F.interpolate(mid_ten, size=sources[4].size()[2:],
                                mode='bilinear', align_corners=False)
        mid_ten = torch.cat([mid_ten, sources[4]], dim=1)
        feature = self._upconv4(mid_ten)

        mid_ten = self._conv_cls(feature)

        return mid_ten.permute(0, 2, 3, 1)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)


if __name__ == "__main__":
    main()
