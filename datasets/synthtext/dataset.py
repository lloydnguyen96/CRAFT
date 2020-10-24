from pathlib import Path
current_file_path = Path(__file__).resolve()
import sys
sys.path.append(str(current_file_path.parent.parent.parent))
from datasets.datasets import CRAFTDataset
from datasets.synthtext.transforms import Transform

import torch
from torchvision import transforms
from scipy.io import loadmat
import numpy as np
import re
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class SynthTextDataset(CRAFTDataset):

    """Docstring for SynthTextDataset. """

    def __init__(self,
                 data_root,
                 transform=Transform(mode='fully-supervised', model=None)):
        """TODO: to be defined.

        :data_root: TODO
        :transform: TODO

        """
        CRAFTDataset.__init__(self, data_root, transform)
        self._gt = loadmat(str(self._data_root.joinpath('gt.mat')))
        assert isinstance(self._gt, dict)

    def __len__(self):
        """TODO: Docstring for __len__.
        :returns: TODO

        """
        return self._gt['imnames'].shape[1]

    # NOTE [ Data Loading ]
    def __getitem__(self, i):
        """TODO: Docstring for __getitem__.

        :i: TODO
        :returns: TODO

        """
        charBB = self._gt['charBB'][0, i]
        wordBB = self._gt['wordBB'][0, i]
        imname = self._gt['imnames'][0, i]
        txt = self._gt['txt'][0, i]
        assert isinstance(charBB, np.ndarray)
        assert isinstance(wordBB, np.ndarray)
        assert isinstance(imname, np.ndarray)
        assert isinstance(txt, np.ndarray)

        assert charBB.dtype == np.float64
        assert wordBB.dtype == np.float32

        assert charBB.ndim == 3
        assert charBB.shape[0] == 2
        assert charBB.shape[1] == 4
        num_chars = charBB.shape[2]

        assert wordBB.ndim == 3 or wordBB.ndim == 2
        assert wordBB.shape[0] == 2
        assert wordBB.shape[1] == 4
        num_words = wordBB.shape[2] if wordBB.ndim == 3 else 1

        assert imname.ndim == 1
        assert imname.shape[0] == 1
        assert isinstance(imname[0], str)
        assert re.findall('[0-9][0-9]?[0-9]?/[a-zA-Z0-9_+]*.jpg',
                          imname[0])[0] == imname[0]

        assert txt.ndim == 1
        num_text_instances = txt.shape[0]
        words = []
        for j in range(num_text_instances):
            text_instance = txt[j]
            assert isinstance(text_instance, str)
            words += [word.strip() for word in text_instance.split()]

        assert num_words == len(words)
        wordlengths = [len(w) for w in words]
        assert num_chars == sum(wordlengths)

        image_path = self._data_root.joinpath(imname[0])
        image = Image.open(str(image_path))

        # NOTE [ Data Preprocessing ]
        # NOTE [ Data Sanitization ]
        # Input specifications:
        # - image: PIL RGB image
        # - coordinates:
        # + numpy.ndarray
        # + shape (2, 4, num_objects)
        # + np.float32 (bboxes' coordinates need to be of compatible type with
        # cv2 methods like cv2.getPerspectiveTransform)
        # - wordlengths: list of lengths of words
        if self._transform._mode == 'weakly-supervised':
            example = {
                'image': image,
                'coordinates': wordBB,
                'wordlengths': wordlengths
            }
        elif self._transform._mode in ['fully-supervised', 'val']:
            example = {
                'image': image,
                'coordinates': charBB.astype(np.float32),
                'wordlengths': wordlengths
            }
        elif self._transform._mode == 'test':
            example = {
                'image': image
            }
        else:
            raise ValueError

        example = self._transform(example)

        # add arbitrary number of features, labels for your need
        example['imname'] = imname[0]
        # END [ Data Preprocessing ]
        return example


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    # ----- Mean and Standard Deviation Caculation -----
    # transform = Transform(mode='fully-supervised', model=None)
    # dts = SynthTextDataset(
    #     data_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/'
    #     'SynthText/',
    #     transform=transforms.Compose([transforms.ToTensor()])
    # )
    # num_pixels = 0
    # running_sum = 0.
    # for i in tqdm(range(0, len(dts))):
    #     num_pixels += (dts[i].shape[1] * dts[i].shape[2])
    #     running_sum += dts[i].sum(dim=-1).sum(dim=-1)
    # mean = torch.true_divide(running_sum, num_pixels)
    # print(mean)
    # for i in tqdm(range(0, len(dts))):
    #     num_pixels += (dts[i].shape[1] * dts[i].shape[2])
    #     running_sum +=\
    #         torch.pow(dts[i] - mean.view(3, 1, 1),
    #                   2).sum(dim=-1).sum(dim=-1)
    # std = torch.sqrt(torch.true_divide(running_sum, (num_pixels - 1)))
    # print(std)

    # ----- Resize and GenerateHeatMap Test -----
    # transform = Transform(mode='fully-supervised', model=None)
    # dts = SynthTextDataset(
    #     data_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/'
    #     'SynthText/',
    #     transform=transform
    # )
    # example = dts[0]
    # image = example['image']
    # region_map = example['region_map']
    # affinity_map = example['affinity_map']
    # confidence_map = example['confidence_map']
    # print(image.size)

    # print(type(region_map))
    # print(region_map.shape)
    # print(region_map.dtype)

    # print(type(affinity_map))
    # print(affinity_map.shape)
    # print(affinity_map.dtype)

    # print(type(confidence_map))
    # print(confidence_map.shape)
    # print(confidence_map.dtype)

    # print(region_map.max())
    # print(region_map.min())
    # print(affinity_map.max())
    # print(affinity_map.min())
    # print(confidence_map.max())
    # print(confidence_map.min())
    # if region_map.ndim != 3:
    #     region_map = np.expand_dims(region_map, axis=-1)
    # if affinity_map.ndim != 3:
    #     affinity_map = np.expand_dims(affinity_map, axis=-1)
    # image = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
    # blended_region =\
    #     (np.array(image) * 0.3 + region_map * 0.7).astype(np.uint8)
    # blended_affinity =\
    #     (np.array(image) * 0.3 + affinity_map * 0.7).astype(np.uint8)
    # fig, ((ax1, ax2), (ax3, ax4)) =\
    #     plt.subplots(
    #         nrows=2,
    #         ncols=2,
    #         sharex=True,
    #         sharey=True,
    #     )
    # ax1.set_title('image')
    # ax2.set_title('region_map')
    # ax3.set_title('confidence_map')
    # ax4.set_title('affinity_map')
    # ax1.imshow(image)
    # ax2.imshow(blended_region)
    # ax3.imshow(confidence_map, cmap='gray', vmin=0, vmax=255)
    # ax4.imshow(blended_affinity)
    # plt.show()

    # ----- ToTensor Test -----
    # transform = Transform(mode='fully-supervised', model=None)
    # dts = SynthTextDataset(
    #     data_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/'
    #     'SynthText/',
    #     transform=transform
    # )
    # example = dts[0]
    # assert len(example) == 5
    # image = example['image']
    # print(type(image))
    # print(image.shape)
    # print(image.dtype)
    # print(image.device)
    # print(image.requires_grad)
    # print(torch.max(image))
    # print(torch.min(image))

    # region_map = example['region_map']
    # print(type(region_map))
    # print(region_map.shape)
    # print(region_map.dtype)
    # print(region_map.device)
    # print(region_map.requires_grad)

    # affinity_map = example['affinity_map']
    # print(type(affinity_map))
    # print(affinity_map.shape)
    # print(affinity_map.dtype)
    # print(affinity_map.device)
    # print(affinity_map.requires_grad)

    # confidence_map = example['confidence_map']
    # print(type(confidence_map))
    # print(confidence_map.shape)
    # print(confidence_map.dtype)
    # print(confidence_map.device)
    # print(confidence_map.requires_grad)

    # ----- Normalize Test -----
    # transform = Transform(mode='fully-supervised', model=None)
    # dts = SynthTextDataset(
    #     data_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/'
    #     'SynthText/',
    #     transform=transform
    # )
    # example = dts[0]
    # assert len(example) == 5
    # image = example['image']
    # print(torch.max(image))
    # print(torch.min(image))

    # transform = Transform(mode='fully-supervised', model=None)
    # dts = SynthTextDataset(
    #     data_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/'
    #     'SynthText/',
    #     transform=transform
    # )
    # for i in tqdm(range(0, len(dts))):
    #     if i != 0:
    #         break
    #     example = dts[i]
    #     print(torch.max(example['region_map']))
    #     print(torch.min(example['region_map']))
    #     print(torch.max(example['affinity_map']))
    #     print(torch.min(example['affinity_map']))
    #     print(torch.max(example['confidence_map']))
    #     print(torch.min(example['confidence_map']))
    #     print(example['confidence_map'].dtype)


if __name__ == "__main__":
    main()
