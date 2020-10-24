import sys
from pathlib import Path
from collections.abc import Iterable
import math
import cv2
from shapely.geometry import LineString, Polygon
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F

current_file_path = Path(__file__).resolve()
sys.path.append(str(current_file_path.parent.parent))
import project_config as pconfig
from models.craft import CRAFT


class Resize(object):

    """Docstring for Resize. """

    def __init__(self, output_size, interpolation=Image.BILINEAR):
        """TODO: to be defined.

        :output_size: TODO
        :interpolation: TODO

        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            assert output_size > 0
            assert not output_size % pconfig.SDR
        if isinstance(output_size, tuple):
            assert output_size[0] > 0 and output_size[1] > 0
            assert not output_size[0] % pconfig.SDR
            assert not output_size[1] % pconfig.SDR
        self._output_size = output_size
        self._interpolation = interpolation

    def __call__(self, example):
        assert isinstance(example, dict)
        assert 'image' in example
        # assert 'coordinates' in example
        # assert isinstance(example['coordinates'], np.ndarray)
        # assert example['coordinates'].ndim == 3
        # assert example['coordinates'].shape[0] == 2
        # assert example['coordinates'].shape[1] == 4

        image = example['image']

        if not F._is_pil_image(image):
            raise TypeError('image should be PIL Image. Got {}'.format(
                type(image)))
        if not (isinstance(self._output_size, int) or
                (isinstance(self._output_size, Iterable) and
                 len(self._output_size) == 2)):
            raise TypeError('Got inappropriate self._output_size '
                            'arg: {}'.format(self._output_size))

        input_width, input_height = image.size
        if isinstance(self._output_size, int):
            if (input_width <= input_height and
                    input_width == self._output_size) or\
                    (input_height <= input_width and
                     input_height == self._output_size):
                if not (input_height % pconfig.SDR or
                        input_width % pconfig.SDR):
                    return example
            if input_width < input_height:
                output_width = self._output_size
                output_height =\
                    int(self._output_size * input_height / input_width)
                padded_height =\
                    math.ceil(output_height / pconfig.SDR) * pconfig.SDR
                image = image.resize((output_width, output_height),
                                     self._interpolation)
                image = F.pad(image,
                              padding=(0, 0, 0,
                                       padded_height - output_height),
                              fill=0,
                              padding_mode='constant')
            else:
                output_height = self._output_size
                output_width =\
                    int(self._output_size * input_width / input_height)
                padded_width =\
                    math.ceil(output_width / pconfig.SDR) * pconfig.SDR
                image = image.resize((output_width, output_height),
                                     self._interpolation)
                image = F.pad(image,
                              padding=(0, 0,
                                       padded_width - output_width,
                                       0),
                              fill=0,
                              padding_mode='constant')
        else:
            output_width, output_height = self._output_size[::-1]
            image = image.resize(self._output_size[::-1], self._interpolation)
        example['image'] = image
        if 'coordinates' in example:
            example['coordinates'][0, ...] =\
                example['coordinates'][0, ...] * (output_width / input_width)
            example['coordinates'][1, ...] =\
                example['coordinates'][1, ...] * (output_height / input_height)
        if ('region_map' in example and
            (example['region_map'].shape[0] != int(image.size[1] / 2) or
             example['region_map'].shape[1] != int(image.size[0] / 2))):
            example['region_map'] =\
                cv2.resize(example['region_map'],
                           (int(image.size[0] / 2), int(image.size[1] / 2)))
        if ('affinity_map' in example and
            (example['affinity_map'].shape[0] != int(image.size[1] / 2) or
             example['affinity_map'].shape[1] != int(image.size[0] / 2))):
            example['affinity_map'] =\
                cv2.resize(example['affinity_map'],
                           (int(image.size[0] / 2), int(image.size[1] / 2)))
        if ('confidence_map' in example and
            (example['confidence_map'].shape[0] != int(image.size[1] / 2) or
             example['confidence_map'].shape[1] != int(image.size[0] / 2))):
            example['confidence_map'] =\
                cv2.resize(example['confidence_map'],
                           (int(image.size[0] / 2), int(image.size[1] / 2)))
        return example


class GenerateHeatMap(object):

    """Docstring for GenerateHeatMap. """

    def __init__(self, mode, model=None):
        """TODO: to be defined.

        :mode: TODO
        :model: TODO

        """
        assert isinstance(mode, str)
        assert mode in pconfig.AVAILABLE_MODES
        if mode in ['fully-supervised', 'val']:
            self._model = None
        elif mode == 'weakly-supervised':
            assert model is not None
            assert isinstance(model, CRAFT)
            self._model = model
        else:
            raise ValueError('Invalid mode!!!')
        self._mode = mode
        self._reference_map =\
            self.isotropic2d_gaussian_grayscale_map(
                pconfig.ISOTROPIC_GAUSSIAN_MAP_SIZE
            )
        # self._reference_map =\
        #     self.isotropic2d_gaussian_heat_map(
        #         self._reference_map
        #     )

    def __call__(self, example):
        """TODO: Docstring for __call__.

        :example: TODO
        :returns: TODO

        """
        assert isinstance(example, dict)
        assert 'image' in example
        assert 'coordinates' in example
        assert isinstance(example['coordinates'], np.ndarray)
        assert example['coordinates'].ndim == 3
        assert example['coordinates'].shape[0] == 2
        assert example['coordinates'].shape[1] == 4
        assert 'wordlengths' in example

        image = example['image']
        if not F._is_pil_image(image):
            raise TypeError('image should be PIL Image. Got {}'.format(
                type(image)))
        if self._mode == 'weakly-supervised':
            # NOTE [ https://arxiv.org/pdf/1904.01941.pdf ]
            # If the confidence score sconf(w) is below 0.5, the estimated
            # character bounding boxes should be neglected since they have
            # adverse effects when training the model. In this case, we assume
            # the width of the individual character is constant and compute the
            # character-level predictions by simply dividing the word region
            # R(w) by the number of characters l(w). Then, sconf(w) is set to
            # 0.5 to learn unseen appearances of texts.
            pass
        elif self._mode in ['fully-supervised', 'val']:
            charBB = example['coordinates']
            affinityBB = self.affinity_box(
                example['coordinates'],
                example['wordlengths'])

            region_map = self.heatmap(image, charBB)
            affinity_map = self.heatmap(image, affinityBB)
            # confidence_map = np.ones(
            #     (int(image.size[1] / 2),
            #      int(image.size[0] / 2)), dtype=np.uint8) * 255
            confidence_map = np.ones(
                (image.size[1],
                 image.size[0]), dtype=np.uint8) * 255
        else:
            raise ValueError
        example = {
            'image': image,
            'region_map': region_map,
            'affinity_map': affinity_map,
            'confidence_map': confidence_map
        }
        return example

    def isotropic2d_gaussian_grayscale_map(
            self,
            size=pconfig.ISOTROPIC_GAUSSIAN_MAP_SIZE
    ):
        """TODO: Docstring for isotropic2d_gaussian_grayscale_map.

        :size: TODO
        :returns: TODO

        """
        assert isinstance(size, int)
        assert size > 0
        # gaussian = lambda x: exp(-(1/2) * ((x)**2)) / (sqrt(2*pi))  # pdf
        def scaled_gaussian(x):
            return np.exp(-(1/2) * (x**2))  # not pdf

        # the first way:
        # gaussian_map = np.zeros((size, size), dtype=np.uint8)
        # for i in range(size):
        #     for j in range(size):
        #         distance_from_center =\
        #             np.linalg.norm(
        #                 np.array([i - size / 2, j - size / 2])
        #             )
        #         distance_from_center =\
        #             2.5 * distance_from_center / (size / 2)
        #         scaled_gaussian_prob =\
        #             scaled_gaussian(distance_from_center)
        #         gaussian_map[i, j] =\
        #             np.clip(np.round(scaled_gaussian_prob * 255), 0, 255)

        # the second way: much faster and more correct than the first way
        x, y = np.meshgrid(np.linspace(-2.5, 2.5, size),
                           np.linspace(-2.5, 2.5, size))
        distance_from_center =\
            np.linalg.norm(np.stack([x, y], axis=0), axis=0, keepdims=False)
        scaled_gaussian_prob = scaled_gaussian(distance_from_center)
        gaussian_map =\
            np.clip(np.round(scaled_gaussian_prob * 255),
                    0,
                    255).astype(np.uint8)
        return gaussian_map

    def isotropic2d_gaussian_heat_map(
            self,
            grayscale_map):
        """TODO: Docstring for isotropic2d_gaussian_heat_map.

        :size: TODO
        :returns: TODO

        """
        assert isinstance(grayscale_map, np.ndarray)
        assert grayscale_map.ndim == 2
        assert grayscale_map.dtype == np.uint8
        bgr_heatmap = cv2.applyColorMap(
            grayscale_map,
            cv2.COLORMAP_JET)
        return cv2.cvtColor(bgr_heatmap, cv2.COLOR_BGR2RGB)

    def affinity_box(self, charBB, wordlengths):
        """TODO: Docstring for affinity_box.

        :charBB: TODO
        :returns: TODO

        """
        exclusive_upper_wordindices =\
            np.cumsum(
                wordlengths,
                axis=0,
                dtype=np.int32
            )
        inclusive_lower_wordindices =\
            np.insert(exclusive_upper_wordindices, 0, 0)[:-1]

        charBB = [np.transpose(charBB[..., l:u], (2, 1, 0)) for (l, u) in zip(
            inclusive_lower_wordindices,
            exclusive_upper_wordindices)]

        affinitylengths = [length - 1 for length in wordlengths]
        exclusive_upper_affinityindices =\
            np.cumsum(
                affinitylengths,
                axis=0,
                dtype=np.int32
            )
        inclusive_lower_affinityindices =\
            np.insert(exclusive_upper_affinityindices, 0, 0)[:-1]
        affinityBB =\
            np.zeros((2, 4, sum(affinitylengths)), dtype=np.float32)
        for (boxes, wordlength, i) in zip(
                charBB,
                wordlengths,
                inclusive_lower_affinityindices):
            if wordlength == 1:
                continue
            for j in range(wordlength):
                box = boxes[j, ...]
                # intersection of two diagonals
                diagonal1 = LineString([box[0], box[2]])
                diagonal2 = LineString([box[1], box[3]])
                quad_center = diagonal1.intersection(diagonal2).coords[:][0]
                upper_triangle = Polygon([box[0], box[1], quad_center])
                lower_triangle = Polygon([box[2], box[3], quad_center])
                upper_centroid = upper_triangle.centroid.coords[:][0]
                lower_centroid = lower_triangle.centroid.coords[:][0]
                if j == 0:
                    affinityBB[:, 0, i + j] = upper_centroid
                    affinityBB[:, 3, i + j] = lower_centroid
                elif j == wordlength - 1:
                    affinityBB[:, 1, i + j - 1] = upper_centroid
                    affinityBB[:, 2, i + j - 1] = lower_centroid
                else:
                    affinityBB[:, 0, i + j] = upper_centroid
                    affinityBB[:, 3, i + j] = lower_centroid
                    affinityBB[:, 1, i + j - 1] = upper_centroid
                    affinityBB[:, 2, i + j - 1] = lower_centroid
        return affinityBB

    def heatmap(self, image, boxes):
        """TODO: Docstring for heatmap.

        :image: TODO
        :boxes: TODO
        :returns: TODO

        """
        def perspective_transform(isotropic_map, box, image_size):
            # dst = np.array([
            #     [0, 0],
            #     [isotropic_map.shape[1] - 1, 0],
            #     [isotropic_map.shape[1] - 1, isotropic_map.shape[0] - 1],
            #     [0, isotropic_map.shape[0] - 1]], dtype=np.float32)
            map_height = isotropic_map.shape[0]
            map_width = isotropic_map.shape[1]
            dst = np.array([
                [int(0.15 * map_width), int(0.15 * map_height)],
                [int(0.85 * (map_width - 1)), int(0.15 * map_height)],
                [int(0.85 * (map_width - 1)), int(0.85 * (map_height - 1))],
                [int(0.15 * map_width), int(0.85 * (map_height - 1))]],
                           dtype=np.float32)

            transformation_matrix = cv2.getPerspectiveTransform(dst, box)
            warped_map = cv2.warpPerspective(
                isotropic_map,
                transformation_matrix,
                image_size)
            out_map = warped_map.astype(np.uint8)
            assert (warped_map - out_map == 0).all()
            return out_map
        num_boxes = boxes.shape[2]
        heatmaps = []
        for i in range(num_boxes):
            box = boxes[..., i]
            warped_map = perspective_transform(
                self._reference_map,
                np.transpose(box, (1, 0)),
                image.size
            )
            heatmaps.append(warped_map)
        heatmaps = np.stack(heatmaps, axis=0)
        heatmap = np.max(heatmaps, axis=0)

        # heatmap =\
        #     cv2.resize(heatmap,
        #                (int(image.size[0] / 2), int(image.size[1] / 2)))
        return heatmap


class ToTensor(object):

    """Docstring for ToTensor. """

    def __call__(self, example):
        """TODO: Docstring for __call__.

        :example: TODO
        :returns: TODO

        """
        if 'image' in example:
            example['image'] = F.to_tensor(example['image'])
        if 'region_map' in example:
            example['region_map'] = torch.from_numpy(example['region_map'])
        if 'affinity_map' in example:
            example['affinity_map'] =\
                torch.from_numpy(example['affinity_map'])
        if 'confidence_map' in example:
            example['confidence_map'] =\
                torch.from_numpy(example['confidence_map'])
        # if 'wordlengths' in example:
        #     example['wordlengths'] = torch.tensor(example['wordlengths'])
        return example

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):

    """Docstring for Normalize. """

    def __init__(self, mean, std, inplace=False):
        self._mean = mean
        self._std = std
        self._inplace = inplace

    def __call__(self, example):
        """TODO: Docstring for __call__.

        :example: TODO
        :returns: TODO

        """
        example['image'] =\
            F.normalize(example['image'],
                        self._mean,
                        self._std,
                        self._inplace)

        def normalize_minmax(tensor, nmin=0, nmax=255):
            # cast to float in order to avoid overflow
            assert tensor.dtype == torch.uint8
            tensor = tensor.to(dtype=torch.float32)
            omin = torch.min(tensor)
            omax = torch.max(tensor)
            return torch.clamp(
                torch.round(
                    torch.true_divide(
                        (tensor - omin) * nmax, (omax - omin)) + nmin).to(
                            dtype=torch.uint8),
                nmin, nmax)
        if 'region_map' in example:
            example['region_map'] = normalize_minmax(example['region_map'])
        if 'affinity_map' in example:
            example['affinity_map'] = normalize_minmax(example['affinity_map'])
        if 'confidence_map' in example:
            if (torch.max(example['confidence_map']) ==
                    torch.min(example['confidence_map'])):
                # all pixels are equal ==> not affected by geometry
                # transformations ==> no need for normalization
                assert torch.max(example['confidence_map']) >= 0
                assert torch.max(example['confidence_map']) <= 255
            else:
                example['confidence_map'] =\
                    normalize_minmax(
                        example['confidence_map'])
            example['confidence_map'] =\
                torch.true_divide(example['confidence_map'], 255)

        # if 'region_map' in example:
        #     example['region_map'] =\
        #         cv2.normalize(
        #             example['region_map'],
        #             None,
        #             0, 255, cv2.NORM_MINMAX)
        # if 'affinity_map' in example:
        #     example['affinity_map'] =\
        #         cv2.normalize(
        #             example['affinity_map'],
        #             None,
        #             0, 255, cv2.NORM_MINMAX)
        # if 'confidence_map' in example:
        #     example['confidence_map'] =\
        #         cv2.normalize(
        #             example['confidence_map'],
        #             None,
        #             0, 255, cv2.NORM_MINMAX)
        return example

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self._mean, self._std)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """


if __name__ == "__main__":
    main()
