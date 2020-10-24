import sys
from pathlib import Path
from torchvision import transforms

current_file_path = Path(__file__).resolve()
sys.path.append(str(current_file_path.parent.parent))
import project_config as pconfig
import datasets.synthtext.config as dconfig
from datasets.transforms import Resize, GenerateHeatMap, ToTensor, Normalize
from models.craft import CRAFT


class Transform(object):

    """Docstring for Transform. """

    def __init__(self, mode, model=None):
        """TODO: to be defined.

        :mode: TODO
        :model: TODO

        """
        assert isinstance(mode, str)
        assert mode in pconfig.AVAILABLE_MODES
        if mode in ['fully-supervised', 'val', 'test']:
            assert model is None
        elif mode == 'weakly-supervised':
            assert model is not None
            assert isinstance(model, CRAFT)
        else:
            raise ValueError
        self._mode = mode

        # NOTE [ Data Transformation ]
        # 1. Must-have steps:
        # 1.1. Heatmap generation (region_map, affinity_map, confidence_map)
        # ----- NOTE [ Data Augmentation ] -----
        # 2. A permutation of color variations:
        # - Brightness
        # - Contrast
        # - Saturation
        # - Hue
        # - ...
        # 3. A permutation of geometry transformations:
        # - Horizontal flipping
        # - Cropping
        # - Scaling (input image size reduces)
        # - Padding (input image size doesn't change)
        # - Rotation
        # - Translation
        # - Shearing
        # - Resizing (variable input)
        # ----- END [ Data Augmentation ] -----
        # 1.2. ToTensor
        # 1.3. Normalization (mean-std, ...)

        if self._mode in ['fully-supervised', 'weakly-supervised', 'val']:
            self._generate_heatmap = GenerateHeatMap(self._mode, model)
            self._resize = Resize(pconfig.TRAIN_IMAGE_SIZE)
        elif self._mode == 'test':
            self._resize = Resize(pconfig.TEST_IMAGE_SIZE)
        else:
            raise ValueError
        self._to_tensor = ToTensor()
        self._normalize =\
            Normalize(mean=dconfig.MEAN,
                      std=dconfig.STD,
                      inplace=False)

        # transform1 = transforms.RandomOrder([color variations])
        # or
        # transform1 = transforms.ColorJitter(
        #                   brightness=(),
        #                   contrast=(),
        #                   saturation=(),
        #                   hue=())
        # transform2 = transforms.RandomOrder([geometry transformations])
        # self._transform3 =\
        #     transforms.Compose([Resize(pconfig.TRAIN_IMAGE_SIZE),
        #                         GenerateHeatMap(mode, model),
        #                         ToTensor(),
        #                         Normalize(
        #                             mean=dconfig.MEAN,
        #                             std=dconfig.STD,
        #                             inplace=False
        #                         )])

    def __call__(self, example):
        """TODO: Docstring for __call__.

        :example: TODO
        :returns: TODO

        """
        if self._mode in ['weakly-supervised', 'fully-supervised']:
            transform = transforms.Compose([
                self._generate_heatmap,
                self._resize,
                self._to_tensor,
                self._normalize
            ])
            return transform(example)
        if self._mode == 'val':
            transform = transforms.Compose([
                self._generate_heatmap,
                self._resize,
                self._to_tensor,
                self._normalize
            ])
            return transform(example)
        if self._mode == 'test':
            transform = transforms.Compose([
                self._resize,
                self._to_tensor,
                self._normalize
            ])
            return transform(example)
        raise ValueError

    def train(self, mode, model=None, image_size=pconfig.TRAIN_IMAGE_SIZE):
        """TODO: Docstring for train.

        :mode: TODO
        :model: TODO
        :image_size: TODO
        :returns: TODO

        """
        assert isinstance(mode, str)
        assert mode in ['fully-supervised', 'weakly-supervised']
        if mode in ['fully-supervised']:
            assert model is None
        elif mode == 'weakly-supervised':
            assert model is not None
            assert isinstance(model, CRAFT)
        else:
            raise ValueError
        self._mode = mode
        self._generate_heatmap = GenerateHeatMap(mode, model)
        self._resize = Resize(image_size)

    def val(self):
        """TODO: Docstring for val.
        :returns: TODO

        """
        self._mode = 'val'
        self._generate_heatmap = GenerateHeatMap('val', None)

    def test(self, image_size=pconfig.TEST_IMAGE_SIZE):
        """TODO: Docstring for test.

        :image_size: TODO
        :returns: TODO

        """
        self._mode = 'test'
        self._resize = Resize(image_size)
