from pathlib import Path
current_file_path = Path(__file__).resolve()
import sys
sys.path.append(str(current_file_path.parent.parent))

from torch.utils.data.dataset import Dataset
import project_config as pconfig


class CRAFTDataset(Dataset):

    """Docstring for CRAFTDataset. """

    def __init__(self,
                 data_root,
                 transform=None):
        """TODO: to be defined.

        :data_root: TODO
        :transform: TODO

        """
        Dataset.__init__(self)

        assert isinstance(data_root, str)
        assert Path(data_root).resolve().exists()
        self._data_root = Path(data_root).resolve()
        assert transform is not None
        self._transform = transform

    def train(self, mode, model=None, image_size=pconfig.TRAIN_IMAGE_SIZE):
        """TODO: Docstring for train.

        :returns: TODO

        """
        self._transform.train(mode, model, image_size)

    def val(self):
        """TODO: Docstring for val.
        :returns: TODO

        """
        self._transform.val()

    def test(self, image_size=pconfig.TEST_IMAGE_SIZE):
        """TODO: Docstring for test.
        :returns: TODO

        """
        self._transform.test(image_size)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    pass


if __name__ == "__main__":
    main()
