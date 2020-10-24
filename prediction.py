import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data.dataset import Subset

from datasets.synthtext.dataset import SynthTextDataset
from datasets.synthtext.transforms import Transform
from models.craft import CRAFT


def main():
    """TODO: Docstring for main.

    :arg1: TODO
    :returns: TODO

    """
    transform = Transform(mode='test')
    dataset_synthtext = SynthTextDataset(
        data_root='/home/loinguyenvan/Projects/OneDriveHUST/Datasets/'
        'SynthText/',
        transform=transform
    )
    # NOTE [ Data Sharding or Data Splitting ]
    subdataset_synthtext =\
        Subset(dataset=dataset_synthtext,
               indices=list(range(2000)))

    # NOTE [ Data Sampling ]
    # NOTE [ Data Batching or Data Collating ]
    # NOTE [ Data Shuffling ]
    # NOTE [ Data Parallelization ]
    data_loader = DataLoader(
        dataset=subdataset_synthtext,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=1
    )

    device = torch.device('cuda')
    craft = CRAFT(pretrained=False)
    if Path('./models/craft.gt').resolve().exists:
        print('Model exists! Loading model ...')
        craft.load_state_dict(torch.load(
            './models/craft.gt',
            map_location=device
        ))
    craft.eval()
    craft.to(device=device)

    for i, data in enumerate(data_loader):
        if i > 0:
            break
        image = data['image']
        image = image.to(device=device)
        output = craft(image)
        print(output)
        print(output.shape)


if __name__ == "__main__":
    main()
