import datetime
import torch
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
from torch.utils.data.dataset import Subset

from datasets.synthtext.dataset import SynthTextDataset
from datasets.synthtext.transforms import Transform
from models.craft import CRAFT
import project_config as pconfig


# 1. Initialize datasets: dataset1, dataset2, ...
# 2. Split dataset into train, test, val set: dataset1_train, dataset2_val, ...
# 3. Group train datasets, val datasets: dataset_fs(ws)train, dataset_val
#
# NOTE [ Training procedure : https://arxiv.org/pdf/1904.01941.pdf ]
# - step1: train for 50k iterations with SynthText dataset
# - step2: fine-tune with each benchmark dataset (ICDAR2015, ...): ignore 'DO
# NOT CARE' text regions, train with ratio (1 SynthText : 5 Benchmark)
# - optimizer: Adam
# - if multi-GPU training: one GPU for generating pseudo GTs and saving them to
# memory, others for training
# - On-line Hard Negative Mining: ratio (1 : 3)


def training_loop(
        num_epochs,
        train_loader,
        model,
        loss_function,
        optimizer,
        device):
    """TODO: Docstring for training_loop.

    :num_epochs: TODO
    :train_loader: TODO
    :model: TODO
    :loss_function: TODO
    :optimizer: TODO
    :device: TODO
    :returns: TODO

    """
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        for i, examples in enumerate(train_loader):
            images = examples['image']
            region_maps = examples['region_map']
            affinity_maps = examples['affinity_map']
            confidence_maps = examples['confidence_map']

            images = images.to(device=device)
            region_maps = region_maps.to(device=device)
            affinity_maps = affinity_maps.to(device=device)
            confidence_maps = confidence_maps.to(device=device)

            # NOTE [ Data Fetching ]
            outputs = model(images)
            loss = loss_function(
                outputs,
                (region_maps, affinity_maps, confidence_maps),
                device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i == 0 or (i + 1) % 50 == 0:
                print('Epoch: {} - Iteration: {} - Loss: {}'.format(
                    epoch, i + 1, loss
                ))
            if (i + 1) % 50 == 0:
                print('Saving model to ./models/craft.gt')
                torch.save(model.state_dict(), './models/craft.gt')
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch: {} - Training loss: {}'.format(
                datetime.datetime.now(),
                epoch,
                train_loss / len(train_loader)))
    print('Saving model to ./models/craft.gt')
    torch.save(model.state_dict(), './models/craft.gt')


def loss_function(predictions, labels, device):
    """TODO: Docstring for loss_function.

    :predictions: TODO
    :labels: TODO
    :device: TODO
    :returns: TODO

    """
    tregion_maps, taffinity_maps, confidence_maps = labels
    # tregion_maps = torch.true_divide(tregion_maps, 255.)
    # taffinity_maps = torch.true_divide(taffinity_maps, 255.)
    targets = torch.stack([tregion_maps, taffinity_maps], dim=3)

    # normal loss
    # loss =\
    #     torch.mean(
    #         confidence_maps * torch.sum(
    #             torch.pow(predictions - targets, 2),
    #             dim=-1))

    # loss with online hard negative mining
    # print(predictions.shape)
    # print(tregion_maps.shape)
    # print(taffinity_maps.shape)
    # print(confidence_maps.shape)
    # print(targets.shape)
    threshold_tensor =\
        torch.tensor(
            [pconfig.REGION_THRESHOLD,
             pconfig.AFFINITY_THRESHOLD],
            device=device).view(
                1, 1, 1, 2)
    # print(threshold_tensor)
    pos_masks = torch.gt(targets, threshold_tensor)  # (BS, H, W, 2)
    neg_masks = torch.logical_not(pos_masks)  # (BS, H, W, 2)
    # print(pos_masks.shape)
    # print(neg_masks.shape)
    # print()
    num_pos = torch.sum(torch.sum(pos_masks, dim=1), dim=1)  # (BS, 2)
    num_neg, _ = torch.min(torch.stack(
        [pconfig.NEG_POS_RATIO * num_pos,
         torch.sum(torch.sum(neg_masks, dim=1), dim=1)],
        dim=2),
                        dim=2)  # (BS, 2)
    # print(num_pos)
    # print(torch.sum(torch.sum(neg_masks, dim=1), dim=1))
    # print(num_neg)
    min_prediction_values, _ =\
        torch.min(
            torch.min(
                predictions, dim=1
            )[0], dim=1
        )
    neg_maps =\
        ((predictions * neg_masks) +
         (torch.unsqueeze(
             torch.unsqueeze(
                 min_prediction_values - 1.,
                 dim=1),
             dim=1) * pos_masks))
    # print(neg_maps.shape)
    # print(predictions.dtype)
    # print(neg_masks.dtype)
    heat_values, _ =\
        torch.sort(
            neg_maps.view(pconfig.BATCH_SIZE, -1, 2),
            dim=1,
            descending=True)
    # print(heat_values)
    # print(heat_values.shape)
    chosen_min_neg_value =\
        torch.squeeze(
            torch.gather(
                heat_values,
                1,
                torch.unsqueeze(num_neg - 1, dim=1)),
            dim=1)
    # print(chosen_min_neg_value)
    # assert 1 == 2
    hard_neg_masks = neg_maps >= torch.unsqueeze(
        torch.unsqueeze(
            chosen_min_neg_value,
            dim=1),
        dim=1)

    masks = torch.logical_or(pos_masks, hard_neg_masks)
    # num_examples = torch.sum(num_pos + num_neg)
    num_examples = torch.sum(masks)
    loss =\
        torch.true_divide(
            torch.sum(
                confidence_maps * torch.sum(
                    torch.pow(masks * (predictions - targets),
                              2),
                    dim=-1)),
            num_examples)

    return loss


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    transform = Transform(mode='fully-supervised', model=None)
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
    train_loader = DataLoader(
        dataset=subdataset_synthtext,
        batch_size=pconfig.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=pconfig.DATA_LOADING_NUM_WORKERS
    )

    device = torch.device('cuda')
    craft = CRAFT(pretrained=True)
    # if Path('./models/craft.gt').resolve().exists:
    #     print('Model exists! Loading model ...')
    #     craft.load_state_dict(torch.load(
    #         './models/craft.gt',
    #         map_location=device
    #     ))
    craft.train()
    craft.to(device=device)
    training_loop(
        num_epochs=400,
        train_loader=train_loader,
        model=craft,
        loss_function=loss_function,
        optimizer=optim.Adam(params=craft.parameters(), lr=0.5 * 1e-3,
                             weight_decay=1e-1),
        device=device
    )


if __name__ == "__main__":
    main()
