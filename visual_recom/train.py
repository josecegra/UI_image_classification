import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import argparse
import lightly.utils.io as io







def main():
    input_size = 128
    num_ftrs = 32
    batch_size = 64
    num_workers = 8
    seed = 1

    pl.seed_everything(seed)

    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        vf_prob=0.5,
        rr_prob=0.5,
        cj_prob=0.0,
        random_gray_scale=0.0
    )


    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    dataset_train_simclr = lightly.data.LightlyDataset(
        input_dir=path_to_data
    )

    dataset_test = lightly.data.LightlyDataset(
        input_dir=path_to_data,
        transform=test_transforms
    )


    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )


    # %%
    # Create the SimCLR model

    resnet = lightly.models.ResNetGenerator('resnet-18')
    last_conv_channels = list(resnet.children())[-1].in_features
    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1)
    )

    model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)

    criterion = lightly.loss.NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    encoder = lightly.embedding.SelfSupervisedEmbedding(
        model,
        criterion,
        optimizer,
        dataloader_train_simclr
    )


    gpus = 1 if torch.cuda.is_available() else 0
    encoder.train_embedding(gpus=gpus, 
                            progress_bar_refresh_rate=100,
                            max_epochs=max_epochs)


    device = 'cuda' if gpus==1 else 'cpu'
    encoder = encoder.to(device)

    embeddings, labels, fnames = encoder.embed(dataloader_test, device=device)
    embeddings = normalize(embeddings)

    torch.save(encoder.model.state_dict(),os.path.join(dest_path,model_filename))

    io.save_embeddings(os.path.join(dest_path,embeddings_filename), embeddings, labels, fnames)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)


    path_to_data = '/home/jcejudo/interface_dataset/visualization/static/training_data'
    dest_path = '/home/jcejudo/visual_recommendation'

    model_filename = 'model.pth'
    embeddings_filename = 'embeddings.csv'


    max_epochs = 100

    main()




