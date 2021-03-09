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

path_to_data = '/home/jcejudo/interface_dataset/visualization/static/training_data'
dest_path = '/home/jcejudo/visual_recommendation'
num_workers = 8
batch_size = 32
seed = 1
max_epochs = 100
input_size = 100
num_ftrs = 64

# %%
# Let's set the seed for our experiments
pl.seed_everything(seed)



collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5,
    cj_prob=0.0,
    random_gray_scale=0.0
)

# We create a torchvision transformation for embedding the dataset after 
# training
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
# -----------------------
# Create a ResNet backbone and remove the classification head


resnet = lightly.models.ResNetGenerator('resnet-18')
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
    nn.AdaptiveAvgPool2d(1)
)

# create the SimCLR model using the newly created backbone
model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)

# %%
# We now use the SelfSupervisedEmbedding class from the embedding module.
# First, we create a criterion and an optimizer and then pass them together
# with the model and the dataloader.
criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

# %% 
# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0
encoder.train_embedding(gpus=gpus, 
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)

# %%
# Now, let's make sure we move the trained model to the gpu if we have one
device = 'cuda' if gpus==1 else 'cpu'
encoder = encoder.to(device)

# %%
# We can use the .embed method to create an embedding of the dataset. The method
# returns a list of embedding vectors as well as a list of filenames.
embeddings, labels, fnames = encoder.embed(dataloader_test, device=device)
embeddings = normalize(embeddings)

model_filename = 'model.pth'
torch.save(encoder.model.state_dict(),os.path.join(dest_path,model_filename))

import lightly.utils.io as io
embeddings_filename = 'embeddings.csv'
io.save_embeddings(os.path.join(dest_path,embeddings_filename), embeddings, labels, fnames)



