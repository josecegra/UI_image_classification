
from flask import Flask, jsonify, request
import sys
sys.path.append('../')

import lightly.utils.io as io
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
import json
import argparse

def load_model(model_path,data_path,embeddings_path,n_neighbors):

    #load embeddings
    embeddings, labels, filenames = io.load_embeddings(embeddings_path)

    #fit nearest neighbours
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    #load model
    batch_size = 64
    num_workers = 2
    num_ftrs = 32
    input_size = 128

    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        vf_prob=0.5,
        rr_prob=0.5,
        cj_prob=0.0,
        random_gray_scale=0.0
    )
    dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=data_path
    )

    dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
    )
 
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = encoder.to(device)
    if device == 'cpu':
        encoder.model.load_state_dict(torch.load(model_path,map_location=device))
    else:
        encoder.model.load_state_dict(torch.load(model_path))
    encoder.model.eval()
    
    return encoder,filenames, indices, distances



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--embeddings_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--n_neighbors', required=False)

    parser.add_argument('--endpoint_name', required=False)
    parser.add_argument('--host', required=False)
    parser.add_argument('--port', required=False)
    args = parser.parse_args()

    if args.endpoint_name:
        endpoint_name = args.endpoint_name
    else:
        endpoint_name = 'img_recommendation'

    if args.port:
        port = int(args.port)
    else:
        port = 5000

    if args.host:
        host = args.host
    else:
        host = '0.0.0.0'

    if args.n_neighbors:
        n_neighbors = int(args.n_neighbors)
    else:
        n_neighbors = 20

    
    #n_neighbors = 20
    #model_path = '/home/jcejudo/visual_recommendation/model_3000.pth'
    #embeddings_path = '/home/jcejudo/visual_recommendation/embeddings_3000.csv'

    #data_path = '/home/jcejudo/interface_dataset/visualization/static/training_data_3000'

    model,filenames,indices,distances = load_model(args.model_path,args.data_path,args.embeddings_path,n_neighbors)

    app = Flask(__name__)

    @app.route(f'/{endpoint_name}', methods=['POST','GET'])
    def predict():
        if request.method == 'POST':
            output_dict = {'fnames':[]}
            file = None
            for k in request.files.keys():
                file = request.files[k]
            
            if file:
                img_bytes = file.read()   
                pred_filename = file.filename
                matches = [s for s in filenames if pred_filename in s]
                if matches:
                    idx = filenames.index(matches[0])
                    for i,(neighbor_idx,d) in enumerate(zip(indices[idx],distances[idx])):
                        if i>0:
                            output_dict['fnames'].append(filenames[neighbor_idx])
                
            return jsonify(output_dict)

        if request.method == 'GET':
            return jsonify({})

    app.run(host=host, port=port)


