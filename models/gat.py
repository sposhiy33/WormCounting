'''
GNN forumulations for the crowd counting model. GNN variants: 
    1. GAT (Graph Attention Network)
    2. GCN (Graph Convolutional Network)
'''


import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

from util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .matcher import build_matcher_crowd
from .classification import *
from .mlp import *


class GATClassifier(nn.Module):
    '''
    GAT (Graph Attention Network) based network for proposal point classification (node classification).
    '''
    def __init__(self, backbone, num_classes, row=2, line=2, gnn_layers=3, knn=4):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes + 1 
        self.row = row
        self.line = line
        self.in_feat = 256

        # GAT paramas
        self.hidden_dim = 64   # Dimension of the hidden layers in the GAT (per head)
        self.num_heads = 4     # Number of attention heads in hidden layers
        self.num_gat_layers = gnn_layers # Number of GAT layers (determines message passing depth)
        self.dropout_rate = 0.6 # Dropout rate 
        self.k = knn  # Number of nearest neighbors for KNN graph construction
        self.loop= False

        num_anchor_points = row * line

        self.linear = Linear(
            in_feat=256,
            out_feat=self.num_classes,
        )


        self.gat_classifier = GATNodeClassifier(
            in_channels=self.in_feat,
            hidden_channels=self.hidden_dim,
            out_channels=self.num_classes,
            num_heads=self.num_heads,
            num_layers=self.num_gat_layers,
            dropout=self.dropout_rate,
        )

        self.anchor_points = AnchorPoints(
            pyramid_levels=[
                3,
            ],
            row=row,
            line=line,
        )

        self.fpn = Decoder(256, 512, 512)

    def create_edges(self, square_dim:int, batch_size:int):
        """
        square_dim -> number of proposal points in a square grid of initialized points
        knn -> number of points to make connections to; choices = [4,8]
        """
        # create edge index for the entire batch, based on specified number of neighbors
        if self.k == 4:
            # undirected graph, so (a,b) and (b,a) are included
            edge_pairs = []

            # batch index loop
            for b in range(batch_size):
                # point index loop
                # get start and end index of the anchor points (relative position in the batch)
                start = b * (square_dim**2)
                end = start + (square_dim**2)

                for p in np.arange(start, end):
                    # add in the 4 cardinal directions
                    if p+1 <= end : edge_pairs.append([p, p+1])
                    if p-1 >= start : edge_pairs.append([p,p-1])
                    if p-square_dim >= start : edge_pairs.append([p,p-square_dim])
                    if p+square_dim <= end : edge_pairs.append([p,p+square_dim])
                
                    if self.loop:
                        edge_pairs.append([p,p])

            # trasnform edge pair to tensor
            edge_pairs = torch.tensor(edge_pairs).type(torch.LongTensor)
            edge_pairs = edge_pairs.permute(1,0)
            
        elif self.k == 8:
            pass

        return edge_pairs

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)

        # generate the anchor points 
        batch_size = features[0].shape[0]
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1) 
        
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        feat_space = features_fpn[1]

        # prepare the feature space for the graph network
        batch_features = feat_space.reshape((batch_size, -1, feat_space.shape[2] * feat_space.shape[3]))
        batch_features = batch_features.permute(0,2,1)
        flat_features = batch_features.reshape((batch_size * batch_features.shape[1], -1))
        flat_coords = anchor_points.view(batch_size * anchor_points.shape[1], -1)

        # create the batch index for the KNN graph (seperate graph for each sample in batch)
        batch_index = torch.arange(batch_size).repeat_interleave(anchor_points.shape[1]) 

        # create edge index
        edge_index = self.create_edges(feat_space.size(2), batch_size)
        edge_index = edge_index.to(batch_features.device)

        data = Batch(x=flat_features, edge_index=edge_index, batch=batch_index)
        # pass through the GAT classifier
        classification = self.gat_classifier(data)
        # reshape GNN output back to batch formate
        classification = classification.view(batch_size, -1, self.num_classes)

        # prepare output
        output_coord = anchor_points
        output_class = classification
        
        out = {"pred_logits": output_class, "pred_points": output_coord}
        return out 


# --- Graph Attention Network (GAT) Model for Node Classification ---
class GATNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, dropout):
        super(GATNodeClassifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Input layer: Apply GAT and project to hidden_channels * num_heads
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        # Hidden layers: Take concatenated output from previous layer
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
        # Output layer: Project to final out_channels with a single head
        # The output layer of a node classification GAT typically uses heads=1
        self.convs.append(GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply hidden layers with ELU activation and dropout
        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = F.elu(x) # ELU is a common activation after GAT layers

        # Apply output layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        return x # Output logits for each node (proposal point)

def build_gat_classifier(args, training):
    
    backbone = build_backbone(args)

    model = GATClassifier(backbone, 
                          num_classes=args.num_classes, 
                          row=args.row, 
                          line=args.line,
                          gnn_layers=args.gnn_layers,
                          knn=args.knn)
    if not training:
        return model 
    
    weight_dict = {
        "loss_ce": args.label_loss_coef,
        "loss_point": args.point_loss_coef,
        "loss_dense": args.dense_loss_coef,
        "loss_distance": args.distance_loss_coef,
        "loss_count": args.count_loss_coef,
    }
 
    losses = args.loss
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(
        args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        ce_coef=args.ce_coef,
        map_res=args.map_res,
        gauss_kernel_res=args.gauss_kernel_res,
        losses=losses,
    )

    return model, criterion

