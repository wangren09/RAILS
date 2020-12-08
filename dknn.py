#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from collections import Counter
import math
import numpy as np
from tqdm import trange
from sklearn.neighbors import NearestNeighbors

import torch


class Calibration():
    def __init__(self, x_cali, y_cali):
        self.x = x_cali
        self.y = y_cali
        self.n_sample = len(y_cali)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_sample


class DKNN():
    def __init__(self, model, device, x_train, y_train,
                 batch_size, n_neighbors, n_embs):
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.input_shape = x_train.shape[1:]

        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.n_embs = n_embs

        self.rng = np.random.RandomState(1)

        x_train, y_train, self.calib_dataset = self._sep_data(x_train, y_train)

        self.conv_features = self._build_rep_train(x_train)
        self.train_targets = y_train
        self.neighs = self._build_neighs()
        self.alpha_calib_sum = self._build_calibration()

    def _sep_data(self, xx, yy):
        calieps = 720
        cali_indice = np.array([i for i in range(0, xx.size(0), math.floor(xx.size(0) / calieps))])
        cali_images = np.zeros((calieps,) + self.input_shape)
        cali_labels = self.rng.randint(0, 10, calieps)
        for i in range(calieps):
            cali_images[i] = xx[cali_indice[i]]
            cali_labels[i] = yy[i]
        x_train = np.concatenate((xx, cali_images), axis=0)
        y_train = np.concatenate((yy, cali_labels), axis=0)
        x_train = np.delete(x_train, cali_indice, axis=0)
        y_train = np.delete(y_train, cali_indice, axis=0)
        calib_dataset = Calibration(cali_images.astype(np.float32), cali_labels)
        return x_train, y_train, calib_dataset

    def _build_rep_train(self, xx_batch_train, batchs=2000):
        print('Building the feature spaces from the selected set.')
        xhs = [[] for _ in range(self.n_embs)]
        for i in trange(0, len(xx_batch_train), batchs):
            xx = xx_batch_train[i:i + batchs]
            *out_convs, _ = self.model(torch.Tensor(xx).to(self.device))
            for j, out_conv in enumerate(out_convs):
                out_conv = out_conv.contiguous().reshape(out_conv.size(0), -1).cpu().detach().numpy()
                xhs[j].append(out_conv)
        xhs = [np.concatenate(out_convs, axis=0) for out_convs in xhs]
        print("Finished.")
        return xhs

    def _build_calibration(self):
        print('Building calibration set.')
        sequential_calib_loader = torch.utils.data.DataLoader(
            self.calib_dataset,
            shuffle    = False,
            batch_size = self.batch_size
        )
        alpha_by_batch = [self._alpha(X, y) for X, y in sequential_calib_loader]
        alpha_values   =  np.concatenate(alpha_by_batch)
        c              = Counter(alpha_values)
        alpha_sum_cum  = []

        for alpha_value in range(self.n_embs * self.n_neighbors, -1, -1):
            alpha_sum_cum.append(c[alpha_value] + (alpha_sum_cum[-1] if len(alpha_sum_cum) > 0 else 0))
            
        return np.array(alpha_sum_cum[::-1])
    
    def _build_neighs(self):
        print('Building Nearest Neighbor finders.')
        return [
            NearestNeighbors(
                n_neighbors = self.n_neighbors, 
                metric      = 'cosine'
            ).fit(feats) 
            for feats in self.conv_features
        ]
        
    def _alpha(self, X, y):
        neighbors_by_layer     = self._get_closest_points(X)
        closest_points_classes = self.train_targets[neighbors_by_layer]
        same_class_neighbors   = torch.Tensor(closest_points_classes) != y.reshape(y.shape[0], 1, 1)
        print((closest_points_classes.dtype,closest_points_classes.shape))
        print(y.dtype,y.shape)
        same_class_neighbors   = same_class_neighbors.reshape(-1, self.n_neighbors * self.n_embs)
        alpha                  = same_class_neighbors.sum(axis = 1)
        
        return alpha

    def _compute_nonconformity(self, X):
        neighbors_by_layer  = self._get_closest_points(X)
        closest_points_label   = self.train_targets[neighbors_by_layer]
        closest_points_label   = closest_points_label.reshape(-1, self.n_embs*self.n_neighbors)
        nonconformity          = [(closest_points_label != label).sum(axis = 1) for label in range(10)]
        nonconformity          = np.stack(nonconformity, axis = 1)
        
        return nonconformity
    
    def _compute_p_value(self, X):
        nonconformity = self._compute_nonconformity(X)
        empirical_p_value      = self.alpha_calib_sum[nonconformity] / len(self.calib_dataset)
        
        return empirical_p_value
    
    def _get_closest_points(self, X):
        *out_convs,_ = self.model(X.to(self.device))
        neighbors_by_layer = []
        for i, (neigh, layer_emb) in enumerate(zip(self.neighs, out_convs)):
            emb       = layer_emb.detach().cpu().reshape(X.size(0), -1).numpy()
            neighbors = neigh.kneighbors(emb, return_distance = False) 
            neighbors_by_layer.append(neighbors)
        return torch.tensor(np.stack(neighbors_by_layer, axis = 1))

    def predict(self, X):
        p_value     = self._compute_p_value(X)
        y_pred      = p_value.argmax(axis = 1)
        # Partitioning according to the second to last value in order to compute
        # credibility and confidence
        partition   = np.partition(p_value, -2)
        credibility = partition[:, -1]
        confidence  = 1 - partition[:, -2]
        
        return p_value, confidence, credibility

