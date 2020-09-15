import pandas as pd
import pdb
import numpy as np
import scipy.sparse
import sys
import logging
import glob
from sklearn import preprocessing
from math import *
import os

logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

def load_data(data):
    if 'cate' in data:
        train = read_cate_data('%s/train/'%data)
        valid = read_cate_data('%s/valid/'%data)
        test = read_cate_data('%s/test/'%data)
        logging.info('Categorical data loaded.\n train_x shape: {trn_x_shape}. train_y shape: {trn_y_shape}.\n valid_x shape: {vld_x_shape}. valid_y shape: {vld_y_shape}.\n test_x shape: {test_x_shape}. test_y shape: {test_y_shape}.' .format(trn_x_shape = train[0].shape, trn_y_shape = train[1].shape, vld_x_shape = valid[0].shape, vld_y_shape = valid[1].shape, test_x_shape = test[0].shape, test_y_shape = test[1].shape))
        return train, valid, test
    if os.path.exists("%s/train_features.npy"%data):
        trn_x = np.load("%s/train_features.npy"%data).astype(np.float32)
        vld_x = np.load("%s/valid_features.npy"%data).astype(np.float32)
        test_x = np.load("%s/test_features.npy"%data).astype(np.float32)
        mean = np.mean(trn_x, axis=0)
        std = np.std(trn_x, axis=0)
        trn_x = (trn_x - mean) / (std + 1e-5)
        vld_x = (vld_x - mean) / (std + 1e-5)
        test_x = (test_x - mean) / (std + 1e-5)
    else:
        trn_x = scipy.sparse.load_npz("%s/train_x_csr.npz"%data)
        vld_x = scipy.sparse.load_npz("%s/valid_x_csr.npz"%data)
        test_x = scipy.sparse.load_npz("%s/test_x_csr.npz"%data)
    trn_y = np.load("%s/train_labels.npy"%data).astype(np.float32)
    vld_y = np.load("%s/valid_labels.npy"%data).astype(np.float32)
    test_y = np.load("%s/test_labels.npy"%data).astype(np.float32)

    logging.info('data loaded.\n train_x shape: {trn_x_shape}. train_y shape: {trn_y_shape}.\n valid_x shape: {vld_x_shape}. valid_y shape: {vld_y_shape}.\n test_x shape: {test_x_shape}. test_y shape: {test_y_shape}.' .format(trn_x_shape = trn_x.shape, trn_y_shape = trn_y.shape, vld_x_shape = vld_x.shape, vld_y_shape = vld_y.shape, test_x_shape = test_x.shape, test_y_shape = test_y.shape))
    return trn_x, trn_y, vld_x, vld_y, test_x, test_y

def read_cate_data(dir_path):
    y = np.load(dir_path + '_label.npy')[:,None]
    xi = np.load(dir_path + '_index.npy')
    feature_sizes = np.load(dir_path + '_feature_sizes.npy').tolist()
    print("loaded from %s."%dir_path)
    # xv = np.load(dir_path + '_value.npy')
    # x = np.concatenate([xi[:,:,None],xv[:,:,None]], axis=-1)
    return xi, y.astype(np.float32), feature_sizes

# for fast version cateNN
def trans_cate_data(cate_data, old_feature_sizes=None):
    train, valid, test = cate_data
    train_xs, train_y, feature_sizes = train
    valid_xs, valid_y, _ = valid
    test_xs, test_y, _ = test
    if old_feature_sizes is not None:
        feature_sizes = old_feature_sizes
    sum_feats = feature_sizes[0]
    for idx in range(1, len(feature_sizes)):
        # sum_feats.append(sum_feats[idx-1]+feature_sizes[idx-1])
        train_xs[:,idx] += sum_feats
        valid_xs[:,idx] += sum_feats
        test_xs[:,idx] += sum_feats
        sum_feats += feature_sizes[idx]
    return ((train_xs, train_y, feature_sizes), (valid_xs, valid_y, _), (test_xs, test_y, _))
