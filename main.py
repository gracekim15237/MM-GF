import random
import numpy as np
import torch
import scipy.sparse as sp
import os
import pandas as pd
import argparse
from src.utils.utils_PF import (csr2torch, recall_at_k, ndcg_at_k, normalize_sparse_adjacency_matrix,json_to_dok_matrix, 
                                normalize, experiment, recall_, ndcg_, map_, precision_)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

args, _ = parser.parse_known_args()

dataset = args.dataset

random.seed(999)
np.random.seed(999)

filename = dataset+".inter"
path = os.getcwd() +"/data/"+dataset

df = pd.read_csv(path+"/"+filename, sep = "\t")

df_tr = df[df["x_label"]!=2]
df_ts = df[df["x_label"]==2]

df_tr = df_tr.drop(["timestamp", "x_label"], axis = 1)
df_ts = df_ts.drop(["timestamp", "x_label"], axis = 1)

R_tr = sp.csr_matrix((df_tr.rating, (df_tr.userID, df_tr.itemID)))
R_ts = sp.csr_matrix((df_ts.rating, (df_ts.userID, df_ts.itemID)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

verbose = 0  # 0: no print, 1: print results

alpha = 0.89
power = 0.67
a = 600
b = 0.1

R_tr = csr2torch(R_tr)
R_ts = csr2torch(R_ts)

n_users = R_tr.shape[0]
n_items = R_tr.shape[1]
if verbose:
    print(f"number of users: {n_users}")
    print(f"number of items: {n_items}")

n_inters = torch.nonzero(R_tr._values()).cpu().size(0) + torch.nonzero(R_ts[0]._values()).cpu().size(0)

if verbose:
    print(f"number of overall ratings: {n_inters}")


mceg_norm = normalize_sparse_adjacency_matrix(R_tr.to_dense(), alpha)
R = R_tr.to_dense()
P = mceg_norm.T @ mceg_norm
P.data **= power
P = P.to(device=device).float()
R = R.to(device=device).float()

path_ti = os.getcwd()+"/data/"+dataset+"/"

text = np.load(path_ti + 'text_feat.npy')
img = np.load(path_ti + 'image_feat.npy')

text_csr = sp.csr_matrix(text)
img_csr = sp.csr_matrix(img)

R_text = csr2torch(text_csr)
R_img = csr2torch(img_csr)

mceg_norm_text = normalize(R_text.to_dense(), alpha)
P_text = mceg_norm_text @ mceg_norm_text.T
P_text.data **= power
P_text = P_text.to(device=device).float()

mceg_norm_img = normalize(R_img.to_dense(), alpha)
P_img = mceg_norm_img @ mceg_norm_img.T
P_img.data **= power
P_img = P_img.to(device=device).float()

print("dataset : " +dataset)
experiment(R, R_tr, R_ts, P, P_text, P_img, a , b, 1, 1, 1, 0.05,20)