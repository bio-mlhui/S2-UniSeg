import torch
import numpy as np
import torch.nn.functional as F

def aggo_merge(features, threshes, min_size):
    import time
    compare_time = 0
    clusters = []
    similarities = {}
    H, W = features.shape[:2]
    # 初始化相似度和cluster
    cluster_idx = 0
    whole_time = time.time()
    for y in range(H):
        for x in range(W):
            mask = np.zeros((H, W))
            mask[y, x] = 1
            clusters.append({
                'feature': features[y, x],
                'normalized_feature': F.normalize(features[y, x], dim=0),
                'mask': mask,
                'num_of_patch': 1,
                'neighbors': set()
            })

            if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                clusters[cluster_idx]['neighbors'].add(cluster_idx-1)
                clusters[cluster_idx-1]['neighbors'].add(cluster_idx)
                similarities[(cluster_idx-1, cluster_idx)] = \
                    torch.dot(clusters[cluster_idx-1]['normalized_feature'], clusters[cluster_idx]['normalized_feature'])
            if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                clusters[cluster_idx]['neighbors'].add(cluster_idx-W)
                clusters[cluster_idx-W]['neighbors'].add(cluster_idx)
                similarities[(cluster_idx-W, cluster_idx)] = \
                    torch.dot(clusters[cluster_idx-W]['normalized_feature'], clusters[cluster_idx]['normalized_feature'])
                
            cluster_idx += 1

    all_masks = []   # list[nqi h w], level
    all_features = []   # list[nqi c], level
    lengths = []
    for thresh in threshes: # 每个threshold提取出来对应的mask
        while len(similarities): # merge, 直到similarity的最大值小于threshold
            start = time.time()
            i, j = max(similarities, key=similarities.get)
            compare_time += time.time() - start
            if similarities[(i, j)] < thresh: break 
            
            c1, c2 = clusters[i], clusters[j]
            weighted_sum = ((c1['feature'] + c2['feature']) / (c1['num_of_patch'] + c2['num_of_patch']))
            #weighted_sum = ((c1['num_of_patch']*c1['feature'] + c2['num_of_patch']*c2['feature']) / (c1['num_of_patch'] + c2['num_of_patch'])).float()
            merged = {
                'feature': c1['feature'] + c2['feature'],
                'normalized_feature': F.normalize(weighted_sum, dim=0),
                'mask': (c1['mask'] + c2['mask']) > 0,
                'num_of_patch': c1['num_of_patch'] + c2['num_of_patch'],
                'neighbors': c1['neighbors'].union(c2['neighbors']).difference(set([i, j]))
            }
            clusters.append(merged)

            del similarities[(i, j)]
            for neighbor in merged['neighbors']:
                if i in clusters[neighbor]['neighbors']:
                    if neighbor < i: del similarities[(neighbor, i)]
                    else: del similarities[(i, neighbor)]
                    clusters[neighbor]['neighbors'].discard(i)
                if j in clusters[neighbor]['neighbors']:
                    if neighbor < j: del similarities[(neighbor, j)]
                    else: del similarities[(j, neighbor)]
                    clusters[neighbor]['neighbors'].discard(j)

                similarities[(neighbor, cluster_idx)] = \
                    torch.dot(clusters[neighbor]['normalized_feature'], clusters[cluster_idx]['normalized_feature'])
                clusters[neighbor]['neighbors'].add(cluster_idx)

            cluster_idx += 1
        
        single_level_masks = []
        single_level_features = []

        counted_cluster = set()
        for (m, n) in similarities:
            if m not in counted_cluster:
                counted_cluster.add(m)
                if clusters[m]['num_of_patch'] >= min_size:
                    single_level_masks.append(clusters[m]['mask'])
                    single_level_features.append(clusters[m]['feature'])
            if n not in counted_cluster:
                counted_cluster.add(n)
                if clusters[n]['num_of_patch'] >= min_size:
                    single_level_masks.append(clusters[n]['mask'])
                    single_level_features.append(clusters[n]['feature'])

        if len(single_level_features):
            all_masks.append(torch.from_numpy(np.stack(single_level_masks)))
            all_features.append(torch.stack(single_level_features))
            lengths.append(len(single_level_features))
    whole_time = time.time() - whole_time
    print(f'whole_time: {whole_time}, Compare time: {compare_time}')
    return torch.cat(all_masks, dim=0), torch.cat(all_features, dim=0), lengths # nq c

# heap
from torch_geometric.data import Batch, Data
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import Aggregation
import dgl
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import Data

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import (
    dense_to_sparse,
    one_hot,
    to_dense_adj,
    to_scipy_sparse_matrix,
)


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor

import time
def aggo_merge_graph(features):
    # H W c
    H, W = features.shape[:2]
    nodes_feature = features.flatten(0, 1) # hw c
    edge_index = []
    cluster_idx = 0
    for y in range(H):
        for x in range(W):
            assert cluster_idx == y * H + x
            if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                edge_index.append([cluster_idx-1, cluster_idx])
            if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                edge_index.append([cluster_idx-W, cluster_idx])
            cluster_idx += 1
    edge_index = torch.tensor(edge_index).permute(1, 0).contiguous().long().to(features.device) # 2 N
    batch_tensor = torch.zeros(H*W).long().to(features.device)
    node_num_patches = torch.ones(H*W).to(features.device).int()
    src_nodes_feats = nodes_feature[edge_index[0]] # e c
    tgt_nodes_feats = nodes_feature[edge_index[1]] # e c
    edge_score = torch.einsum('ec,ec->e',
                                F.normalize(src_nodes_feats, dim=-1, eps=1e-10), 
                                F.normalize(tgt_nodes_feats, dim=-1, eps=1e-10))

    thre_time = []
    clusters_features = []
    lengths = []
    # for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    # for threshold in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # for threshold in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
    for threshold in [0.8, 0.7, 0.6, 0.5, 0.4]:
    # for threshold in [0.96, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        # 0.96
        # 0.9的部分有很多重复的，证明很多语义近的 相似度都小于0.9
        # 背景类: 987654, 87654 很好， 987/654很不好，证明背景类都在987上面，单个背景类自身很聚集，但是背景类和前景的相似度也不差，所以在后面进行
        # 背景类聚的不好，很多背景类
        # 没有除以num_patches之后没有了无意义
        start = time.time()
        # threshold是top25%
        while (edge_score > threshold).any():
            nodes_feature, edge_index, batch_tensor, unpool_info, node_num_patches = hc_graph(x=nodes_feature, # N c
                                                                            edge_index=edge_index, # 2 E
                                                                            batch=batch_tensor, # N
                                                                            threshold=threshold,
                                                                            edge_score=edge_score,
                                                                            node_num_patches=node_num_patches) # E
            src_nodes_feats = nodes_feature[edge_index[0]]
            tgt_nodes_feats = nodes_feature[edge_index[1]]
            edge_score = torch.einsum('ec,ec->e',
                                        F.normalize(src_nodes_feats, dim=-1, eps=1e-10), 
                                        F.normalize(tgt_nodes_feats, dim=-1, eps=1e-10))  
        thre_time.append(time.time() - start)

        legal_features = []
        counted_cluster = set()
        for (m, n) in edge_index.permute(1, 0).tolist(): # E 2
            if m not in counted_cluster:
                counted_cluster.add(m)
                if node_num_patches[m] >= 4:
                    legal_features.append(nodes_feature[m])
            if n not in counted_cluster:
                counted_cluster.add(n)
                if node_num_patches[n] >= 4:
                    legal_features.append(nodes_feature[n])
        if len(legal_features):
            legal_features = torch.stack(legal_features, dim=0)
        clusters_features.append(legal_features)
        lengths.append(len(legal_features))
    print(thre_time)
    print(sum(thre_time))
    return None, torch.cat(clusters_features,dim=0), lengths


from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_undirected
from torch_geometric.nn.pool import ClusterPooling
def hc_graph(
    x: Tensor,
    # x_num_patches: Tensor, # N, 有多大的像素
    edge_index: Tensor, # 2 E, 每个edge从小到大, 但是是一个无向图
    batch: Tensor, # N
    threshold: float,
    edge_score, # E, float, -1/1
    node_num_patches, # N int
) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:

    edge_contract = edge_index[:, edge_score > threshold] # 2 E_chosen
    # A_contract = to_dense_adj(edge_contract,max_num_nodes=x.size(0)).squeeze(0)
    # A_contract = A_contract + A_contract.T

    adj = to_scipy_sparse_matrix(edge_contract, num_nodes=x.size(0))
    adj = adj + adj.T
    # N N, 原始节点之间是否被选中
    A_contract = torch.from_numpy(adj.toarray()).float().to(edge_contract.device)
    # N 每个节点之后的index
    _, cluster_np = connected_components(adj, directed=False)
    cluster = torch.tensor(cluster_np, dtype=torch.long, device=x.device)#
    C = one_hot(cluster) # N N' 新节点的归属问题, sum(-1)都是1

    # N N, 原始节点之间是否连接
    A = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)
    A = A + A.T
    # N N, 原始节点之间的相似度, 不相连的两个节点之间是0
    # S = to_dense_adj(edge_index, edge_attr=edge_score, max_num_nodes=x.size(0)).squeeze(0)
    # S = S + S.T
    # # 单个节点自成一派的话就是1
    # nodes_single = ((A_contract.sum(dim=-1) + A_contract.sum(dim=-2)) == 0).nonzero()
    # S[nodes_single, nodes_single] = 1.0
    
    # N N * N N' -> N' N @ N c -> N' c
    # x_out = (S @ C).t() @ x

    # N': N' N @ N -> N'
    node_num_patches = (C.float().t() @ (node_num_patches.float())).int()
    # N' N @ N c -> N' c 
    x_out = C.t() @ x
    # x_out = x_out / node_num_patches[:, None]

    # N' N @ N N @ N N' -> N' N'
    new_new_adj = (C.T @ A @ C).fill_diagonal_(0)
    edge_index_out, _ = dense_to_sparse(torch.triu(new_new_adj))

    # N'
    # batch_out[cluster[i]]=batch[i], i=1...N
    batch_out = batch.new_empty(x_out.size(0)).scatter_(0, cluster, batch)
    unpool_info = UnpoolInfo(edge_index, cluster, batch)
    return x_out, edge_index_out, batch_out, unpool_info, node_num_patches


def get_edge_score(nodes_feature, edge_index):
    src_nodes_feats = nodes_feature[edge_index[0]] # e c
    tgt_nodes_feats = nodes_feature[edge_index[1]] # e c
    edge_score = torch.einsum('ec,ec->e', 
                              F.normalize(src_nodes_feats, dim=-1, eps=1e-10),  F.normalize(tgt_nodes_feats, dim=-1, eps=1e-10))
    return edge_score

def aggo_whole_batch(nodes_feature, edge_index, node_batch_tensor, edge_batch_tensor, node_num_patches):
    # N c, 2 E, N, E, N
    # list[ni c], crop_batch
    batch_tensor = node_batch_tensor # V
    edge_score = get_edge_score(nodes_feature, edge_index) # E
    thre_time = []
    clusters_features = [] # list[ni c], thre
    clusters_feats_batch = [] # list[ni], thre
    for threshold in [0.8, 0.7, 0.6, 0.5, 0.4]:
        start = time.time()
        while (edge_score > threshold).any():
            nodes_feature, edge_index, batch_tensor, unpool_info, node_num_patches = hc_graph(x=nodes_feature, # N c
                                                                                            edge_index=edge_index, # 2 E
                                                                                            batch=batch_tensor, # N
                                                                                            threshold=threshold,
                                                                                            edge_score=edge_score,# E
                                                                                            node_num_patches=node_num_patches) # E
            edge_score = get_edge_score(nodes_feature, edge_index)

        thre_time.append(time.time() - start)

        legal_features = [] # ni c
        legal_feat_batch = [] # ni
        counted_cluster = set()
        for (m, n) in edge_index.permute(1, 0).tolist(): # E 2
            if m not in counted_cluster:
                counted_cluster.add(m)
                if node_num_patches[m] >= 4:
                    legal_features.append(nodes_feature[m])
                    legal_feat_batch.append(batch_tensor[m])
            if n not in counted_cluster:
                counted_cluster.add(n)
                if node_num_patches[n] >= 4:
                    legal_features.append(nodes_feature[n])
                    legal_feat_batch.append(batch_tensor[n])
        if len(legal_features):
            legal_features = torch.stack(legal_features, dim=0)
            legal_feat_batch = torch.tensor(legal_feat_batch)
            clusters_features.append(legal_features)
            clusters_feats_batch.append(legal_feat_batch)
    

    unique_batch_ids = node_batch_tensor.unique().tolist()
    batch_clusters_feats = [] # list[list[ni c], threshold] batch
    for batch_id in unique_batch_ids:
        thre_feats = [] # list[ni c], threshold
        for clu_fea, clu_btc in zip(clusters_features, clusters_feats_batch):
            thre_feats.append(clu_fea[clu_btc == batch_id])
        batch_clusters_feats.append(thre_feats)

    # print(thre_time)
    # print(sum(thre_time))
    
    return batch_clusters_feats

