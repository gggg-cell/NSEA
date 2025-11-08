import faiss
import  numpy as np
import  torch
from scipy.stats import rankdata
from torch import Tensor
from tqdm import tqdm
from torch_scatter import scatter, scatter_max, scatter_min
import sys
from eval import bi_csls_matrix
import keras.backend as K
import tensorflow as tf
import scipy.sparse as sp
sys.setrecursionlimit(2000)
global_dict = {}


def normalize_adj(adj):
    print('normalize adj')
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def get_matrix(triples, ent_size, rel_size):
    print(ent_size, rel_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    for i in range(ent_size):
        adj_features[i, i] = 1

    for h, r, t in triples:
        adj_matrix[h, t] = 1;
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1;
        adj_features[t, h] = 1
        radj.append([h, t, r]);
        radj.append([t, h, r + rel_size])
        rel_out[h][r] += 1;
        rel_in[t][r] += 1

    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    return adj_matrix, r_index, r_val, adj_features, rel_features


def add_log(key, value):
    global_dict[key] = value

def remain_topk_sim(matrix: Tensor, dim=0, k=1500, split=False):
    # print(matrix.size())
    if matrix.is_sparse:
        matrix = matrix.to_dense()
    val0, ind0 = torch.topk(matrix, dim=1 - dim, k=min(k, int(matrix.size(1 - dim))))
    return topk2spmat(val0, ind0, matrix.size(), dim, matrix.device, split)

def argprint(**kwargs):
    return '\n'.join([str(k) + "=" + str(v) for k, v in kwargs.items()])

def cosine_sim(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def ind2sparse(indices: Tensor, size, size2=None, dtype=torch.float, values=None):
    device = indices.device
    if isinstance(size, int):
        size = (size, size if size2 is None else size2)

    assert indices.dim() == 2 and len(size) == indices.size(0)
    if values is None:
        values = torch.ones([indices.size(1)], device=device, dtype=dtype)
    else:
        assert values.dim() == 1 and values.size(0) == indices.size(1)
    return torch.sparse_coo_tensor(indices, values, size)



def add_cnt_for(mp, val, begin=None):
    if begin is None:
        if val not in mp:
            mp[val] = len(mp)
        return mp, mp[val]
    else:
        if val not in mp:
            mp[val] = begin
            begin += 1
        return mp, mp[val], begin



def topk2spmat(val0, ind0, size, dim=0, device: torch.device = 'cuda', split=False):
    if isinstance(val0, np.ndarray):
        val0, ind0 = torch.from_numpy(val0).to(device), \
                     torch.from_numpy(ind0).to(device)

    if split:
        return val0, ind0, size

    ind_x = torch.arange(size[dim]).to(device)
    ind_x = ind_x.view(-1, 1).expand_as(ind0).reshape(-1)
    ind_y = ind0.reshape(-1)
    ind = torch.stack([ind_x, ind_y])
    val0 = val0.reshape(-1)
    filter_invalid = torch.logical_and(ind[0] >= 0, ind[1] >= 0)
    ind = ind[:, filter_invalid]
    val0 = val0[filter_invalid]
    return ind2sparse(ind, list(size), values=val0).coalesce()

def apply(func, *args):
    if func is None:
        func = lambda x: x
    lst = []
    for arg in args:
        lst.append(func(arg))
    return tuple(lst)

def get_batch_csls_sim(embed, topk=50, csls=10, split=True, sim_func=cosine_sim):
    device = embed[0].device
    from utils_largeea import cosine_sim
    sim0 = sim_func(embed[0], embed[1])
    sim1 = sim_func(embed[1], embed[0])
    sim = bi_csls_matrix(sim0, sim1, k=csls, return2=False)
    val, ind = sim.topk(topk)
    spmat = topk2spmat(val, ind, sim.size(), device=device)
    if split:
        return spmat._indices(), spmat._values()
    else:
        return spmat


def ranks_sim(sims:Tensor):
    coo_tensor = sims.coalesce().cpu().detach()
    coo_values = coo_tensor.values()
    coo_indices = coo_tensor.indices()
    sims = sp.csr_matrix((coo_values, coo_indices), shape=sims.size())
    nonzero_counts = sims.getnnz(axis=1)
    nonzero_rows = nonzero_counts.nonzero()[0]
    ranked_values = np.zeros_like(coo_values)
    start_idx = 0
    for row in nonzero_rows:
        row_length = nonzero_counts[row]
        row_values = coo_values[start_idx:start_idx + row_length]
        ranks = rankdata(-row_values, method="average")
        ranked_values[start_idx:start_idx + row_length] = ranks
        start_idx += row_length
    coo_values[:] = torch.tensor(500 - ranked_values)
    sim1 = torch.sparse_coo_tensor(coo_indices, coo_values, size=sims.shape)
    return sim1

