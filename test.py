import pickle

import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras.backend as K
from torch import Tensor
from tqdm import trange
from scipy.sparse import csr_matrix
import faiss
import torch


def ill(pairs):
    return np.array(pairs, dtype=np.int64)

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
def sparse_sinkhorn_sims( sim1, top_k=50, iteration=15):

    sim1 = sim1.coalesce()
    sim1 = scipy.sparse.coo_matrix((sim1.values().numpy(),
                             (sim1.indices()[0].numpy(),
                              sim1.indices()[1].numpy())),
                            shape=sim1.size())
    sparse_matrix = sim1.tocsr()

    def sparse_topk(matrix,k):
        topk_values = []
        topk_indices = []
        for row in trange(matrix.shape[0]):
            row_data = matrix.getrow(row).toarray().flatten()
            if row_data.size == 0:
                topk_values.append([])
                topk_indices.append([])
                continue

            if row_data.size <= k:
                topk_idx = np.argsort(row_data)[::-1]
            else:
                topk_idx = np.argpartition(row_data, -k)[-k:]
                topk_idx = topk_idx[np.argsort(row_data[topk_idx])[::-1]]

            topk_values.append(row_data[topk_idx])
            topk_indices.append(topk_idx)
        return np.array(topk_values), np.array(topk_indices)

    # 获取top-k值和索引
    sims, index = sparse_topk(sparse_matrix, top_k)
    # sims, index = map(lambda x: x.numpy(), sim1.topk(top_k))
    row_sims = K.exp(sims.flatten() / 0.02)
    index = K.flatten(index.astype("int32"))

    size = sims.shape[0]
    row_index = K.transpose(([K.arange(size * top_k) // top_k, index, K.arange(size * top_k)]))
    col_index = tf.gather(row_index, tf.argsort(row_index[:, 1]))
    covert_idx = tf.argsort(col_index[:, 2])

    for _ in trange(iteration):
        row_sims = row_sims / tf.gather(indices=row_index[:, 0], params=tf.math.segment_sum(row_sims, row_index[:, 0]))
        col_sims = tf.gather(row_sims, col_index[:, 2])
        col_sims = col_sims / tf.gather(indices=col_index[:, 1], params=tf.math.segment_sum(col_sims, col_index[:, 1]))
        row_sims = tf.gather(col_sims, covert_idx)

    return K.reshape(row_index[:, 1], (-1, top_k)), K.reshape(row_sims, (-1, top_k))

def test(test_pair, sparse_tensor, top_k=50, iteration=15):
    left, right = test_pair[:, 0], np.unique(test_pair[:, 1])
    sparse_tensor = sparse_tensor.coalesce()
    nonzero_rows = sparse_tensor._indices()[0].unique()
    nonzero_cols = sparse_tensor._indices()[1].unique()

    # 创建映射张量
    row_mapping = torch.zeros(sparse_tensor.size()[0], dtype=torch.long)
    col_mapping = torch.zeros(sparse_tensor.size()[1], dtype=torch.long)

    # 填充映射张量
    row_mapping[nonzero_rows] = torch.arange(len(nonzero_rows))
    col_mapping[nonzero_cols] = torch.arange(len(nonzero_cols))

    # 重新编号
    new_row_indices = row_mapping[sparse_tensor._indices()[0]]
    new_col_indices = col_mapping[sparse_tensor._indices()[1]]

    # 创建新的稀疏张量
    new_indices = torch.stack([new_row_indices, new_col_indices])
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices, sparse_tensor._values(), (len(nonzero_rows), len(nonzero_cols)))

    index, sims = sparse_sinkhorn_sims(new_sparse_tensor, top_k, iteration)
    ranks = tf.argsort(-sims, -1).numpy()
    index = torch.from_numpy(index.numpy())
    sims = torch.from_numpy(sims.numpy())
    spmat = topk2spmat(sims, index, [len(nonzero_rows), len(nonzero_cols)], 0, torch.device('cpu'), False)
    ind = spmat._indices()
    val = spmat._values()
    ind = torch.stack(
        [ind[0] if nonzero_rows is None else nonzero_rows[ind[0]],
         ind[1] if nonzero_cols is None else nonzero_cols[ind[1]]]
    )
    batch_sim = ind2sparse(ind.to('cpu'), sparse_tensor.size(), values=val.to('cpu'))
    return batch_sim
def main():
    # fname = 'tmp/sim_large_ours_fr_duala_sinkhorn_30_it1.pkl'
    fname = 'tmp/sim_large_ours_fr_duala_none_30_it1.pkl'
    with open(fname, 'rb') as f:
        framework, sim1, sim2 = pickle.load(f)
    # index, sims = sparse_sinkhorn_sims(sim1)
    test(ill(framework.ds.test), sim1,)
    # print(index,sims)

# 检查当前模块是否为主模块，以避免在导入时执行主函数
if __name__ == "__main__":
    main()