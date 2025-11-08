from tqdm import trange
from utils import *
from typing import *


@torch.no_grad()
def resize_sparse(x: Tensor, new_size, ind_shift):
    xi, xv = x._indices(), x._values()
    for i, shift in enumerate(ind_shift):
        if shift == 0:
            continue
        xi[i] += shift
    return ind2sparse(xi, new_size, values=xv)
def get_hit_k(match_id: Tensor, link: Tensor, src=0, k_list=(1, 3, 5, 10), ignore=None, start=""):
    trg = 1 - src
    total = link.size(1)
    if ignore is not None:
        match_id[ignore] = torch.ones_like(match_id[ignore], device=match_id.device, dtype=torch.long) * -1
        ignore_sum = ignore.clone()
        ignore_sum[link[src]] = False
        print(start + "total ignore:", ignore.sum(), ", valid ignore", ignore_sum.sum())
        total = total - ignore_sum.sum()
    print(start + "total is ", total)
    match_id = match_id[link[src]]
    link: Tensor = link[trg]
    hitk_result = {}
    for k in k_list:
        if k > match_id.size(1):
            break
        match_k = match_id[:, :k]
        link_k = link.view(-1, 1).expand(-1, k)
        hit_k = (match_k == link_k).sum().item()
        hitk_result['hits@{}'.format(k)] = hit_k / total
        print("{2}hits@{0} is {1}".format(k, hit_k / total, start))

    return hitk_result

def sparse_max(tensor: Tensor, dim=-1):
    tensor = tensor.coalesce()
    return scatter_max(tensor._values(), tensor._indices()[dim], dim_size=tensor.size(dim))

def sparse_argmax(tensor, scatter_dim, dim=0):
    tensor = tensor.coalesce()
    argmax = sparse_max(tensor, dim)[1]
    argmax[argmax == tensor._indices().size(1)] = 0
    return tensor._indices()[scatter_dim][argmax]

@torch.no_grad()
def matrix_argmax(tensor: Tensor, dim=1):
    assert tensor.dim() == 2
    if tensor.is_sparse:
        return sparse_argmax(tensor, dim, 1 - dim)
    else:
        return torch.argmax(tensor, dim)

def apply(func, *args):
    if func is None:
        func = lambda x: x
    lst = []
    for arg in args:
        lst.append(func(arg))
    return tuple(lst)
@torch.no_grad()
def sparse_acc(sp_sim: Tensor, link: Tensor, device='cpu'):
    # ind, val, sz = split_sp(sp_sim)

    print('Total link is', link.size(1))
    # add_ind = torch.arange(min(sp_sim.size()))
    # add_val = torch.tensor(0.0001, dtype=torch.float32, device=link.device).expand_as(add_ind)
    # add_ind = torch.stack([add_ind, add_ind])
    # add = ind2sparse(add_ind, sp_sim.size(), values=add_val)
    # print(add.size(), add._values().size())
    # sp_sim = sp_sim + add
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link)
    print(sp_sim.size(), sp_sim._indices().size(), sp_sim._values().size())
    pred = matrix_argmax(sp_sim).view(-1)
    acc: Tensor = pred[link[0]] == link[1]
    print('calculate acc complete')
    return (acc.sum() / acc.numel()).item()

@torch.no_grad()
def cur_max(sparse_matrix1, sparse_matrix2):
    count1 = 0
    count2 = 0
    sparse_matrix1 = sparse_matrix1.coalesce()
    sparse_matrix2 = sparse_matrix2.coalesce()
    indices1 = sparse_matrix1._indices().t()
    indices2 = sparse_matrix2._indices().t()

    values1 = sparse_matrix1._values()
    values2 = sparse_matrix2._values()

    indices = []
    values = []
    while count1 < len(indices1) and count2 < len(indices2):
        if indices1[count1][0] < indices2[count2][0] or (
                indices1[count1][0] == indices2[count2][0] and indices1[count1][1] < indices2[count2][1]):
            indices.append(indices1[count1].tolist())
            values.append(values1[count1].tolist())
            count1 += 1
        elif indices1[count1][0] > indices2[count2][0] or (
                indices1[count1][0] == indices2[count2][0] and indices1[count1][1] > indices2[count2][1]):
            indices.append(indices2[count2].tolist())
            values.append(values2[count2].tolist())
            count2 += 1
        else:
            indices.append(indices1[count1])
            values.append(max(values1[count1], values2[count2]).tolist())
            count1 += 1
            count2 += 1
    if count1 != len(indices1):
        while count1 < len(indices1):
            indices.append(indices1[count1].tolist())
            values.append(values1[count1].tolist())
            count1 += 1
    elif count2 != len(indices2):
        while count2 < len(indices2):
            indices.append(indices2[count2].tolist())
            values.append(values2[count2].tolist())
            count2 += 1
    indices = torch.tensor(indices).t()
    values = torch.tensor(values)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, size=sparse_matrix1.shape)
    return sparse_matrix
@torch.no_grad()
def sparse_top_k(sp_sim: Tensor, link: Tensor, device='cuda', needed=(1, 5, 10, 50), batch_size=5000):
    # assert device == 'cpu'
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link)
    all_len = sp_sim.size(0)
    link_len = link.size(1)
    trg_len = sp_sim.size(1)
    # all_link = -1 * torch.ones(all_len).to(device)
    # all_link[link[0]] = link[1]
    topks = []
    for i_batch in trange(0, all_len, batch_size):
        i_end = min(all_len, i_batch + batch_size)
        curr_topk = resize_sparse(filter_which(sp_sim, ind_0=([torch.ge, torch.lt], [i_batch, i_end])),
                                  [i_end - i_batch, trg_len], [-i_batch, 0]).to_dense().topk(needed[-1])[1]
        topks.append(curr_topk)
    topks = torch.cat(topks)

    results = get_hit_k(topks, link, k_list=needed)
    results['MRR'] = truncated_mrr(topks, link)
    return results

def bi_csls_matrix(sim_matrix0, sim_matrix1, k=10, return2=True) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    dist0, _ = torch.topk(sim_matrix0, k)
    dist1, _ = torch.topk(sim_matrix1, k)
    if return2:
        return csls_impl(sim_matrix0, dist0, dist1), csls_impl(sim_matrix1, dist1, dist0)
    return csls_impl(sim_matrix0, dist0, dist1)

def csls_impl(sim_matrix, dist0, dist1) -> Tensor:
    dist0 = dist0.mean(dim=1).view(-1, 1).expand_as(sim_matrix)
    dist1 = dist1.mean(dim=1).view(1, -1).expand_as(sim_matrix)
    sim_matrix = sim_matrix * 2 - dist0 - dist1
    return sim_matrix

def truncated_mrr(topks: Tensor, link: Tensor, fail=None):
    src, trg = link
    if fail is None:
        fail = int(topks.size(0) / 2)
    rank = (topks[src] == trg.view(-1, 1)).to(torch.long) * 2
    rank_dummy = torch.ones([rank.size(0), 1]).to(topks.device)
    rank = torch.cat([rank, rank_dummy], dim=1)
    dummy_pos = rank.size(1) - 1
    rank = torch.argmax(rank, dim=-1, keepdim=False)
    rank[rank == dummy_pos] = fail
    mrr = (1.0 / (rank + 1).to(torch.float)).mean().item()
    print("MRR is {}".format(mrr))
    return mrr
def dense_to_sparse(x):
    if x.is_sparse:
        return x
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size()).coalesce()

@torch.no_grad()
def filter_which(x: Tensor, **kwargs):
    # ind_0 : gt 0
    # val: lt 0.01
    # ind_1  : eq 2
    # ind_1: eq [2,4,56]
    if not x.is_sparse:
        filter_which(dense_to_sparse(x), **kwargs)

    def cmpn(op, tensor, val):
        if not isinstance(val, Iterable):
            return op(tensor, val)
        mask = None
        multiple_op = isinstance(op, Iterable)
        for i, item in enumerate(val):
            curr = op[i](tensor, item) if multiple_op else op(tensor, item)
            mask = curr if mask is None else torch.logical_and(mask, curr)
        return mask

    ind, val = x._indices(), x._values()
    mask = None
    for k, v in kwargs.items():
        cmd = k.split('_')
        op, v = v
        if cmd[0] == 'ind':
            dim = int(cmd[1])
            curr = cmpn(op, ind[dim], v)
        elif cmd[0] == 'val':
            curr = cmpn(op, val, v)
        else:
            continue
        mask = curr if mask is None else torch.logical_and(mask, curr)
    # print('total', mask.numel(), 'total remain', mask.sum())
    return ind2sparse(ind[:, mask], x.size(), values=val[mask])


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


