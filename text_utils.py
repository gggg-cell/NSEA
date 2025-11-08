import faiss
from utils import *
import gc

PREFIX = r'http(s)?://[a-z\.]+/[^/]+/'


def norm_process(embed: torch.Tensor, eps=1e-5) -> torch.Tensor:
    n = embed.norm(dim=1, p=2, keepdim=True)
    embed = embed / (n + eps)
    return embed


def faiss_search_impl(emb_q, emb_id, emb_size, shift, k=50, search_batch_sz=50000, gpu=True):
    index = faiss.IndexFlat(emb_size)
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(emb_id)
    print('Total index =', index.ntotal)
    vals, inds = [], []
    for i_batch in tqdm(range(0, len(emb_q), search_batch_sz)):
        val, ind = index.search(emb_q[i_batch:min(i_batch + search_batch_sz, len(emb_q))], k)
        val = torch.from_numpy(val)
        val = 1 - val
        vals.append(val)
        inds.append(torch.from_numpy(ind) + shift)
        # print(vals[-1].size())
        # print(inds[-1].size())
    del index, emb_id, emb_q
    vals, inds = torch.cat(vals), torch.cat(inds)
    return vals, inds


@torch.no_grad()
def global_level_semantic_sim(embs, k=50, search_batch_sz=50000, index_batch_sz=500000
                              , split=False, norm=True, gpu=True):
    print('FAISS number of GPUs=', faiss.get_num_gpus())
    size = [embs[0].size(0), embs[1].size(0)]
    emb_size = embs[0].size(1)
    if norm:
        embs = apply(norm_process, *embs)
    emb_q, emb_id = apply(lambda x: x.cpu().numpy(), *embs)
    del embs
    gc.collect()
    vals, inds = [], []
    total_size = emb_id.shape[0]
    for i_batch in range(0, total_size, index_batch_sz):
        i_end = min(total_size, i_batch + index_batch_sz)
        val, ind = faiss_search_impl(emb_q, emb_id[i_batch:i_end], emb_size, i_batch, k, search_batch_sz, gpu)
        vals.append(val)
        inds.append(ind)

    vals, inds = torch.cat(vals, dim=1), torch.cat(inds, dim=1)
    print(vals.size(), inds.size())

    return topk2spmat(vals, inds, size, 0, torch.device('cpu'), split)



def get_batch_sim(embed, topk=50, split=True):
    # embed = self.get_gnn_embed()
    size = apply(lambda x: x.size(0), *embed)
    # x2y_val, x2y_argmax = fast_topk(2, embed[0], embed[1])
    # y2x_val, y2x_argmax = fast_topk(2, embed[1], embed[0])
    # ind, val = filter_mapping(x2y_argmax, y2x_argmax, size, (x2y_val, y2x_val), 0)
    spmat = global_level_semantic_sim(embed, k=topk, gpu=False).to(embed[0].device)
    if split:
        return spmat._indices(), spmat._values()
    else:
        return spmat


