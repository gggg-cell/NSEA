import scipy
import torch
from scipy.stats import rankdata

from dataset import *
from Partition import *
import pickle
from tools import *
from prev_models.wrapper import *
from common.sinkhorn import sparse_sinkhorn_sims,matrix_sinkhorn
import text_utils
from utils_largeea import *
from align_batch import *
from eval import sparse_top_k
class LargepartitonFramework:
    def __init__(self, ds: EAData, device="cuda", k = 30, src= 0):
        self.src = src
        self.ds = ds
        self.device = device
        self.partition = Partition(ds, k=k, src=src)

    @torch.no_grad()
    def get_test_sim(self, sim):

        d: EAData = self.ds
        candidates = SelectedCandidates(d.test, *d.ents)

        stru_sim = candidates.filter_sim_mat(sim)
        return stru_sim

    @torch.no_grad()
    def eval_sim(self, sim,batch_size=3000):
        d: EAData = self.ds
        # print('Size of sim:', sparse_tensor.size(), sparse_tensor._values().size())
        sim = self.get_test_sim(sim)
        print('Size of test sim:', sim.size(), sim._values().size())
        torch.cuda.empty_cache()
        acc = sparse_top_k(sim.to(self.device), d.ill(d.test, self.device), self.device, batch_size=batch_size)
        return sim,acc

    def get_cluster_result(self, top_k_corr, backbone, src_nodes, trg_nodes, max_sinkhorn_sz):
        mapping = self.partition.train_map[self.src]
        test_map = self.partition.test_map[self.src]

        # 优化：预计算训练集，避免重复计算
        src_train = self.partition.subgraph_trainset(src_nodes, self.src, no_trg=True)
        trg_train = self.partition.subgraph_trainset(trg_nodes, 1 - self.src, no_trg=True)

        # 优化：预计算重叠矩阵，使用更高效的数据结构
        overlap_data = [
            set(self.partition.train_map[self.src][i] for i in s) 
            for s in src_train
        ]
        trg_data = [set(s) for s in trg_train]
        
        corr = torch.from_numpy(overlaps(overlap_data, trg_data))
        corr_val, corr_ind = map(lambda x: x.numpy(), corr.topk(top_k_corr))
        
        # 优化：只计算一次align_loss，避免重复计算
        align_loss_stats = align_loss(src_nodes, trg_nodes, mapping=test_map, corr_ind=corr_ind, data=self.ds)
        
        # 优化：预计算三元组批次，减少重复的multi_place_triple调用
        triple1_batch = multi_place_triple(self.ds.triples[self.src], src_nodes)
        triple2_batch = multi_place_triple(self.ds.triples[1 - self.src], trg_nodes)
        
        # 优化：批量生成AlignmentBatch对象，减少循环开销
        for src_id, src_corr in enumerate(corr_ind):
            print(src_id)
            ids1_nodes, train1 = src_nodes[src_id], src_train[src_id]
            train2 = trg_train[src_id]
            ids2_nodes = trg_nodes[src_id]
            triple2 = triple2_batch[src_id]
            
            # 优化：使用集合操作，避免重复的set()调用
            ids1_nodes_set, ids2_nodes_set = set(ids1_nodes), set(ids2_nodes)
            train1_set, train2_set = set(train1), set(train2)
            
            train_pairs = make_pairs(train1_set, train2_set, mapping)
            test_pairs = make_pairs(ids1_nodes_set, ids2_nodes_set, test_map)
            
            batch = AlignmentBatch(
                triple1_batch[src_id], triple2,
                ids1_nodes_set, ids2_nodes_set, 
                train_pairs, test_pairs,
                backbone=backbone,
                max_sinkhorn_sz=max_sinkhorn_sz
            )
            # 将align_loss统计数据添加到batch对象中
            batch.align_loss_stats = align_loss_stats
            yield batch


class AlignmentBatch:
    def __init__(self, triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs, backbone='eakit',max_sinkhorn_sz=None):
        self.backbone = backbone
        self.device = "cuda"
        self.merge = True
        self.max_sinkhorn_sz = max_sinkhorn_sz
        if self.backbone in ['rrea', 'gcn-align', 'dual-amn', 'mraea', "dual-large", "rrea-large", "duala"]:
            self.ent_maps, self.rel_maps, ent_ids, rel_ids, \
                [t1, t2, train_ill, test_ill] = rearrange_ids([src_nodes, trg_nodes], self.merge,
                                                              triple1, triple2, train_pairs, test_pairs)
            # ids_train = [[(e1, e2 + es)for e1, e2 in i] for i in ids_train.values() ]
            self.shift = len(src_nodes)
            self.test_ill = test_ill
            self.train_ill = train_ill
            self.len_src, self.len_trg = len(src_nodes), len(trg_nodes)
            self.assoc = make_assoc(self.ent_maps, self.len_src, self.len_trg, self.merge)
            self.model = ModelWrapper(self.backbone,
                                      triples=t1 + t2,
                                      link=torch.tensor(test_ill).t(),
                                      ent_sizes=[len(ids) for ids in ent_ids],
                                      rel_sizes=[len(ids) for ids in rel_ids],
                                      device='cuda',
                                      dim=200
                                      )

            self.model.update_trainset(np.array(train_ill).T)
            self.model.update_devset(np.array(test_ill).T)
        else:
            raise NotImplementedError



    @torch.no_grad()
    def get_sim_mat(self, all_embeds, size, sinkhorn=True, norm=False):
        if isinstance(all_embeds, tuple):
            embeds = all_embeds
        else:
            embeds = [all_embeds[:self.shift], all_embeds[self.shift:]]

        ind, val = text_utils.get_batch_sim(embeds)

        ind = torch.stack(
            [self.assoc[ind[0]],
             self.assoc[ind[1] + self.shift]]
        )
        return ind2sparse(ind, size, values=val)

    def create_batch_sim(self, left, right, enhance='none', size=None, norm=False,
                          return_use_sinkhorn=False):
        use_sinkhorn = 0
        if size is None:
            size = [left.size(0), right.size(0)]

        assoc0 = torch.tensor([i[0] for i in self.test_ill])
        assoc1 = torch.tensor([i[1] - self.shift for i in self.test_ill ])
        left = left[assoc0.clone().detach().long()]
        right = right[assoc1.clone().detach().long()]
        if left.numel() == 0:
            return None, None
        sz = left.size(0) * right.size(0)
        if enhance == 'sinkhorn' and sz < (self.max_sinkhorn_sz ** 2):
            # sim = get_batch_csls_sim((left_embedding[assoc0], right_embedding[assoc1]), topk=300, split=False)
            if norm:
                dist_func = cosine_distance
            else:
                dist_func = l1_dist
            # if norm:
            #     dist_func = cosine_distance
            # sim = Lin_Sinkhorn(left, right, 1, 500, self.device)
            sim = matrix_sinkhorn(left, right, dist_func=dist_func, device=self.device)
            spmat = remain_topk_sim(sim, dim=0, k=50, split=False)
            # spmat = sim.to_sparse()
            ind = spmat._indices()
            val = spmat._values()
            use_sinkhorn = 1
        elif enhance == 'cls':
            sim_function = text_utils.get_batch_sim
            ind, val = sim_function((left.to(self.device),
                                     right.to(self.device)))
        else:
            sim = cosine_sim(left.to(self.device), right.to(self.device))
            spmat = remain_topk_sim(sim, dim=0, k=500, split=False)
            # sim = sim.to_sparse()
            ind = spmat._indices()
            val = spmat._values()
            use_sinkhorn = 1
        ind = torch.stack(
            [ind[0] if assoc0 is None else assoc0[ind[0]],
            ind[1] if assoc1 is None else assoc1[ind[1]]]
        )
        ind = torch.stack(
            [self.assoc[ind[0]],
             self.assoc[ind[1] + self.shift]]
        )
        # ind, val = self.keep_sim(ind, val, self.ids1)
        batch_sim = ind2sparse(ind.to(self.device), size, values=val.to(self.device))
        if return_use_sinkhorn:
            return batch_sim, use_sinkhorn
        return batch_sim

