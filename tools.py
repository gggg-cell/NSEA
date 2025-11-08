import pickle
import random
import time

import faiss
from sklearn import preprocessing
from tqdm import tqdm
from collections import defaultdict
from utils import *


def read_cluser(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    src_nodes = loaded_data["var1"]
    trg_nodes = loaded_data["var2"]
    tmp1_nodes = loaded_data["var3"]
    tmp2_nodes = loaded_data["var4"]
    return src_nodes, trg_nodes, tmp1_nodes, tmp2_nodes


def save_cluster(file_name, src_nodes, trg_nodes, tmp1_nodes, tmp2_nodes):
    data = {
        "var1": src_nodes,
        "var2": trg_nodes,
        "var3": tmp1_nodes,
        "var4": tmp2_nodes
    }
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)


def multi_place_triple(triples, nodes_batch):
    batch = defaultdict(list)
    print('split triplets')
    node = set()
    for i, nodes in enumerate(tqdm(nodes_batch)):  # enumerate(nodes_batch):
        set_nodes = set(nodes)
        for h, r, t in triples:
            if h in set_nodes and t in set_nodes:
                node.add(h)
                node.add(t)
                batch[i].append((h, r, t))
    return batch


def build_batch_dict(nodes_list):
    batch_dict = defaultdict()
    for i, nodes in enumerate(nodes_list):
        for n in nodes:
            batch_dict[n] = i
    return batch_dict


def process_triples(triples, mini_batch_dict, batch_dict):
    subgraph_link = defaultdict(list)
    for h, r, t in triples:
        mini_h_batch, mini_t_batch, h_batch, t_batch = mini_batch_dict.get(h, -1), mini_batch_dict.get(t,
                                                                                                       -1), batch_dict.get(
            h, -1), batch_dict.get(t, -1)
        if h_batch == t_batch or h_batch < 0 or t_batch < 0 or mini_h_batch < 0 or mini_t_batch < 0:
            continue
        elif mini_h_batch != mini_t_batch and h_batch != t_batch:
            subgraph_link[h_batch].append(mini_t_batch)
            subgraph_link[t_batch].append(mini_h_batch)
        else:
            raise Exception("The mini_src nodes and src nodes are not the same, please check the code.")
    return subgraph_link


def combine_link(src_nodes, trg_nodes, mini_src_nodes, mini_trg_nodes, subgraph_src_link, subgraph_trg_link, len_src,
                 len_trg):
    sub_src_link = process_subgraph_link(src_nodes, mini_src_nodes, subgraph_src_link)
    sub_trg_link = process_subgraph_link(trg_nodes, mini_trg_nodes, subgraph_trg_link)
    result = defaultdict(list)
    subgraph_link = defaultdict(list)
    value_dict = defaultdict(list)
    for key, value in sub_src_link.items():
        tmp = []
        for i in range(len(value)):
            tmp.append(value[i] + sub_trg_link[key][i])
        value_dict[key] = tmp
    for key, value in value_dict.items():
        tmp = sorted(value, reverse=True)
        subgraph_link[key] = [value.index(j) for j in tmp]

    for key, value in subgraph_link.items():
        weight1 = [len(mini_src_nodes[i]) // 500 for i in value]
        weight2 = [len(mini_trg_nodes[i]) // 500 for i in value]
        values = [value_dict[key][i] for i in value]

        path = knapsack(weight1, weight2, values, (len_src - len(src_nodes[key])) // 500,
                        (len_trg - len(trg_nodes[key])) // 500)
        result[key] = [subgraph_link[key][i] for i in path]
    return result


def process_subgraph_link(nodes, mini_nodes, subgraph_src_link):
    subgraph_link = defaultdict(list)
    # for i in subgraph_src_link.keys():
    #     subgraph_link[i] = subgraph_src_link[i]  
    for i in tqdm(range(len(nodes)), desc="Processing subgraph_link"):
        subgraph_link[i] = [subgraph_src_link.get(i).count(j) / len(nodes[i]) for j in range(len(mini_nodes))]
    return subgraph_link


def process_nodes(result, nodes, mini_nodes):
    tmp = defaultdict(list)
    for key, value in result.items():
        tmp[key] = nodes[key].copy()
        for i in value:
            tmp[key] = tmp[key] + mini_nodes[i]
    tuple1 = sorted(tmp.items(), key=lambda x: x[0])
    tmp_nodes = [list(i[1]) for i in tuple1]
    return tmp_nodes


def addtriples(g, triple, weight=None):
    if weight is None:
        for t in triple:
            g.add_node(t[0])
            g.add_node(t[-1])
            g.add_edge(t[0], t[-1], weight=random.randint(1, 10))
    else:
        for t in triple:
            g.add_node(t[0])
            g.add_node(t[-1])
            g.add_edge(t[0], t[-1], weight=weight)
    return g

def add_nodes(mini_src_nodes, src_nodes, mini_trg_nodes, trg_nodes, triple1, triple2):
    mini_src_batch = build_batch_dict(mini_src_nodes)
    mini_trg_batch = build_batch_dict(mini_trg_nodes)
    src_batch = build_batch_dict(src_nodes)
    trg_batch = build_batch_dict(trg_nodes)
    subgraph_src_link = process_triples(triple1, mini_src_batch, src_batch)
    subgraph_trg_link = process_triples(triple2, mini_trg_batch, trg_batch)

    return subgraph_src_link, subgraph_trg_link


def knapsack(weights, volumes, values, W, V):
    n = len(weights)
    dp = [[[0 for _ in range(V + 1)] for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            for v in range(1, V + 1):
                if weights[i - 1] <= w and volumes[i - 1] <= v:
                    dp[i][w][v] = max(dp[i - 1][w][v],
                                      values[i - 1] + dp[i - 1][w - weights[i - 1]][v - volumes[i - 1]])
                else:
                    dp[i][w][v] = dp[i - 1][w][v]

    path = []
    i, w, v = n, W, V
    while i > 0 and w > 0 and v > 0:
        if dp[i][w][v] != dp[i - 1][w][v]:
            path.append(i - 1)
            w -= weights[i - 1]
            v -= volumes[i - 1]
        i -= 1

    return path


def rearrange_ids(nodes, merge: bool, *to_map):
    ent_mappings = [{}, {}]
    rel_mappings = [{}, {}]
    ent_ids = [[], []]
    shift = 0
    for w, node_set in enumerate(nodes):
        for n in node_set:
            ent_mappings[w], nn, shift = add_cnt_for(ent_mappings[w], n, shift)
            ent_ids[w].append(nn)
        shift = len(ent_ids[w]) if merge else 0
    print(len(ent_ids[0]), len(ent_ids[1]))
    print(len(set(ent_ids[0])), len(set(ent_ids[1])))
    mapped = []
    shift = 0
    curr = 0
    for i, need in enumerate(to_map):
        now = []
        if len(need) == 0:
            mapped.append([])
            continue
        is_triple = len(need[0]) == 3
        for tu in need:
            if is_triple:

                h, t = ent_mappings[curr][tu[0]], ent_mappings[curr][tu[-1]]
                rel_mappings[curr], r, shift = add_cnt_for(rel_mappings[curr], tu[1], shift)
                if h is None or t is None:
                    raise ValueError("None in mapping")
                now.append([h, r, t])
            else:
                now.append([ent_mappings[0][tu[0]], ent_mappings[1][tu[-1]]])
        seen = set()
        now = [x for x in now if tuple(x) not in seen and not seen.add(tuple(x))]
        mapped.append(now)
        curr += is_triple
        if not merge:
            shift = 0
    rel_ids = [list(rm.values()) for rm in rel_mappings]

    return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped


# def rearrange_ids(nodes, merge: bool, *to_map):
#     ent_mappings = [{}, {}]
#     rel_mappings = [{}, {}]
#     ent_ids = [[], []]

#     mapped = []
#     curr = 0
#     shift1 = 0
#     shift2 = 0
#     for i, need in enumerate(to_map):
#         now = []
#         if len(need) == 0:
#             mapped.append([])
#             continue
#         is_triple = len(need[0]) == 3
#         if is_triple:
#             for tu in need:
#                 ent_mappings[curr], h, shift1 = add_cnt_for(ent_mappings[curr], tu[0], shift1)
#                 ent_mappings[curr], t, shift1 = add_cnt_for(ent_mappings[curr], tu[-1], shift1)
#                 rel_mappings[curr], r, shift2 = add_cnt_for(rel_mappings[curr], tu[1], shift2)
#                 now.append([h, r, t])
#         else:
#             for tu in need:
#                 mp1 = ent_mappings[0][tu[0]]
#                 mp2 = ent_mappings[1][tu[-1]]
#                 if mp1 is None or mp2 is None:
#                     continue
#                 now.append([mp1, mp2])
#         seen = set()
#         now = [x for x in now if tuple(x) not in seen and not seen.add(tuple(x))]
#         mapped.append(now)
#         curr += is_triple

#     for w, node_set in enumerate(nodes):
#         for n in node_set:
#             nn = ent_mappings[w][n]
#             if nn is None:
#                 continue
#             ent_ids[w].append(nn)
#     print(len(ent_ids[0])+len(ent_ids[1]))
#     print(len(ent_mappings[0])+len(ent_mappings[1]))

#     rel_ids = [list(rm.values()) for rm in rel_mappings]

#     return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped


def make_assoc(maps, src_len, trg_len, merge):
    assoc = np.empty(src_len + trg_len, dtype=int)
    shift = 0 if merge else 1
    shift = shift * src_len
    for idx, ent_mp in enumerate(maps):
        for k, v in ent_mp.items():
            assoc[v + idx * shift] = k
    return torch.tensor(assoc)


def make_pairs(src, trg, mp):
    return list(filter(lambda p: p[1] in trg, [(e, mp[e]) for e in set(filter(lambda x: x in mp, src))]))


def align_loss(src_nodes, trg_nodes, mapping=None, corr_ind=None, data=None):
    # 优化：使用集合操作和向量化计算，减少循环开销
    sum_count = 0
    pair = set()
    has_nodes1 = set()
    has_nodes2 = set()
    
    # 优化：预计算映射的键集合，避免重复的in操作
    mapping_keys = set(mapping.keys()) if mapping else set()
    
    # 优化：批量处理所有相关性索引
    for src_id, src_corr in enumerate(corr_ind):
        # 优化：使用集合的并集操作，比逐个添加更高效
        current_src_nodes = src_nodes[src_id]
        current_trg_nodes = trg_nodes[src_corr.tolist()[0]]
        
        has_nodes1.update(current_src_nodes)
        has_nodes2.update(current_trg_nodes)
        
        # 优化：使用集合交集操作，减少过滤开销
        src_filtered = current_src_nodes & mapping_keys
        trg_batch_set = set(current_trg_nodes)
        
        # 优化：批量检查映射关系，减少循环次数
        for e in src_filtered:
            if mapping[e] in trg_batch_set:
                pair.add(e)
            sum_count += 1
    
    # 优化：使用条件表达式，避免重复的长度计算
    mapping_len = len(mapping) if mapping else 1  # 避免除零错误
    
    # 计算各种统计指标
    align_loss_percentage = len(pair) / mapping_len * 100 if mapping_len > 0 else 0
    overlap_percentage = sum_count / mapping_len if mapping_len > 0 else 0
    
    # 优化：预计算实体总数，避免重复访问
    total_ent1 = len(data.ent1) if data else 0
    total_ent2 = len(data.ent2) if data else 0
    ent_loss1 = total_ent1 - len(has_nodes1)
    ent_loss2 = total_ent2 - len(has_nodes2)
    
    # 打印结果（保持原有功能）
    print("align loss", align_loss_percentage, "%  total", sum_count)
    print("overlap", overlap_percentage, "%total", sum_count)
    print("ent loss", ent_loss1)
    print("ent loss", ent_loss2)
    
    # 返回统计数据
    return {
        'align_loss_percentage': align_loss_percentage,
        'overlap_percentage': overlap_percentage,
        'total_sum': sum_count,
        'pair_count': len(pair),
        'mapping_count': mapping_len,
        'ent_loss1': ent_loss1,
        'ent_loss2': ent_loss2,
        'has_nodes1_count': len(has_nodes1),
        'has_nodes2_count': len(has_nodes2),
        'total_ent1': total_ent1,
        'total_ent2': total_ent2
    }


def eva(sim_mat, use_min=False):
    if use_min is True:
        predicted = np.argmin(sim_mat, axis=1)
    else:
        predicted = np.argmax(sim_mat, axis=1)
    cor = predicted == np.array(range(sim_mat.shape[0]))
    cor_num = np.sum(cor)
    print("Acc: " + str(cor_num) + ' / ' + str(len(cor)) + ' = ' + str(cor_num * 1.0 / len(cor)))

def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks

def search_faiss(index, query, num, bj, batch, top_k):
    t = time.time()
    hits = [0] * len(top_k)
    mr, mrr = 0, 0
    _, index_mat = index.search(query, top_k[-1])
    for i, ent_i in enumerate(batch):
        golden = ent_i
        vec = index_mat[i,]
        golden_index = np.where(vec == golden)[0]
        if len(golden_index) > 0:
            rank = golden_index[0]
            mr += (rank + 1)
            mrr += 1 / (rank + 1)
            for j in range(len(top_k)):
                if rank < top_k[j]:
                    hits[j] += 1
    print("alignment evaluating at batch {}, hits@{} = {} time = {:.3f} s ".
          format(bj, top_k, np.array(hits) / len(batch), time.time() - t))
    return np.array(hits), mrr, mr

def test_by_faiss_batch(embeds2, embeds1, top_k=[1, 5, 10], is_norm=True, batch_num=512):
    start = time.time()
    if is_norm:
        embeds1 = preprocessing.normalize(embeds1)
        embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]
    hits = np.array([0] * len(top_k))
    mr, mrr = 0, 0
    index = faiss.IndexFlatL2(dim)  # build the index
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(embeds1)  # add vectors to the index
    batches = task_divide(list(range(num)), batch_num)
    query_num = 0

    rest = []
    for bj, batch in enumerate(batches):
        query_num += len(batch)
        query = embeds2[batch, :]
        print("test_by_faiss_batch", bj)
        rest.append(search_faiss(index, query, embeds1.shape[0], bj, batch, top_k))
    for hits_, mrr_, mr_ in rest:
        hits += hits_
        mrr += mrr_
        mr += mr_
    mr /= num
    mrr /= num
    hits = hits / num
    mr = round(mr, 8)
    mrr = round(mrr, 8)
    for i in range(len(hits)):
        hits[i] = round(hits[i], 8)
    print("alignment results with faiss: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, total time = {:.3f} s ".
          format(top_k, hits, mr, mrr, time.time() - start))
    return hits, mrr, mr
