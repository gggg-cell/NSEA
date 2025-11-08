from dataset import *

import faiss
import scipy.spatial
from torch import Tensor

def get_hits_slow(em1, em2, test_pair, top_k=(1, 10)):
    em1 = em1.detach().numpy()
    em2 = em2.detach().numpy()
    Lvec = np.array([em1[e1] for e1, e2 in test_pair])
    Rvec = np.array([em2[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))



def my_dist_func(L, R, k=100):
    dim = len(L[0])
    print(f"初始化FAISS索引，维度: {dim}, 查询数: {len(L)}, 索引数: {len(R)}")
    torch.cuda.empty_cache()
    # faiss.normalize_L2(L)
    # faiss.normalize_L2(R)
    index = faiss.IndexFlatIP(dim)
    # index = faiss.index_cpu_to_all_gpus(index)
    print(f"添加 {len(R)} 个向量到索引...")
    index.add(R)
    print(f"执行 {len(L)} 个查询，top_k={k}...")
    D, I = index.search(L, k)
    print("FAISS搜索完成")
    return D,I

def get_hits(em1, em2, test_pair, top_k=(1, 5, 10, 50, 100), partition=1, norm=False, src_nodes=None, trg_nodes=None):
    if isinstance(em1, Tensor):
        em1 = em1.cpu().detach().numpy()
        em2 = em2.cpu().detach().numpy()
    if norm:
        # em1= norm_process(torch.from_numpy(em1)).detach().numpy()
        # em2= norm_process(torch.from_numpy(em2)).detach().numpy()
        em1 = em1 / np.linalg.norm(em1, axis=-1, keepdims=True)
        em2 = em2 / np.linalg.norm(em2, axis=-1, keepdims=True)

    def filter_pair(pair, src, trg):
        if src is None or trg is None:
            return pair
        src = set(src)
        trg = set(trg)
        return list(filter(lambda x: x[0] in src and x[1] in trg, pair))

    batch_size = len(test_pair) // partition
    print(batch_size)
    total_size = 0
    top_lr = [0] * len(top_k)
    for x in range(partition):
        left = x * batch_size
        right = left + batch_size if left + batch_size < len(test_pair) else len(test_pair)
        filtered = filter_pair(test_pair[left:right], src_nodes, trg_nodes)
        print(len(filtered))
        if len(filtered) == 0:
            continue
        total_size += len(filtered)
        Lvec = np.array([em1[e1] for e1, e2 in filtered])
        Rvec = np.array([em2[e2] for e1, e2 in filtered])
        ranks = my_dist_func(Lvec, Rvec)
        for i in range(Lvec.shape[0]):
            rank = ranks[i]
            rank_index = np.where(rank == i)[0][0] if i in rank else 1000
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1
    print('For each left:')
    print('Total size=', total_size)
    for i in range(len(top_lr)):
        str = 'Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / (total_size + 1e-8) * 100)
        with open('hits.txt', 'a+') as f:
            f.write(str + '\n')
            f.close
        print(str)

    return top_k, top_lr, total_size
    
def test(test_pair, features, top_k=200, iteration=15, min_precision=1e-12, temperature=0.02):
    """
    优化版本：确保Sinkhorn迭代稳定运行
    
    Args:
        test_pair: 测试对
        features: 特征矩阵
        top_k: 候选数量
        iteration: Sinkhorn迭代次数
        min_precision: 最小精度阈值，低于此值会被设为此值
        temperature: 温度参数
    """
    
    # 内存清理
    print("🧹 清理内存和显存...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ PyTorch GPU缓存已清理")
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph() if hasattr(tf.compat.v1, 'reset_default_graph') else None
        print("✓ TensorFlow GPU内存已清理")
    except (ImportError, AttributeError):
        pass
     
    import gc
    gc.collect()
    print("✓ Python垃圾回收完成")
    
    left, right = test_pair[:,0], np.unique(test_pair[:,1])
    print(f"Test函数 - left: {len(left)}, right: {len(right)}, top_k: {top_k}")
    
    # 获取特征
    features_l = features[left]
    features_r = features[right]
    print(f"特征维度 - left: {features_l.shape}, right: {features_r.shape}")
    
    # 归一化
    faiss.normalize_L2(features_l)
    faiss.normalize_L2(features_r)
    
    # 动态调整top_k，确保有足够的候选
    min_candidates = max(50, len(np.unique(test_pair[:,1])) // 10)  # 至少50个候选或目标数量的10%
    safe_top_k = min(max(top_k, min_candidates), len(features_r))
    
    if safe_top_k < top_k:
        print(f"⚠️ top_k调整: {top_k} -> {safe_top_k} (受限于候选数量)")
    elif safe_top_k > top_k:
        print(f"📈 top_k扩展: {top_k} -> {safe_top_k} (确保足够候选)")
    
    print(f"最终top_k: {safe_top_k}")
    
    # FAISS检索
    dim = features_l.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(features_r)
    sims, indices = index.search(features_l, safe_top_k)
    
    print("开始稳定版Sinkhorn迭代...")
    
    # 计算相似度，并应用精度保护
    raw_sims = sims.flatten()
    
    # 检查并修复精度问题
    low_precision_mask = np.abs(raw_sims) < min_precision
    if np.any(low_precision_mask):
        num_low = np.sum(low_precision_mask)
        print(f"⚠️ 发现 {num_low} 个低精度值 (< {min_precision})，进行修正...")
        raw_sims[low_precision_mask] = np.sign(raw_sims[low_precision_mask]) * min_precision
    
    # 应用温度缩放
    row_sims = np.exp(raw_sims / temperature)
    
    # 检查是否有NaN或Inf
    if np.any(np.isnan(row_sims)) or np.any(np.isinf(row_sims)):
        print("⚠️ 发现NaN或Inf值，进行清理...")
        row_sims = np.nan_to_num(row_sims, nan=min_precision, posinf=1.0, neginf=min_precision)
    
    flat_indices = indices.astype(np.int32).flatten()
    
    size = len(left)
    total_elements = size * safe_top_k
    
    # 创建索引
    row_ids = np.repeat(np.arange(size), safe_top_k)
    
    print(f"Sinkhorn参数:")
    print(f"  - 迭代次数: {iteration}")
    print(f"  - 温度参数: {temperature}")
    print(f"  - 最小精度: {min_precision}")
    print(f"  - 矩阵大小: {size} x {len(features_r)}")
    print(f"  - 非零元素: {total_elements}")
    
    # 稳定的Sinkhorn迭代
    convergence_threshold = 1e-6
    prev_row_sims = row_sims.copy()
    
    for iter_num in range(iteration):
        # 行归一化 - 添加稳定性检查
        row_sums = np.bincount(row_ids, weights=row_sims, minlength=size)
        
        # 防止除零
        row_sums = np.maximum(row_sums, min_precision)
        row_normalizers = row_sums[row_ids]
        row_sims = row_sims / row_normalizers
        
        # 列归一化 - 添加稳定性检查  
        col_sums = np.bincount(flat_indices, weights=row_sims, minlength=len(features_r))
        
        # 防止除零
        col_sums = np.maximum(col_sums, min_precision)
        col_normalizers = col_sums[flat_indices]
        row_sims = row_sims / col_normalizers
        
        # 数值稳定性检查
        if np.any(np.isnan(row_sims)) or np.any(np.isinf(row_sims)):
            print(f"⚠️ 迭代 {iter_num + 1}: 发现数值不稳定，进行修正")
            row_sims = np.nan_to_num(row_sims, nan=min_precision, posinf=1.0, neginf=min_precision)
        
        # 收敛检查（可选）
        if iter_num > 0 and iter_num % 5 == 0:
            diff = np.mean(np.abs(row_sims - prev_row_sims))
            print(f"  迭代 {iter_num + 1}/{iteration} - 变化: {diff:.2e}")
            if diff < convergence_threshold:
                print(f"  ✓ 在迭代 {iter_num + 1} 达到收敛")
                break
        
        if (iter_num + 1) % 5 == 0:
            prev_row_sims = row_sims.copy()
    
    # 最终数值检查
    final_min = np.min(row_sims)
    final_max = np.max(row_sims)
    final_mean = np.mean(row_sims)
    print(f"Sinkhorn完成统计:")
    print(f"  - 值范围: [{final_min:.2e}, {final_max:.2e}]")
    print(f"  - 平均值: {final_mean:.2e}")
    print(f"  - NaN数量: {np.sum(np.isnan(row_sims))}")
    print(f"  - Inf数量: {np.sum(np.isinf(row_sims))}")
    
    # 重构结果
    final_indices = indices  # 保持原形状 [size, safe_top_k]
    final_sims = row_sims.reshape(size, safe_top_k)
    
    print(f"结果维度: indices={final_indices.shape}, sims={final_sims.shape}")
    
    # 计算排名
    ranks = np.argsort(-final_sims, axis=1)
    
    # 评估
    wrong_list, right_list = [], []
    h1, h10, mrr = 0, 0, 0
    pos = np.zeros(np.max(right)+1, dtype=int)
    pos[right] = np.arange(len(right))
    
    print("计算评估指标...")
    for i in range(len(test_pair)):
        rank = np.where(pos[test_pair[i,1]] == final_indices[i,ranks[i]])[0]
        if len(rank) != 0:
            if rank[0] == 0:
                h1 += 1
                right_list.append(test_pair[i])
            else:
                wrong_list.append((test_pair[i], right[final_indices[i,ranks[i]][0]]))
            if rank[0] < 10:
                h10 += 1
            mrr += 1/(rank[0]+1) 
    
    print(f"🎯 稳定版Sinkhorn Test结果:")
    print("Hits@1: %.3f Hits@10: %.3f MRR: %.3f\n"%(h1/len(test_pair), h10/len(test_pair), mrr/len(test_pair)))
    
    return right_list, wrong_list
