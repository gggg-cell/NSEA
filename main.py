import argparse
import copy

import numpy as np
import torch
from scipy.stats import rankdata
from framework import *
import time
from eval import sparse_acc,sparse_top_k, cur_max

# 导入智能缓存系统
try:
    from smart_memory_cache import smart_cache
    CACHE_AVAILABLE = True
    print("✅ 智能缓存系统已加载")
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ 智能缓存系统未找到，使用原始方式")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./mkdata/', help='path to dataset')
    parser.add_argument('--scale', type=str, default="large", help='scale of data')
    parser.add_argument('--step', type=int, default=3, help='step of data')
    parser.add_argument("--if_store", action='store_true', default=False)
    parser.add_argument("--result_folder", type=str, default='result/')
    parser.add_argument('--train_ratio', type=int, default=30)
    parser.add_argument("--top_k_corr", type=int, default=1)
    parser.add_argument("--k_partition", type=int, default=50)
    parser.add_argument("--backbone", type=str, default='duala')
    parser.add_argument("--method", type=str, default="ours",help="partition method")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs to train")
    parser.add_argument('--it_round', type=int, default=2)
    parser.add_argument('--round', type=int, default=10)
    parser.add_argument('--sbp', type=bool, default=False)
    parser.add_argument("--enhance", type=str, default='sinkhorn', help='mini-batch normalization')
    parser.add_argument("--save_folder", type=str, default='tmp/')
    parser.add_argument('--lang', type=str, default='fr', help='dataset language (fr, de)')
    parser.add_argument("--shuffle", type= bool, default=True, help="if shuffle data" )
    parser.add_argument("--src", type=int, default=0, help="which to train, 0 or 1")
    parser.add_argument("--norm", action="store_true", default=True, help="whether to normalize embeddings")
    parser.add_argument("--max_sinkhorn_sz", type=int, default=33000,
                        help="max matrix size to run Sinkhorn iteration"
                             ", if the matrix size is higher than this value"
                             ", it will calculate kNN search without normalizing to avoid OOM"
                             ", default is set for 33000^2 (for RTX3090)."
                             " could be set to higher value in case there is GPU with larger memory")
    return parser.parse_args()


global_arguments = get_arguments()
norm = global_arguments.norm
train_ratio = global_arguments.train_ratio
data = global_arguments.data
scale = global_arguments.scale
step = global_arguments.step
result_folder = global_arguments.result_folder
if_store = global_arguments.if_store
top_k_corr = global_arguments.top_k_corr
k_partiton = global_arguments.k_partition
backbone = global_arguments.backbone
max_sinkhorn_sz = global_arguments.max_sinkhorn_sz
method= global_arguments.method
lang = global_arguments.lang
device = global_arguments.device
enhance = global_arguments.enhance
epoch = global_arguments.epoch
n_semi_iter = global_arguments.it_round
sbp = global_arguments.sbp
save_folder = global_arguments.save_folder
shuffle_data = global_arguments.shuffle
src = global_arguments.src
if global_arguments.epoch == -1:
    train_epoch = \
        {'gcn-align': [2000] * n_semi_iter, 'rrea': [100] * n_semi_iter, 'dual-amn': [20] + [5] * (n_semi_iter - 1),
         'gcn-large': [50], 'dual-large': [20], 'rrea-large': [50], "duala": [20]}[
            backbone]

PHASE_PARTITION = 1  
PHASE_TRAINING = 2


def load_curr_objs(phase):
    try:
        return readobj(save_folder + get_suffix(phase))
    except:
        return readobj(save_folder + get_suffix(phase))

def get_suffix(phase, i=None):
    now = 'sim' if PHASE_TRAINING == phase else 'partition'
    if phase == PHASE_PARTITION:
        if i is not None:
            now += f"_{scale}_{method}_{lang}_shuffle{shuffle_data}_k{k_partiton}_ratio{train_ratio}_{i}.pkl"
        else:
            now += f"_{scale}_{method}_{lang}_shuffle{shuffle_data}_k{k_partiton}_ratio{train_ratio}.pkl"
        # now += ablation_args(sampler_methods, 'CST')
    elif phase == PHASE_TRAINING:
        if i is not None:
            now += f"_{scale}_{method}_{lang}_{backbone}_{enhance}_{train_ratio}_it{n_semi_iter}_{i}.pkl"
        else:
            now += f"_{scale}_{method}_{lang}_{backbone}_{enhance}_{train_ratio}_it{n_semi_iter}.pkl"
    else:
        raise NotImplementedError
    return now


def save_curr_objs(objs, phase,i=None):
    saveobj(objs, save_folder + get_suffix(phase, i))

def train(batch: AlignmentBatch, device: torch.device = 'cuda', **kwargs):
    # 支持两种参数传递方式：config_dict（缓存优化）和args（原始方式）
    if 'config_dict' in kwargs:
        config_dict = kwargs['config_dict']
        it_round = config_dict.get('it_round', 1)
    else:
        args = kwargs['args']
        it_round = args.it_round
    
    if hasattr(batch, 'model'): 
        model = batch.model
        try:
            for it in range(it_round):
                model.train1step(train_epoch[0])
                if it < it_round - 1:
                    model.mraea_iteration()
            return model.get_curr_embeddings()

        except Exception as e:
            print('TF error', str(e))
            return None
        #  TODO
        pass
        # #  TODO
        # pass
    else:
        raise NotImplementedError
    

def step1_partition():
    ds = load_dataset(data, scale,lang, train_ratio=train_ratio*0.01, shuffle=shuffle_data)
    for i in range(global_arguments.round):
        partition = Partition(ds, k=k_partiton, src=src)
    # ours_nodes1, ours_nodes2 = partition.split_clusters(method='past')
        tmp1_nodes, tmp2_nodes, src_nodes, trg_nodes = partition.split_clusters(method=method)
        yield ds, tmp1_nodes, tmp2_nodes, src_nodes, trg_nodes

def step2_embeding(ds, framework, src_nodes, trg_nodes, round_idx, node_type):

    batch_sim_folder = save_folder + 'batch_sims/'
    import os
    os.makedirs(batch_sim_folder, exist_ok=True)
    
    curr_sim = None
    use_sinkhorn = 0
    batch_idx = 0
    align_loss_data = None  # 存储align_loss统计数据
    
    for batch in framework.get_cluster_result(top_k_corr, backbone, src_nodes, trg_nodes, max_sinkhorn_sz):
        # 收集align_loss统计数据（每个batch都有相同的align_loss_stats）
        if hasattr(batch, 'align_loss_stats') and align_loss_data is None:
            align_loss_data = batch.align_loss_stats.copy()
            align_loss_data['round'] = round_idx + 1
            align_loss_data['node_type'] = node_type
            align_loss_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 使用缓存优化的训练参数传递
        if CACHE_AVAILABLE:
            # 使用配置字典替代深拷贝，减少内存开销
            embed = train(batch, device, config_dict={'it_round': global_arguments.it_round})
        else:
            embed = train(batch, device, args=copy.deepcopy(global_arguments))
        
        if embed is None:
            continue
        left_embeding, right_embedding = tuple(embed)
        batch_sim, curr_use_sinkhorn = batch.create_batch_sim(left_embeding, right_embedding, enhance, ds.size(),
                                                             norm=norm, return_use_sinkhorn=True)
        if batch_sim is None:
            continue
        use_sinkhorn += curr_use_sinkhorn
        print('Total sinkhorn=', use_sinkhorn)
        
        if curr_sim is None:
            curr_sim = batch_sim
        else:
            curr_sim = curr_sim + batch_sim        
            curr_sim = curr_sim.coalesce()

        del batch_sim
        torch.cuda.empty_cache()
    result = sparse_acc(curr_sim, ds.ill(ds.test, 'cuda'))
    print(f'Batch {batch_idx} accumulated acc is', result)
    return curr_sim, align_loss_data

def run():
    start_total = time.time()
    torch.cuda.set_device(0)
    
    # 创建时间统计字典
    time_stats = {
        'step1_partition': 0,
        'step2_embedding': 0, 
        'step3_evaluation': 0,
        'total_time': 0
    }
    
    # eval_large()
    if step <= 1:  
        step1_start = time.time()
        print("开始执行步骤1: 分区...")
        for index, nodes in enumerate(step1_partition()):
            save_curr_objs(nodes, PHASE_PARTITION,index)
        step1_end = time.time()
        time_stats['step1_partition'] = step1_end - step1_start
        print(f"步骤1完成，耗时: {time_stats['step1_partition']:.2f} 秒")

    if step <= 2:  # 如果step为1或2，执行步骤2
        step2_start = time.time()
        print("开始执行步骤2: 嵌入训练...")
        
        # 创建align_loss记录列表
        align_loss_records = []
        
        for i in range(global_arguments.round):
            print(f"\n--- 处理第 {i+1}/{global_arguments.round} 轮 ---")

            ds, tmp1_nodes, tmp2_nodes, src_nodes, trg_nodes = readobj(save_folder + get_suffix(PHASE_PARTITION,i))
            framework = LargepartitonFramework(ds, device='cuda', k=k_partiton, src=0, )
            
            # 执行嵌入训练（节点数据通过缓存减少传递开销）
            sim1, align_loss_data1 = step2_embeding(ds, framework, tmp1_nodes, tmp2_nodes, i, 'tmp')
            sim2, align_loss_data2 = step2_embeding(ds, framework, src_nodes, trg_nodes, i, 'src')
            
            # 收集align_loss数据
            if align_loss_data1:
                align_loss_records.append(align_loss_data1)
            if align_loss_data2:
                align_loss_records.append(align_loss_data2)
            
            save_curr_objs((framework,sim1,sim2), PHASE_TRAINING, i)
            del sim1, sim2
            torch.cuda.empty_cache()
        
        # 保存align_loss数据到本地文件
        if align_loss_records:
            import json
            import os
            from datetime import datetime
            
            # 确保结果文件夹存在
            os.makedirs(result_folder, exist_ok=True)
            
            # 创建详细的align_loss报告
            align_loss_report = {
                'experiment_info': {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'scale': scale,
                    'method': method,
                    'lang': lang,
                    'backbone': backbone,
                    'train_ratio': train_ratio,
                    'k_partition': k_partiton,
                    'total_rounds': global_arguments.round,
                    'device': device
                },
                'align_loss_records': align_loss_records,
                'summary': {
                    'total_records': len(align_loss_records),
                    'avg_align_loss_percentage': sum([r['align_loss_percentage'] for r in align_loss_records]) / len(align_loss_records) if align_loss_records else 0,
                    'avg_overlap_percentage': sum([r['overlap_percentage'] for r in align_loss_records]) / len(align_loss_records) if align_loss_records else 0,
                    'avg_ent_loss1': sum([r['ent_loss1'] for r in align_loss_records]) / len(align_loss_records) if align_loss_records else 0,
                    'avg_ent_loss2': sum([r['ent_loss2'] for r in align_loss_records]) / len(align_loss_records) if align_loss_records else 0
                }
            }
            
            # 保存JSON格式的align_loss记录
            align_loss_json_file = os.path.join(result_folder, f'align_loss_records_{scale}_{method}_{lang}.json')
            with open(align_loss_json_file, 'w', encoding='utf-8') as f:
                json.dump(align_loss_report, f, indent=2, ensure_ascii=False)
            
            # 保存简单的文本格式align_loss记录
            align_loss_txt_file = os.path.join(result_folder, f'align_loss_report_{scale}_{method}_{lang}.txt')
            with open(align_loss_txt_file, 'w', encoding='utf-8') as f:
                f.write("Align Loss 报告\n")
                f.write("="*70 + "\n")
                f.write(f"实验时间: {align_loss_report['experiment_info']['timestamp']}\n")
                f.write(f"数据规模: {scale}\n")
                f.write(f"方法: {method}\n")
                f.write(f"语言: {lang}\n")
                f.write(f"骨干网络: {backbone}\n")
                f.write(f"训练比例: {train_ratio}%\n")
                f.write(f"分区数: {k_partiton}\n")
                f.write(f"总轮次: {global_arguments.round}\n")
                f.write(f"设备: {device}\n")
                f.write("-" * 70 + "\n")
                f.write("各轮次Align Loss详情:\n")
                for i, record in enumerate(align_loss_records):
                    f.write(f"\n记录 {i+1} (第{record['round']}轮, {record['node_type']}节点):\n")
                    f.write(f"  时间戳: {record['timestamp']}\n")
                    f.write(f"  Align Loss百分比: {record['align_loss_percentage']:.4f}%\n")
                    f.write(f"  重叠百分比: {record['overlap_percentage']:.4f}%\n")
                    f.write(f"  配对数量: {record['pair_count']}\n")
                    f.write(f"  映射总数: {record['mapping_count']}\n")
                    f.write(f"  总和: {record['total_sum']}\n")
                    f.write(f"  实体损失1: {record['ent_loss1']}\n")
                    f.write(f"  实体损失2: {record['ent_loss2']}\n")
                    f.write(f"  节点数1: {record['has_nodes1_count']}/{record['total_ent1']}\n")
                    f.write(f"  节点数2: {record['has_nodes2_count']}/{record['total_ent2']}\n")
                f.write("-" * 70 + "\n")
                f.write("统计摘要:\n")
                f.write(f"总记录数: {align_loss_report['summary']['total_records']}\n")
                f.write(f"平均Align Loss百分比: {align_loss_report['summary']['avg_align_loss_percentage']:.4f}%\n")
                f.write(f"平均重叠百分比: {align_loss_report['summary']['avg_overlap_percentage']:.4f}%\n")
                f.write(f"平均实体损失1: {align_loss_report['summary']['avg_ent_loss1']:.2f}\n")
                f.write(f"平均实体损失2: {align_loss_report['summary']['avg_ent_loss2']:.2f}\n")
                f.write("="*70 + "\n")
            
            # 保存CSV格式的align_loss记录
            align_loss_csv_file = os.path.join(result_folder, f'align_loss_data_{scale}_{method}_{lang}.csv')
            with open(align_loss_csv_file, 'w', encoding='utf-8') as f:
                f.write("轮次,节点类型,Align_Loss百分比,重叠百分比,配对数量,映射总数,总和,实体损失1,实体损失2,时间戳\n")
                for record in align_loss_records:
                    f.write(f"{record['round']},{record['node_type']},{record['align_loss_percentage']:.6f},"
                           f"{record['overlap_percentage']:.6f},{record['pair_count']},{record['mapping_count']},"
                           f"{record['total_sum']},{record['ent_loss1']},{record['ent_loss2']},{record['timestamp']}\n")
            
            print(f"\nAlign Loss记录已保存到:")
            print(f"详细报告 (JSON): {align_loss_json_file}")
            print(f"文本报告 (TXT):  {align_loss_txt_file}")
            print(f"数据文件 (CSV):  {align_loss_csv_file}")
        
        step2_end = time.time()
        time_stats['step2_embedding'] = step2_end - step2_start
        print(f"步骤2完成，耗时: {time_stats['step2_embedding']:.2f} 秒")

    if step <= 3:  # 如果step为1、2或3，执行步骤3
        step3_start = time.time()
        print("开始执行步骤3: 最终评估...")
        sum_matrix = None
        framework = None
        
        # 创建精确度记录列表
        accuracy_records = []
        
        for i in range(global_arguments.round):
            try:
                framework, sim1, sim2 = readobj(save_folder + get_suffix(PHASE_TRAINING, i))
                framework.device = device
                print(f'成功加载第{i}轮的framework和相似度矩阵')
            except Exception as e:
                print(f'加载第{i}轮数据失败: {e}')
                continue
            
            # 累加当前轮次的相似度矩阵
            round_sum = None
            
            # 处理sim1（tmp节点的相似度矩阵）
            if sim1 is not None:
                sim1 = sim1.to(device)
                round_sum = sim1
                print(f'第{i}轮添加sim1，大小: {sim1.size()}')
                torch.cuda.empty_cache()

            # 处理sim2（src节点的相似度矩阵）
            if sim2 is not None:
                sim2 = sim2.to(device)
                if round_sum is None:
                    round_sum = sim2
                else:
                    round_sum = round_sum + sim2
                    round_sum = round_sum.coalesce()
                torch.cuda.empty_cache()
                print(f'第{i}轮添加sim2，大小: {sim2.size()}')
            
            # 将当前轮次的累加结果添加到总和中
            if round_sum is not None:
                if sum_matrix is None:
                    sum_matrix = round_sum
                else:
                    # 在GPU上累加矩阵
                    sum_matrix = sum_matrix + round_sum
                    sum_matrix = sum_matrix.coalesce()
                
                # 评估当前结果并记录精确度
                print(f'第{i}轮累加后评估结果:')
                eval_result = framework.eval_sim(sum_matrix)
                
                # 提取所有hits指标
                hits_metrics = extract_hits_metrics(eval_result)
                
                # 记录当前轮次的精确度（包含所有hits指标和MRR）
                round_record = {
                    'round': i + 1,
                    'hits@1': hits_metrics['hits@1'],
                    'hits@5': hits_metrics['hits@5'], 
                    'hits@10': hits_metrics['hits@10'],
                    'MRR': hits_metrics['MRR'],
                    'accuracy': hits_metrics['hits@1'],  # 保持向后兼容
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                accuracy_records.append(round_record)
                
                print(f'第{i+1}轮累加精确度: Hits@1={hits_metrics["hits@1"]:.4f}, Hits@5={hits_metrics["hits@5"]:.4f}, Hits@10={hits_metrics["hits@10"]:.4f}, MRR={hits_metrics["MRR"]:.4f}')
                
                # 释放不再需要的GPU内存
                del sim1, sim2, round_sum
                torch.cuda.empty_cache()
            else:
                print(f'第{i}轮没有有效的相似度矩阵')
                # 即使没有有效矩阵，也记录一个0精确度
                round_record = {
                    'round': i + 1,
                    'hits@1': 0.0,
                    'hits@5': 0.0,
                    'hits@10': 0.0,
                    'MRR': 0.0,
                    'accuracy': 0.0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                accuracy_records.append(round_record)
        
        # 最终评估
        final_hits_metrics = {'hits@1': 0.0, 'hits@5': 0.0, 'hits@10': 0.0, 'MRR': 0.0}
        
        if sum_matrix is not None and framework is not None:
            print('所有轮次累加完成，最终评估结果:')
            final_result = framework.eval_sim(sum_matrix)
            # 提取最终的hits指标
            final_hits_metrics = extract_hits_metrics(final_result)
            test_src_indices = np.array([pair[0] for pair in framework.ds.test])
            test_tgt_indices = np.array([pair[1]+len(framework.ds.ent1) for pair in framework.ds.test])
            test_pair = np.column_stack([test_src_indices, test_tgt_indices])
            
            # # 使用evaluation.py中更新后的test函数处理稀疏相似度矩阵
            # # 使用较小的top_k值以减少内存使用
            # framework.eval_sim(sum_matrix, top_k=100, iteration=15, temperature=1)

            print(f'✅ 最终评估结果:')
            print(f'   Hits@1:  {final_hits_metrics["hits@1"]:.6f}')
            print(f'   Hits@5:  {final_hits_metrics["hits@5"]:.6f}')
            print(f'   Hits@10: {final_hits_metrics["hits@10"]:.6f}')
            print(f'   MRR:     {final_hits_metrics["MRR"]:.6f}')
        else:
            print('没有找到任何有效的相似度矩阵或framework')
            print('使用默认的0值作为最终结果')
        
        # 保存精确度记录到本地文件
        import json
        import os
        from datetime import datetime
        
        # 确保结果文件夹存在
        os.makedirs(result_folder, exist_ok=True)
        
        # 创建详细的精确度报告
        acc_report = {
            'experiment_info': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'scale': scale,
                'method': method,
                'lang': lang,
                'backbone': backbone,
                'train_ratio': train_ratio,
                'k_partition': k_partiton,
                'total_rounds': global_arguments.round,
                'device': device
            },
            'round_accuracies': accuracy_records,
            'final_metrics': {
                'hits@1': float(final_hits_metrics['hits@1']),
                'hits@5': float(final_hits_metrics['hits@5']),
                'hits@10': float(final_hits_metrics['hits@10']),
                'MRR': float(final_hits_metrics['MRR'])
            },
            'final_accuracy': float(final_hits_metrics['hits@1']),  # 保持向后兼容
            'summary': {
                'hits@1': {
                    'max': max([r['hits@1'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'min': min([r['hits@1'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'avg': sum([r['hits@1'] for r in accuracy_records]) / len(accuracy_records) if accuracy_records else 0.0
                },
                'hits@5': {
                    'max': max([r['hits@5'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'min': min([r['hits@5'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'avg': sum([r['hits@5'] for r in accuracy_records]) / len(accuracy_records) if accuracy_records else 0.0
                },
                'hits@10': {
                    'max': max([r['hits@10'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'min': min([r['hits@10'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'avg': sum([r['hits@10'] for r in accuracy_records]) / len(accuracy_records) if accuracy_records else 0.0
                },
                'MRR': {
                    'max': max([r['MRR'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'min': min([r['MRR'] for r in accuracy_records]) if accuracy_records else 0.0,
                    'avg': sum([r['MRR'] for r in accuracy_records]) / len(accuracy_records) if accuracy_records else 0.0
                },
                # 保持向后兼容
                'max_accuracy': max([r['hits@1'] for r in accuracy_records]) if accuracy_records else 0.0,
                'min_accuracy': min([r['hits@1'] for r in accuracy_records]) if accuracy_records else 0.0,
                'avg_accuracy': sum([r['hits@1'] for r in accuracy_records]) / len(accuracy_records) if accuracy_records else 0.0
            }
        }
        
        # 保存JSON格式的精确度记录
        acc_json_file = os.path.join(result_folder, f'accuracy_records_{scale}_{method}_{lang}.json')
        with open(acc_json_file, 'w', encoding='utf-8') as f:
            json.dump(acc_report, f, indent=2, ensure_ascii=False)
        
        # 保存简单的文本格式精确度记录
        acc_txt_file = os.path.join(result_folder, f'accuracy_report_{scale}_{method}_{lang}.txt')
        with open(acc_txt_file, 'w', encoding='utf-8') as f:
            f.write("精确度报告 (包含Hits@1/5/10和MRR指标)\n")
            f.write("="*70 + "\n")
            f.write(f"实验时间: {acc_report['experiment_info']['timestamp']}\n")
            f.write(f"数据规模: {scale}\n")
            f.write(f"方法: {method}\n")
            f.write(f"语言: {lang}\n")
            f.write(f"骨干网络: {backbone}\n")
            f.write(f"训练比例: {train_ratio}%\n")
            f.write(f"分区数: {k_partiton}\n")
            f.write(f"总轮次: {global_arguments.round}\n")
            f.write(f"设备: {device}\n")
            f.write("-" * 70 + "\n")
            f.write("最终评估结果:\n")
            f.write(f"  Hits@1:  {acc_report['final_metrics']['hits@1']:.6f}\n")
            f.write(f"  Hits@5:  {acc_report['final_metrics']['hits@5']:.6f}\n")
            f.write(f"  Hits@10: {acc_report['final_metrics']['hits@10']:.6f}\n")
            f.write(f"  MRR:     {acc_report['final_metrics']['MRR']:.6f}\n")
            f.write("-" * 70 + "\n")
            f.write("各轮次精确度:\n")
            f.write(f"{'轮次':>4} {'Hits@1':>10} {'Hits@5':>10} {'Hits@10':>11} {'MRR':>10} {'时间戳':>20}\n")
            f.write("-" * 70 + "\n")
            for record in accuracy_records:
                f.write(f"{record['round']:>4} {record['hits@1']:>10.6f} {record['hits@5']:>10.6f} {record['hits@10']:>11.6f} {record['MRR']:>10.6f} {record['timestamp']:>20}\n")
            f.write("-" * 70 + "\n")
            f.write("统计摘要:\n")
            f.write("Hits@1:\n")
            f.write(f"  最高: {acc_report['summary']['hits@1']['max']:.6f}\n")
            f.write(f"  最低: {acc_report['summary']['hits@1']['min']:.6f}\n") 
            f.write(f"  平均: {acc_report['summary']['hits@1']['avg']:.6f}\n")
            f.write("Hits@5:\n")
            f.write(f"  最高: {acc_report['summary']['hits@5']['max']:.6f}\n")
            f.write(f"  最低: {acc_report['summary']['hits@5']['min']:.6f}\n")
            f.write(f"  平均: {acc_report['summary']['hits@5']['avg']:.6f}\n")
            f.write("Hits@10:\n")
            f.write(f"  最高: {acc_report['summary']['hits@10']['max']:.6f}\n")
            f.write(f"  最低: {acc_report['summary']['hits@10']['min']:.6f}\n")
            f.write(f"  平均: {acc_report['summary']['hits@10']['avg']:.6f}\n")
            f.write("MRR:\n")
            f.write(f"  最高: {acc_report['summary']['MRR']['max']:.6f}\n")
            f.write(f"  最低: {acc_report['summary']['MRR']['min']:.6f}\n")
            f.write(f"  平均: {acc_report['summary']['MRR']['avg']:.6f}\n")
            f.write("="*70 + "\n")
        
        # 保存CSV格式的精确度记录（便于Excel打开）
        acc_csv_file = os.path.join(result_folder, f'accuracy_data_{scale}_{method}_{lang}.csv')
        with open(acc_csv_file, 'w', encoding='utf-8') as f:
            f.write("轮次,Hits@1,Hits@5,Hits@10,MRR,时间戳\n")
            for record in accuracy_records:
                f.write(f"{record['round']},{record['hits@1']:.6f},{record['hits@5']:.6f},{record['hits@10']:.6f},{record['MRR']:.6f},{record['timestamp']}\n")
            f.write(f"最终,{acc_report['final_metrics']['hits@1']:.6f},{acc_report['final_metrics']['hits@5']:.6f},{acc_report['final_metrics']['hits@10']:.6f},{acc_report['final_metrics']['MRR']:.6f},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\n精确度记录已保存到:")
        print(f"详细报告 (JSON): {acc_json_file}")
        print(f"文本报告 (TXT):  {acc_txt_file}")
        print(f"数据文件 (CSV):  {acc_csv_file}")
        
        step3_end = time.time()
        time_stats['step3_evaluation'] = step3_end - step3_start
        print(f"步骤3完成，耗时: {time_stats['step3_evaluation']:.2f} 秒")

    end_total = time.time()
    time_stats['total_time'] = end_total - start_total
    
    # 打印时间统计
    print("\n" + "="*50)
    print("时间统计报告:")
    print("="*50)
    print(f"步骤1 (分区):     {time_stats['step1_partition']:.2f} 秒")
    print(f"步骤2 (嵌入训练): {time_stats['step2_embedding']:.2f} 秒") 
    print(f"步骤3 (最终评估): {time_stats['step3_evaluation']:.2f} 秒")
    print(f"总耗时:          {time_stats['total_time']:.2f} 秒")
    print("="*50)
    
    # 保存时间统计到本地文件
    import json
    import os
    from datetime import datetime
    
    # 确保结果文件夹存在
    os.makedirs(result_folder, exist_ok=True)
    
    # 添加更多详细信息到统计中
    detailed_stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'scale': scale,
            'method': method,
            'lang': lang,
            'backbone': backbone,
            'train_ratio': train_ratio,
            'k_partition': k_partiton,
            'rounds': global_arguments.round,
            'device': device
        },
        'timing': time_stats
    }
    
    # 保存JSON格式的详细统计
    json_file = os.path.join(result_folder, f'timing_stats_{scale}_{method}_{lang}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
    
    # 保存简单的文本格式统计
    txt_file = os.path.join(result_folder, f'timing_report_{scale}_{method}_{lang}.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("时间统计报告\n")
        f.write("="*50 + "\n")
        f.write(f"运行时间: {detailed_stats['timestamp']}\n")
        f.write(f"数据规模: {scale}\n")
        f.write(f"方法: {method}\n") 
        f.write(f"语言: {lang}\n")
        f.write(f"骨干网络: {backbone}\n")
        f.write(f"训练比例: {train_ratio}%\n")
        f.write(f"分区数: {k_partiton}\n")
        f.write(f"轮次: {global_arguments.round}\n")
        f.write(f"设备: {device}\n")
        f.write("-" * 50 + "\n")
        f.write(f"步骤1 (分区):     {time_stats['step1_partition']:.2f} 秒\n")
        f.write(f"步骤2 (嵌入训练): {time_stats['step2_embedding']:.2f} 秒\n")
        f.write(f"步骤3 (最终评估): {time_stats['step3_evaluation']:.2f} 秒\n")
        f.write(f"总耗时:          {time_stats['total_time']:.2f} 秒\n")
        f.write("="*50 + "\n")
    
    print(f"\n时间统计已保存到:")
    print(f"详细统计 (JSON): {json_file}")
    print(f"简单报告 (TXT):  {txt_file}")
    
    # 输出缓存统计信息和清理
    if CACHE_AVAILABLE:
        print("\n" + "="*60)
        print("🚀 智能缓存系统统计报告")
        print("="*60)
        smart_cache.print_stats()
        
        # 最终清理所有缓存
        print("\n🧹 清理所有缓存...")
        smart_cache.clear_all_cache()
        print("✅ 缓存清理完成")
    
    return time_stats
        
def extract_hits_metrics(eval_result, default_value=0.0):
    """
    从评估结果中提取hits@1, hits@5, hits@10, MRR指标
    
    Args:
        eval_result: 评估函数的返回结果
        default_value: 默认值（当无法提取时）
        
    Returns:
        dict: 包含hits@1, hits@5, hits@10, MRR的字典
    """
    hits_metrics = {
        'hits@1': default_value,
        'hits@5': default_value, 
        'hits@10': default_value,
        'MRR': default_value
    }
    
    try:
        if isinstance(eval_result, tuple) and len(eval_result) >= 2:
            acc_result = eval_result[1]
            if isinstance(acc_result, dict):
                # 如果是字典，直接提取hits指标和MRR
                for key in ['hits@1', 'hits@5', 'hits@10']:
                    if key in acc_result:
                        hits_metrics[key] = float(acc_result[key])
                
                # 处理MRR指标（可能的键名：MRR, mrr, mean_reciprocal_rank）
                for mrr_key in ['MRR', 'mrr', 'mean_reciprocal_rank']:
                    if mrr_key in acc_result:
                        hits_metrics['MRR'] = float(acc_result[mrr_key])
                        break
                
                print(f"✅ 提取到完整指标: {hits_metrics}")
            else:
                # 如果不是字典，将单个值赋给hits@1
                hits_metrics['hits@1'] = float(acc_result)
                print(f"📊 使用单一精确度值作为hits@1: {hits_metrics['hits@1']}")
        elif isinstance(eval_result, (int, float)):
            hits_metrics['hits@1'] = float(eval_result)
            print(f"📊 使用数值结果作为hits@1: {hits_metrics['hits@1']}")
        else:
            print(f"⚠️ 无法识别的评估结果格式: {type(eval_result)}")
    except Exception as e:
        print(f"❌ 提取指标时出错: {e}")
    
    return hits_metrics


def align_loss():
    with open(f'align_loss_result.csv', 'w') as f:
        f.write('data,lang,train_ratio,align_loss,ent_loss1,ent_loss2\n')
    for data in [ 'DBpedia1M']:
        if data == 'mkdata':
            scale = 'large'
        else:
            scale = 'largegnn'
        for lang in ['fr', 'de']:
            for train_ratio in range(10, 31, 5):
                ds = load_dataset(data, scale,lang, train_ratio=train_ratio*0.01, shuffle=shuffle_data)
                partition = Partition(ds, k=k_partiton, src=src)
                result = partition.split_clusters(method="align_loss")
                #将损失保存到文件中 
                with open(f'align_loss_result.csv', 'a') as f:
                    f.write(f'{data},{lang},{train_ratio},{result["align_loss"]},{result["ent_loss1"]},{result["ent_loss2"]}\n')

if __name__ == '__main__':
    align_loss();
