import time
import math
import itertools
from itertools import accumulate, repeat, chain
from contextlib import redirect_stdout
import sys
import os
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from sklearn.covariance import LedoitWolf
import csv
import torch
from torch.utils.data import TensorDataset
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
import pandas as pd
import networkx as nx
from src.utils.metrics import compute_precision_recall_f1, shd_metric as shd_metric_standard
sys.path.insert(0, '/mnt/shared-storage-user/pengbo/created/projects/CausalDiscovery/test_method')
from mwu import to_2d, from_key_to_info, from_path_to_info
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合在服务器上运行
import seaborn as sns

# Import causal-learn for FCI
try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz, chisq, gsq, kci
    FCI_AVAILABLE = True
except ImportError:
    FCI_AVAILABLE = False
    print("Warning: causal-learn not available. FCI method will not work.")


def check_cycles_method1(adj_matrix):
    """
    方法1: 使用NetworkX检测环（最常用）
    """
    # 将numpy数组转换为NetworkX有向图
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # 检测是否为有向无环图(DAG)
    is_dag = nx.is_directed_acyclic_graph(G)
    has_cycles = not is_dag
    
    # print(f"图是否有环: {has_cycles}")
    # print(f"图是否为DAG: {is_dag}")
    # if not has_cycles:
    #     print(adj_matrix)
    
    return has_cycles, G

def visualize_graph_comparison(true_adj, pred_adj, sample_id, output_path, threshold=0.5):
    """
    使用热图可视化真实图和预测图的对比
    
    Parameters:
    -----------
    true_adj: ndarray
        真实的邻接矩阵
    pred_adj: ndarray
        预测的邻接矩阵
    sample_id: str
        样本ID
    output_path: str
        输出图片路径
    threshold: float
        用于二值化的阈值
    """
    # 二值化预测图
    pred_binary = (pred_adj > threshold).astype(int)
    
    # 计算统计信息
    n_nodes = true_adj.shape[0]
    true_edges = int(true_adj.sum())
    pred_edges = int(pred_binary.sum())
    
    # 计算正确、错误的边
    correct = np.sum((true_adj == 1) & (pred_binary == 1))
    false_positive = np.sum((true_adj == 0) & (pred_binary == 1))
    false_negative = np.sum((true_adj == 1) & (pred_binary == 0))
    
    # 创建图形 - 3个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 设置字体大小
    title_fontsize = 12
    label_fontsize = 8 if n_nodes > 20 else 10
    
    # 子图1: 真实图
    sns.heatmap(true_adj, annot=False, fmt='d', cmap='Reds', 
                cbar_kws={'label': 'Edge'}, square=True,
                linewidths=0.5 if n_nodes <= 20 else 0,
                ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title(f'Ground Truth\n(Edges: {true_edges}, Nodes: {n_nodes})', 
                      fontsize=title_fontsize, fontweight='bold')
    axes[0].set_xlabel('To Node', fontsize=label_fontsize)
    axes[0].set_ylabel('From Node', fontsize=label_fontsize)
    
    # 子图2: 预测图（连续值）
    im = sns.heatmap(pred_adj, annot=False, cmap='Blues', 
                     cbar_kws={'label': 'Prediction Score'}, square=True,
                     linewidths=0.5 if n_nodes <= 20 else 0,
                     ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f'Predicted (Continuous)\n(Threshold: {threshold:.2f})', 
                      fontsize=title_fontsize, fontweight='bold')
    axes[1].set_xlabel('To Node', fontsize=label_fontsize)
    axes[1].set_ylabel('From Node', fontsize=label_fontsize)
    
    # 子图3: 预测图（二值化）+ 对比
    # 创建一个三值矩阵：0=无边，1=正确预测，2=错误预测，3=漏检
    comparison = np.zeros_like(true_adj, dtype=float)
    comparison[(true_adj == 1) & (pred_binary == 1)] = 2  # 正确预测 (绿色)
    comparison[(true_adj == 0) & (pred_binary == 1)] = 1  # 假阳性 (橙色)
    comparison[(true_adj == 1) & (pred_binary == 0)] = 3  # 假阴性 (红色)
    
    # 自定义颜色映射
    from matplotlib.colors import ListedColormap
    colors = ['white', 'orange', 'green', 'red']
    cmap = ListedColormap(colors)
    
    sns.heatmap(comparison, annot=False, cmap=cmap, 
                cbar_kws={'label': 'Status', 'ticks': [0.5, 1.5, 2.5, 3.5]},
                square=True, linewidths=0.5 if n_nodes <= 20 else 0,
                ax=axes[2], vmin=0, vmax=4)
    
    # 修改colorbar标签
    colorbar = axes[2].collections[0].colorbar
    colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    colorbar.set_ticklabels(['No Edge', 'False Pos', 'Correct', 'False Neg'])
    
    axes[2].set_title(f'Predicted (Binary)\n(Correct: {correct}, FP: {false_positive}, FN: {false_negative})', 
                      fontsize=title_fontsize, fontweight='bold')
    axes[2].set_xlabel('To Node', fontsize=label_fontsize)
    axes[2].set_ylabel('From Node', fontsize=label_fontsize)
    
    # 添加整体标题
    fig.suptitle(f'Causal Graph Heatmap Comparison: {sample_id}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"  ✓ Saved visualization: {output_path}")
def compute_features(x):
    """
    Implementation depends on matrix size
    x: (num_vars, num_samples)
    """
    if x.shape[0] < 100:
        return np.linalg.pinv(np.cov(x), rcond=1e-10)
    lw = LedoitWolf()
    lw.fit(x.T)
    invcovs = lw.get_precision()
    return invcovs

def to_1d(a):
    n = a.shape[0]
    mask = np.tri(n, k=-1, dtype=bool)
    forward = a[mask]
    backward = a.T[mask]
    return forward, backward

def cartesian_prod(arrays):
    """
        https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/49445693#49445693
    """
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *accumulate(chain((arr,), repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *itertools.repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)

def read_csv(fp, fieldnames=None, delimiter=',', str_keys=[]):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        # iterate and append
        for item in reader:
            data.append(item)
    return data

def sdcd_predict(data, interv, **kwargs):
    mask_interventions_oh = 1 - interv

    # === 2. 标准化数据 ===
    data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

    # === 3. 转换为与 create_intervention_dataset 相同的格式 ===
    # 将数据转换为 torch tensor
    X = torch.FloatTensor(data_normalized.astype(float))
    # print(2)
    # 创建 mask_interventions_oh (regime format)
    # regime format: 1 表示未干预，0 表示已干预
    # mask_interventions_oh = np.ones((len(data), data.shape[1]), dtype=int)
    # for i, regime in enumerate(regimes):
    #     for node in regime:
    #         mask_interventions_oh[i, node] = 0  # 被干预的节点设为0

    mask_interventions_oh = torch.LongTensor(mask_interventions_oh)
    # print(3)
    # 计算每个样本的干预节点数量
    n_regimes = torch.LongTensor(data.shape[1] - mask_interventions_oh.sum(axis=1))

    # === 4. 创建 Dataset (与 create_intervention_dataset 输出格式相同) ===
    X_dataset = TensorDataset(X, mask_interventions_oh, n_regimes)

    # # === 验证 ===
    # print(f"X shape: {X.shape}")
    # print(f"mask_interventions_oh shape: {mask_interventions_oh.shape}")
    # print(f"n_regimes shape: {n_regimes.shape}")
    # print(f"Dataset length: {len(X_dataset)}")
    # print(f"\nExample - First sample:")
    # print(f"  Data: {X_dataset[0][0][:5]}...")  # 前5个特征
    # print(f"  Mask (1=unperturbed, 0=perturbed): {X_dataset[0][1]}")
    # print(f"  Number of interventions: {X_dataset[0][2]}")
    model = SDCD()
    model.train(X_dataset, finetune=True)
    pred = model.get_adjacency_matrix(threshold=False)
    return pred

def shd_metric(pred, target):
    """
    Calculates the structural hamming distance

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    true_labels = target
    predictions = pred

    diff = true_labels - predictions

    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn + fp + rev

def compute_additional_metrics(true, pred, threshold=0.5):
    # convert to 2d and get aligned edges
    true, pred = np.array(true, dtype=int), np.array(pred)
    if true.ndim == 1:
        true, true_f, true_b = to_2d(true)
        pred, pred_f, pred_b = to_2d(pred)
    else:
        true_f, true_b = to_1d(true)
        pred_f, pred_b = to_1d(pred)
    pred_bin = (pred > threshold).astype(int)
    # compute metrics now
    # cycle, _ = check_cycles_method1(pred_bin)
    shd = shd_metric(pred_bin, true)
    true_mask = (true_f + true_b) > 0
    true_direction = (true_f[true_mask] > true_b[true_mask])
    pred_forward = (pred_f[true_mask] > pred_b[true_mask])
    pred_backward = (pred_f[true_mask] < pred_b[true_mask])
    edge_acc = (true_direction == pred_forward)[true_direction].sum() + (~true_direction == pred_backward)[~true_direction].sum()
    edge_acc = edge_acc / len(true_direction)
        
    return shd, edge_acc, _

# def compute_correlation(data, interv, **kwargs):
#     # compute correlation
#     corrs = np.corrcoef(data.T)
#     corrs = np.abs(corrs)
#     # print(corrs)
#     # exit()
#     assert not np.isnan(corrs).any(), f"{fp}: pred contains NaN, {corrs}"
#     return corrs

def compute_correlation(data, interv, **kwargs):
    # 检查数据中是否有常数列
    std_values = np.std(data, axis=0)
    constant_cols = np.where(std_values == 0)[0]
    # assert len(constant_cols) == 0, f"数据中存在常数列，索引: {constant_clos}"
    
    # 检查数据中是否有 NaN
    # assert not np.isnan(data).any(), f"数据中包含 NaN 值"
    
    # compute correlation
    corrs = np.corrcoef(data.T)
    corrs = np.abs(corrs)
    
    # 检查相关系数矩阵是否有 NaN
    if np.isnan(corrs).any():
        print(f"相关系数矩阵包含 NaN.")
    # 将 NaN 替换为 0（表示无相关性）
    corrs = np.nan_to_num(corrs, nan=0.0)
    
    return corrs

def compute_inverse_covariance(data, interv, **kwargs):
    # compute inverse covariance
    # invcovs = np.linalg.pinv(np.cov(data.T, bias=True), rcond=1e-10)
    lw = LedoitWolf()
    lw.fit(data)
    invcovs = lw.get_precision()
    invcovs = np.abs(invcovs)
    # print(invcovs)
    return invcovs

class Avici_predict():
    def __init__(self, version="scm-v0", no_interv=False):
        self.version = version
        self.model = avici.load_pretrained(checkpoint_dir="/mnt/shared-storage-user/pengbo/created/projects/CDLLM/Test-1213/avici_models/scm-v0",  # 指向本地下载的目录
    expects_counts=False
            )
        self.no_interv = no_interv

    def __call__(self, data, interv, **kwargs):
        if self.no_interv:
            return self.model(x=data, interv=None)
        else:
            return self.model(x=data, interv=interv)

bnlearn_node_edge_dict = {
    "alarm": (37, 46),
    "asia": (8, 8),
    "child": (20, 25),
    "cancer": (5, 4),
    "earthquake": (5, 4),
    "sachs": (11, 17),
    "insurance": (27, 52)
}

def process_results(method, feature_baselines, method_threshold, key_extracted=False, 
        specific_data="Nan", save=False, save_path=None, save_graphs=True, output_dir="predict"):
    all_naive_baselines = {}
    baseline_dicts = defaultdict(lambda: defaultdict(list))
    
    # 创建输出目录
    if save_graphs:
        os.makedirs(output_dir, exist_ok=True)
        graphs_dir = os.path.join(output_dir, "graphs")
        results_dir = os.path.join(output_dir, "results")
        visualizations_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
    
    # 存储每个样本的详细结果
    detailed_results = []
    
    # print(len(feature_baselines))
    for idx, (fp_key, item) in enumerate(feature_baselines.items()):
        # Use original_fp if provided, otherwise fallback to fp_key
        fp = item.get("original_fp", fp_key)
        n, e, key = from_path_to_info(fp, key_extracted, specific_data)
        true = item["true"]
        pred = item["pred"]
        true_2d = item.get("true_2d", None)
        pred_2d = item.get("pred_2d", None)
        
        assert true is not None, fp
        assert pred is not None, fp
        assert not np.isnan(true).any(), f"{fp}: true contains NaN"
        assert not np.isnan(pred).any(), f"{fp}: pred contains NaN, {true},{pred}"
        
        # 计算指标
        auc = roc_auc_score(true, pred)
        prc = average_precision_score(true, pred)
        
        if method_threshold == "auto":
            pos_rate = e / n**2
            threshold = np.quantile(pred, 1 - pos_rate)
            shd_, edge_mcc, _ = compute_additional_metrics(true, pred, threshold=threshold)
        else:
            threshold = method_threshold
            shd_, edge_mcc, _ = compute_additional_metrics(true, pred, threshold=threshold)
        
        # 使用 metrics.py 中的函数计算 precision, recall, F1 和 SHD（与 metrics.py 一致）
        if true_2d is not None and pred_2d is not None:
            # 二值化预测矩阵
            pred_2d_bin = (pred_2d > threshold).astype(int)
            # 使用标准方法重新计算 SHD（与 metrics.py 一致）
            shd = shd_metric_standard(pred_2d_bin, true_2d)
            
            # 计算 precision, recall, F1
            metrics_result = compute_precision_recall_f1(pred_2d, true_2d, threshold=threshold)
            precision = metrics_result['precision']
            recall = metrics_result['recall']
            f1 = metrics_result['f1_score']
        else:
            # 如果没有2D矩阵，设置为None或0
            precision = 0
            recall = 0
            f1 = 0
        
        baseline_dicts[key]["auc"].append(auc)
        baseline_dicts[key]["prc"].append(prc)
        baseline_dicts[key]["shd"].append(shd)
        baseline_dicts[key]["edge_mcc"].append(edge_mcc)
        baseline_dicts[key]["precision"].append(precision)
        baseline_dicts[key]["recall"].append(recall)
        baseline_dicts[key]["f1"].append(f1)
        baseline_dicts[key]["time"].append(item["time"])
        baseline_dicts[key]["cycle"].append(item["cycle"])
        
        # 保存图和详细结果
        if save_graphs:
            # 使用 key 作为标识（包含数据集信息）
            # key 格式可能是 "sachs" 或 "child-20-25" 等
            safe_key = key.replace('/', '_').replace(' ', '_')  # 替换不安全字符
            
            # Add run suffix to distinguish between multiple runs of the same dataset
            run_suffix = ""
            if "#run" in fp_key:
                run_suffix = "_" + fp_key.split("#")[-1]
            
            # 在文件名中包含 key 和方法名称
            sample_id = f"{safe_key}{run_suffix}_{method}"
            
            # 保存真实图（2D矩阵）
            if true_2d is not None:
                true_graph_path = os.path.join(graphs_dir, f"{sample_id}_true.csv")
                pd.DataFrame(true_2d).to_csv(true_graph_path, index=False, header=False)
            
            # 保存预测图（2D矩阵）
            if pred_2d is not None:
                pred_graph_path = os.path.join(graphs_dir, f"{sample_id}_pred.csv")
                pd.DataFrame(pred_2d).to_csv(pred_graph_path, index=False, header=False)
                
                # 保存二值化的预测图
                pred_binary = (pred_2d > threshold).astype(int)
                pred_binary_path = os.path.join(graphs_dir, f"{sample_id}_predbinary.csv")
                pd.DataFrame(pred_binary).to_csv(pred_binary_path, index=False, header=False)
            
            # 生成可视化图片
            if true_2d is not None and pred_2d is not None:
                viz_path = os.path.join(visualizations_dir, f"{sample_id}_comparison.png")
                try:
                    visualize_graph_comparison(true_2d, pred_2d, sample_id, viz_path, threshold=threshold)
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to visualize {sample_id}: {e}")
            
            # 保存该样本的详细结果
            detailed_results.append({
                'sample_id': sample_id,
                'file_path': fp,
                'nodes': n,
                'edges': e,
                'key': key,
                'auc': auc,
                'prc': prc,
                'shd': shd,
                'edge_mcc': edge_mcc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'time': item["time"],
                'cycle': item["cycle"],
                'threshold': threshold
            })

    for k,v in baseline_dicts.items():
        for metric, vals in v.items():
            baseline_dicts[k][metric] = np.mean(vals), np.std(vals), len(vals)

    all_naive_baselines[method] = baseline_dicts
    
    dfs_baseline = {}
    all_vals = defaultdict(list)
    for metric in ["auc", "prc", "shd", "edge_mcc", "precision", "recall", "f1", "time", "cycle"]:
        # ours_baselines
        for name, results in all_naive_baselines.items():
            for k, v in sorted(results.items()):
                nodes, edges, mechanism = from_key_to_info(k)
                all_vals["nodes"].append(nodes)
                all_vals["edges"].append(edges)
                all_vals["mechanism"].append(mechanism)
                all_vals["metric"].append(metric)
                all_vals["mean"].append(v[metric][0])
                all_vals["std"].append(v[metric][1])
                all_vals["dag_num"].append(v[metric][2])
                all_vals["model"].append(name)        

    dfs = pd.DataFrame.from_dict(all_vals)
    print(dfs.to_string())
    result = dfs[dfs['metric'] == 'time'].groupby(['nodes', 'edges'])['mean'].mean()
    print(result)
    
    # 保存汇总结果
    if save:
        assert save_path is not None
        dfs.to_csv(save_path)
    
    # 保存详细结果
    if save_graphs and detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_results_path = os.path.join(results_dir, f"{method}_detailed_results.csv")
        detailed_df.to_csv(detailed_results_path, index=False)
        print(f"\n{'='*70}")
        print(f"✅ Saved detailed results to: {detailed_results_path}")
        print(f"✅ Saved {len(detailed_results)} graph pairs (CSV) to: {graphs_dir}")
        print(f"✅ Saved {len(detailed_results)} visualizations (PNG) to: {visualizations_dir}")
        print(f"{'='*70}")
    
    return dfs, detailed_results

def DrBO_predict(data, interv, label,**kwargs):
    print(data.shape)
    print(label.shape)
    # exit()
    pred = DrBO(
        X=data,
        GT=label,
        score_method='BIC',
        normalize=True,
        score_params={
            'noise_var': 'nv',
            'reg': 'gp',
            'med_bw': True,
        },
        max_evals=4000, # the more the better
        pruner=prune_linear,
        device='cuda',
        verbose=True,
        )
    try:
        print(MetricsDAG._count_accuracy(prune_cit(data, pred['raw']), label))
    except:
        print("Error")

    return pred['raw']

def fci_predict(data, interv, label=None, is_discrete=False, alpha=0.05, **kwargs):
    """
    使用FCI算法进行因果发现
    
    Parameters:
    -----------
    data: ndarray, shape (n_samples, n_variables)
        观测数据
    interv: ndarray, shape (n_samples, n_variables)
        干预信息（暂未使用，FCI通常用于观测数据）
    label: ndarray, optional
        真实的因果图（用于调试和参考）
    is_discrete: bool, default=False
        数据是否为离散型。True表示离散，False表示连续
    alpha: float, default=0.05
        独立性检验的显著性水平
    
    Returns:
    --------
    adj_matrix: ndarray, shape (n_variables, n_variables)
        预测的邻接矩阵
    """
    if not FCI_AVAILABLE:
        raise ImportError("causal-learn is not installed. Please install it using: pip install causal-learn")
    
    n_samples, n_vars = data.shape
    
    # 选择条件独立性检验方法
    if is_discrete:
        # 离散数据：使用卡方检验 (Chi-square test)
        independence_test_method = chisq
        print(f"Using FCI with Chi-square test (discrete data), alpha={alpha}")
    else:
        # 连续数据：使用Fisher's Z检验
        independence_test_method = fisherz
        print(f"Using FCI with Fisher's Z test (continuous data), alpha={alpha}")
    
    try:
        # 运行FCI算法
        # fci返回一个包含图结构的对象
        G, edges = fci(
            data, 
            independence_test_method=independence_test_method,
            alpha=alpha,
            verbose=False,
            show_progress=False
        )
        
        # 提取邻接矩阵
        # FCI返回的是PAG (Partial Ancestral Graph)，需要转换为邻接矩阵
        # G.graph[i,j] 的值表示边的类型:
        # -1: no edge
        #  0: circle
        #  1: arrowhead (->)
        #  2: tail (-)
        
        # 创建邻接矩阵：如果存在从i到j的有向边，则adj[i,j]=1
        adj_matrix = np.zeros((n_vars, n_vars))
        
        graph = G.graph
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # 检查是否存在从i到j的边
                    # graph[j,i]=1 表示j端是箭头，graph[i,j]=2 表示i端是尾部
                    # 即 i -> j
                    if graph[j, i] == 1 and graph[i, j] == 2:
                        adj_matrix[i, j] = 1.0
                    # 如果是无向边或circle，给一个较低的权重
                    elif graph[j, i] != -1 and graph[i, j] != -1:
                        adj_matrix[i, j] = 0.5
        
        # 归一化到[0,1]
        if adj_matrix.max() > 0:
            adj_matrix = adj_matrix / adj_matrix.max()
        
        return adj_matrix
        
    except Exception as e:
        print(f"Error in FCI: {e}")
        # 返回零矩阵作为后备
        return np.zeros((n_vars, n_vars))

def detect_discrete_data(data, threshold=10):
    """
    检测数据是否为离散型
    
    Parameters:
    -----------
    data: ndarray
        输入数据
    threshold: int
        如果每列唯一值的平均数量小于此阈值，则认为是离散数据
    
    Returns:
    --------
    is_discrete: bool
    """
    n_unique = []
    for col in range(data.shape[1]):
        n_unique.append(len(np.unique(data[:, col])))
    avg_unique = np.mean(n_unique)
    
    return avg_unique < threshold

if __name__ == "__main__":
    from tqdm import tqdm
    from time import time
    import pickle
    

    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/sergio_8000.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/syntren.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/sachs.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/bnlearn_enco.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/bnlearn_mbrl.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/all.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/both/intervention_8160.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/synthetic_8160_subset.csv"
    # fp_data = "/cpfs04/shared/CausalAI/pengbo/Causal_Discovery/data/csv/test/largenode.csv"
    # fp_data = "/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/csv/test/real_and_synours.csv"
    # fp_data = "/cpfs04/user/pengbo/projects/causalDiscovery/24-4-16/syn_uniform.csv"
    # fp_data = "/cpfs04/user/pengbo/projects/causalDiscovery/24-4-16/syn_largenode.csv"
    # fp_data = "/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/csv/test/obs-intv=1-n/dag-num5-comb4-degree24-node150200/linear_nn_sigmoid_polynomial.csv"
    # fp_data = "/mnt/shared-storage-user/pengbo/created/projects/CDLLM/Test-1213/real.csv"
    fp_data = "/mnt/shared-storage-user/pengbo/created/projects/CDLLM/Test-1213/real_test2.csv"
    # fp_data = "/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/csv/test/real_and_syn_and_sergio.csv"



    method = "avici"  # Options: "corr", "invcov", "avici", "notears", "drbo", "sdcd", "fci"
    if method == "avici":
        import avici
        avici_predict_func = Avici_predict(version="scm-v0", no_interv=False) # scm-v0, neurips-grn
    elif method == "notears":
        sys.path.append('/mnt/shared-storage-user/pengbo/created/projects/CausalDiscovery/notears')
        from notears.linear import notears_linear
    elif method == "fci":
        # FCI需要causal-learn库
        # 安装命令: pip install causal-learn
        if not FCI_AVAILABLE:
            raise ImportError("Please install causal-learn: pip install causal-learn")
        print("FCI method selected. Will auto-detect discrete/continuous data.")
    elif method == "drbo":
        sys.path.append('/cpfs04/user/pengbo/projects/causalDiscovery/24-4-16/test_method/DrBO')
        import warnings
        os.environ['OMP_NUM_THREADS'] = '1'
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        import logging
        logging.basicConfig(force=True)
        from castle.metrics import MetricsDAG
        from drbo.pruners import prune_linear, prune_cit
        from drbo import DrBO
        from drbo.utils import sim_data, read_nonlinear_gp, read_sachs, viz_history
    elif method == "sdcd": # use sea env
        from sdcd.models import SDCD
        from sdcd.utils import create_intervention_dataset
        from sdcd.simulated_data import random_model_gaussian_global_variance # For demonstration

    method_dict = {
        "corr": {"function": compute_correlation, "threshold": "auto", "sample_size": "all", "args": {}},
        "invcov": {"function": compute_inverse_covariance, "threshold": "auto", "sample_size": "all", "args": {}},
        "avici": {"function": avici_predict_func if method == "avici" else None, "threshold": 0.5, "sample_size": 1000, "args": {}},
        "notears": {"function": notears_linear if method == "notears" else None, "threshold": 0.3, "sample_size": 1000, "args": {"lambda1": 0.1, "loss_type": "l2"}},
        "drbo": {"function": DrBO_predict if method == "drbo" else None, "threshold": 0., "sample_size": 500, "args": {}},
        "sdcd": {"function": sdcd_predict if method == "sdcd" else None, "threshold": 0.5, "sample_size": 1000, "args": {}},
        "fci": {"function": fci_predict if method == "fci" else None, "threshold": 0.5, "sample_size": "all", "args": {"alpha": 0.05}}
    }
    method_function = method_dict[method]["function"]
    method_threshold = method_dict[method]["threshold"]
    method_sample_size = method_dict[method]["sample_size"]
    method_args = method_dict[method]["args"]
    items = read_csv(fp_data)

    num_runs = 5  # Number of times to run each experiment
    feature_baselines = {}
    for item in tqdm(items):
        if item["split"] != "test":
            continue
        # if "p150_e600" not in item["fp_graph"]:
        #     continue
        # if "p50" in item["fp_graph"]:
        #     continue
        # if (any(settings in item["fp_graph"] for settings in ["p10_e10", "p20_e80", "p100_e400"]) and \
        #     any(settings in item["fp_graph"] for settings in ["linear_struct", "_nn_struc", "polynomial_struc", "sigmoid_add_struc"])) or \
        #         ("sergio_hill=2.0" in item["fp_graph"]) or (("synthetic" not in fp_data) and ("sergio" not in fp_data) and ("intervention_8160" not in fp_data)) :
        if True:
            # load data
            data_full = np.load(item["fp_data"])
            label_full = np.load(item["fp_graph"])
            if item["fp_regime"].endswith('.npy'):
                interv_full = np.load(item["fp_regime"]).astype(np.float32)
                regimes = []
                for i in range(len(interv_full)):
                    # 找出值为 1 的列索引
                    nodes = tuple(sorted(np.where(interv_full[i] == 1)[0].tolist()))
                    regimes.append(nodes)
            else:
                if item["fp_regime"] != "None":
                    with open(item["fp_regime"]) as f:
                        # if >1 node intervened, formatted as a list
                        lines = [line.strip() for line in f.readlines()]
                else:
                    lines = [""]*len(data_full)
                regimes = [tuple(sorted(int(x) for x in line.split(",")))
                        if len(line) > 0 else () for line in lines]
            assert len(regimes) == len(data_full)

            # process_intervention
            interv_matrix = np.zeros((len(data_full), len(label_full)))
            for i, regime in enumerate(regimes):
                for node in regime:
                    interv_matrix[i, node] = 1

            for run_idx in range(num_runs):
                # Sample data for this run
                if method_sample_size != "all":
                    sample_idx = np.random.choice(len(data_full), method_sample_size, replace=True)
                    data = data_full[sample_idx]
                    interv = interv_matrix[sample_idx]
                else:
                    data = data_full
                    interv = interv_matrix
                
                label = label_full.copy()

                # exclude diagonal from evaluation
                diag_mask = ~np.eye(data.shape[1], dtype=bool)
                
                # method
                time_start = time()
                
                # 对于FCI方法，需要判断数据是离散还是连续
                if method == "fci":
                    # 判断数据集类型：sachs是连续的，其他都是离散的
                    is_discrete = True
                    if "sachs" in item["fp_graph"].lower() or "sachs" in item["fp_data"].lower():
                        is_discrete = False
                        # Only print on first run to avoid clutter
                        if run_idx == 0: print(f"Detected Sachs dataset - using continuous FCI")
                    else:
                        # 也可以通过自动检测
                        is_discrete_auto = detect_discrete_data(data)
                        if not is_discrete_auto:
                            is_discrete = False
                            if run_idx == 0: print(f"Auto-detected continuous data")
                        else:
                            if run_idx == 0: print(f"Using discrete FCI for this dataset")
                    
                    method_args_with_discrete = {**method_args, "is_discrete": is_discrete}
                    pred = method_function(data, interv=interv, label=label, **method_args_with_discrete)
                else:
                    pred = method_function(data, interv=interv, label=label, **method_args)
                
                # 保存完整的2D矩阵（用于后续保存CSV）
                pred_2d = pred.copy()
                label_2d = label.copy()
                
                pred_no_diag = pred.copy()
                np.fill_diagonal(pred_no_diag, 0)
                # print(pred_no_diag)
                # exit()
                pred_no_diag = (pred_no_diag > 0.5).astype(int)
                has_cycles, _ = check_cycles_method1(pred_no_diag)
                # print(pred)
                # exit()
                time_end = time()
                pred_flat = pred[diag_mask]
                label_flat = label[diag_mask]
                
                # save with run index
                run_key = f"{item['fp_data']}#run{run_idx}"
                feature_baselines[run_key] = {
                    "true": label_flat,
                    "pred": pred_flat,
                    "true_2d": label_2d,  # 保存2D完整矩阵
                    "pred_2d": pred_2d,   # 保存2D完整矩阵
                    "time": time_end - time_start,
                    "cycle": has_cycles,
                    "original_fp": item["fp_data"]
                }
    
    # 处理并保存结果
    summary_df, detailed_results = process_results(
        method=method, 
        feature_baselines=feature_baselines, 
        method_threshold=method_threshold,
        save_graphs=True,
        output_dir="predict"
    )
    
    print(f"\n{'='*70}")
    print(f"All results saved to 'predict' directory:")
    print(f"  - CSV graphs: predict/graphs/")
    print(f"  - Visualizations: predict/visualizations/")
    print(f"  - Detailed results: predict/results/{method}_detailed_results.csv")
    print(f"{'='*70}\n")
