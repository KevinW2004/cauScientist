import numpy as np
import pandas as pd
import torch
import random
import os
from typing import Dict, List
from data_loader import DataLoader
from model_fitting import ModelFittingEngine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_adj_matrix(graph: Dict, variable_names: List[str]) -> np.ndarray:
    n = len(variable_names)
    v2i = {v: i for i, v in enumerate(variable_names)}
    adj = np.zeros((n, n))
    for node in graph['nodes']:
        child_idx = v2i[node['name']]
        for p in node.get('parents', []):
            adj[v2i[p], child_idx] = 1
    return adj

def adj_to_structured(adj: np.ndarray, variable_names: List[str]) -> Dict:
    nodes = []
    for i, name in enumerate(variable_names):
        parents = [variable_names[j] for j in range(len(variable_names)) if adj[j, i] == 1]
        nodes.append({'name': name, 'parents': parents})
    return {'nodes': nodes}

def has_cycle(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    visited = [0] * n
    def visit(u):
        visited[u] = 1 # gray
        for v in range(n):
            if adj[u, v] == 1:
                if visited[v] == 1 or (visited[v] == 0 and visit(v)):
                    return True
        visited[u] = 2 # black
        return False
    for i in range(n):
        if visited[i] == 0 and visit(i): return True
    return False

def compute_shd(adj1: np.ndarray, adj2: np.ndarray) -> int:
    """计算两个邻接矩阵之间的 SHD (Structural Hamming Distance)"""
    return int(np.sum(adj1 != adj2))

def perturb_graph(true_adj: np.ndarray, mode='remove'):
    n = true_adj.shape[0]
    adj = true_adj.copy()
    edges = np.argwhere(true_adj == 1)
    non_edges = np.argwhere((true_adj == 0) & (np.eye(n) == 0))
    
    if mode == 'remove' and len(edges) > 0:
        idx = random.choice(range(len(edges)))
        adj[edges[idx][0], edges[idx][1]] = 0
    elif mode == 'add' and len(non_edges) > 0:
        for _ in range(100): # try to add without cycle
            idx = random.choice(range(len(non_edges)))
            u, v = non_edges[idx]
            adj[u, v] = 1
            if not has_cycle(adj): break
            adj[u, v] = 0
    elif mode == 'reverse' and len(edges) > 0:
        for _ in range(100):
            idx = random.choice(range(len(edges)))
            u, v = edges[idx]
            adj[u, v], adj[v, u] = 0, 1
            if not has_cycle(adj): break
            adj[u, v], adj[v, u] = 1, 0
    elif mode == 'shuffle':
        # 保持结构（邻接矩阵不变），但随机打乱节点顺序
        perm = np.random.permutation(n)
        adj = adj[perm, :][:, perm]
    return adj

def load_dataset_by_name(dataset_name: str, csv_path: str = "real.csv"):
    """辅助函数：根据名称加载数据集"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Config file {csv_path} not found.")
        
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        if dataset_name.lower() in row['fp_data'].lower():
            print(f"  Found {dataset_name} at row {idx} in {csv_path}")
            return DataLoader.load_from_csv_config(csv_path, row_index=idx)
    
    raise ValueError(f"Dataset {dataset_name} not found in {csv_path}")

def run_fast_verify(dataset_name='asia', csv_path='real.csv'):
    print(f"\n>>> 正在验证数据集: {dataset_name}")
    
    try:
        dataset = load_dataset_by_name(dataset_name, csv_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 优先使用 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = ModelFittingEngine(device=device)
    
    # 获取变量类型
    variable_type = dataset.variable_type
    print(f"  变量类型: {variable_type} (Device: {device})")

    common_params = {
        "data": dataset.data,
        "interventions": dataset.interventions,
        "variable_type": variable_type,
        "num_epochs": 50, # 验证性质不需要100轮
        "verbose": False,
        "seed": 42
    }

    results = []
    
    # 1. 真实图
    print("评估真实图 (Ground Truth)...")
    gt_graph = adj_to_structured(dataset.ground_truth_graph, dataset.variable_names)
    res_gt = engine.fit(structured_graph=gt_graph, **common_params)
    results.append({'type': 'GROUND TRUTH', 'bic': res_gt['bic'], 'll': res_gt['log_likelihood']})

    # 2. 扰动图
    modes = ['remove', 'add', 'reverse', 'shuffle']
    labels = ['删边 (-1 edge)', '加边 (+1 edge)', '反转 (reverse 1)', '打乱 (Shuffled GT)']
    
    for mode, label in zip(modes, labels):
        print(f"评估扰动图: {label}...")
        p_adj = perturb_graph(dataset.ground_truth_graph, mode=mode)
        p_graph = adj_to_structured(p_adj, dataset.variable_names)
        res_p = engine.fit(structured_graph=p_graph, **common_params)
        results.append({'type': label, 'bic': res_p['bic'], 'll': res_p['log_likelihood']})

    # 3. 空图 (Baseline)
    print("评估空图 (Empty Graph)...")
    empty_graph = adj_to_structured(np.zeros_like(dataset.ground_truth_graph), dataset.variable_names)
    res_empty = engine.fit(structured_graph=empty_graph, **common_params)
    results.append({'type': 'EMPTY GRAPH', 'bic': res_empty['bic'], 'll': res_empty['log_likelihood']})

    # 结果展示
    df = pd.DataFrame(results)
    df['diff_to_gt'] = df['bic'] - res_gt['bic']
    print("\n" + "="*60)
    print(f"Score Function 有效性检查结果 ({dataset_name})")
    print("BIC 越小越好, diff_to_gt 应该为正数")
    print("="*60)
    print(df[['type', 'bic', 'diff_to_gt']].to_string(index=False))
    
    # 核心判断
    is_valid = all(df.loc[df['type'] != 'GROUND TRUTH', 'diff_to_gt'] > 0)
    
    if is_valid:
        print("\n✅ 验证通过：真实图在局部扰动中得分最优！")
    else:
        print("\n❌ 验证失败：存在得分比真实图更好的结构。")
        print("建议检查: 1. MLP是否收敛; 2. BIC惩罚项(k_eff)计算逻辑; 3. 数据样本量是否充足。")
    
    return df

def run_robust_verify(dataset_name='asia', csv_path='real.csv', n_repeats=50):
    print(f"\n>>> 正在启动鲁棒性验证实验: {dataset_name} (重复次数={n_repeats})")
    
    try:
        dataset = load_dataset_by_name(dataset_name, csv_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 优先使用 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = ModelFittingEngine(device=device)
    
    variable_type = dataset.variable_type
    print(f"  变量类型: {variable_type} (Device: {device})")

    # 1. 基准计算 (Ground Truth) - 训练 100 epoch
    print(f"\n[1] 计算基准: Ground Truth (100 epochs)...")
    gt_graph = adj_to_structured(dataset.ground_truth_graph, dataset.variable_names)
    res_gt = engine.fit(
        structured_graph=gt_graph, 
        data=dataset.data,
        interventions=dataset.interventions,
        variable_type=variable_type,
        num_epochs=100, 
        verbose=False,
        seed=42
    )
    base_gt_bic = res_gt['bic']
    print(f"    Base GT BIC: {base_gt_bic:.2f}")

    # 2. 多次扰动循环
    print(f"\n[2] 开始扰动测试 (每种模式 {n_repeats} 次)...")
    modes = ['remove', 'add', 'reverse', 'shuffle']
    all_summary = []

    for mode in modes:
        print(f"  测试模式: {mode}...")
        mode_deltas = []
        
        for i in range(n_repeats):
            # 重新调用 perturb_graph 生成不同的扰动图
            p_adj = perturb_graph(dataset.ground_truth_graph, mode=mode)
            p_graph = adj_to_structured(p_adj, dataset.variable_names)
            
            # 使用不同的随机种子跑 50 epoch
            res_p = engine.fit(
                structured_graph=p_graph, 
                data=dataset.data,
                interventions=dataset.interventions,
                variable_type=variable_type,
                num_epochs=100, 
                verbose=False,
                seed=42 + i
            )
            
            delta_bic = res_p['bic'] - base_gt_bic
            mode_deltas.append(delta_bic)
            print(f"    - Repeat {i+1}/{n_repeats}: Delta BIC = {delta_bic:.2f}")

        # 统计分析
        all_summary.append({
            'Perturbation Mode': mode,
            'Avg Delta BIC': np.mean(mode_deltas),
            'Std Dev': np.std(mode_deltas),
            'Min Delta': np.min(mode_deltas),
            'Max Delta': np.max(mode_deltas)
        })

    # 3. 统计与聚合
    df = pd.DataFrame(all_summary)
    
    print("\n" + "="*80)
    print(f"鲁棒性验证汇总结果 ({dataset_name})")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    
    # 4. 结论
    min_delta_all = df['Min Delta'].min()
    if min_delta_all > 0:
        print("\n✅ 验证通过：在所有重复实验中，真实图的 BIC 始终优于扰动图！")
    else:
        print(f"\n❌ 验证失败：在某些情况下扰动图的 BIC 反而更好 (Min Delta = {min_delta_all:.2f})")
    
    return df

def run_progressive_verify(dataset_name='asia', csv_path='real.csv', max_steps=20, n_repeats=5):
    print(f"\n>>> 正在启动渐进式扰动实验: {dataset_name} (max_steps={max_steps}, n_repeats={n_repeats})")
    
    try:
        dataset = load_dataset_by_name(dataset_name, csv_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 优先使用 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = ModelFittingEngine(device=device)
    
    variable_type = dataset.variable_type
    print(f"  变量类型: {variable_type} (Device: {device})")

    # 1. 基准计算 (Ground Truth)
    print(f"\n[1] 计算基准: Ground Truth (100 epochs)...")
    gt_adj = dataset.ground_truth_graph
    gt_graph = adj_to_structured(gt_adj, dataset.variable_names)
    res_gt = engine.fit(
        structured_graph=gt_graph, 
        data=dataset.data,
        interventions=dataset.interventions,
        variable_type=variable_type,
        num_epochs=100, 
        verbose=False,
        seed=42
    )
    base_gt_bic = res_gt['bic']
    
    data_points = []
    
    # 2. 渐进式扰动循环
    print(f"\n[2] 开始渐进式扰动轨迹...")
    for r in range(n_repeats):
        print(f"  轨迹 {r+1}/{n_repeats}:")
        current_adj = gt_adj.copy()
        
        for step in range(1, max_steps + 1):
            # 随机选择一种扰动模式 (不包括 shuffle)
            mode = random.choice(['remove', 'add', 'reverse'])
            current_adj = perturb_graph(current_adj, mode=mode)
            
            # 计算 SHD 和 BIC
            current_shd = compute_shd(gt_adj, current_adj)
            p_graph = adj_to_structured(current_adj, dataset.variable_names)
            
            # 使用不同的随机种子跑 50 epoch
            res_p = engine.fit(
                structured_graph=p_graph, 
                data=dataset.data,
                interventions=dataset.interventions,
                variable_type=variable_type,
                num_epochs=100, 
                verbose=False,
                seed=42 + r * max_steps + step
            )
            
            delta_bic = res_p['bic'] - base_gt_bic
            data_points.append({
                'Step': step,
                'SHD': current_shd,
                'Delta_BIC': delta_bic,
                'Trajectory': r
            })
            print(f"    Step {step}: SHD={current_shd}, Delta BIC={delta_bic:.2f}")

    # 3. 统计与可视化
    df = pd.DataFrame(data_points)
    correlation = df['SHD'].corr(df['Delta_BIC'])
    print(f"\nPearson 相关系数 (SHD vs Delta BIC): {correlation:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['SHD'], df['Delta_BIC'], alpha=0.5, label='Data Points')
    
    # 拟合回归线
    if len(df) > 1:
        # 移除重复的 SHD 点进行绘图计算可能更准，或者直接用所有点
        z = np.polyfit(df['SHD'], df['Delta_BIC'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['SHD'].min(), df['SHD'].max(), 100)
        plt.plot(x_range, p(x_range), "r--", label=f'Trend Line (slope={z[0]:.2f})')
    
    plt.xlabel('SHD (Structural Hamming Distance)')
    plt.ylabel('Delta BIC (Perturbed - GT)')
    plt.title(f'Progressive Perturbation: SHD vs BIC Correlation\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = 'progressive_check_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ 散点图已保存至: {output_file}")
    
    return df

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'alarm'
    
    # 你可以选择运行 robust 验证或者 progressive 验证
    # 这里默认运行两者，或者根据参数决定
    print("="*60)
    print("RUNNING SCORE FUNCTION VALIDATION")
    print("="*60)
    
    # run_robust_verify(target)
    run_progressive_verify(target)
