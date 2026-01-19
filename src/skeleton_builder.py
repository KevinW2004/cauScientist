"""
Skeleton Builder - MMHC算法的骨架构建部分
用于在LLM生成因果图之前，通过统计方法缩小搜索空间
"""

import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple
from itertools import combinations
from scipy.stats import pearsonr, chi2_contingency
from scipy import stats

def _skeleton_to_graph_format(skeleton: np.ndarray, variable_names: List[str]) -> Dict:
    """
    将骨架矩阵转换为图结构格式（用于复用_compute_metrics）
    
    Args:
        skeleton: 无向骨架矩阵 [n_vars, n_vars]
        variable_names: 变量名列表
    
    Returns:
        图结构字典 {'nodes': [...]}
    """
    n_vars = skeleton.shape[0]
    nodes = []
    for i in range(n_vars):
        parents = []
        for j in range(n_vars):
            if skeleton[i, j] == 1:
                parents.append(variable_names[j])
        
        nodes.append({
            'name': variable_names[i],
            'parents': parents
        })
    
    return {'nodes': nodes}
    
class SkeletonBuilder:
    """使用MMHC算法构建无向骨架图"""
    
    def __init__(self, alpha: float = 0.05, max_cond_size: int = 3, variable_type: str = "continuous"):
        """
        Args:
            alpha: 独立性检验的显著性水平
            max_cond_size: 条件集的最大大小（控制计算复杂度）
            variable_type: 变量类型 ("continuous" 或 "discrete")
        """
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.variable_type = variable_type
    
    def partial_correlation(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None) -> Tuple[float, float]:
        """
        计算偏相关系数和p值
        
        Args:
            X: 变量X的数据 [n_samples]
            Y: 变量Y的数据 [n_samples]
            Z: 条件变量集的数据 [n_samples, n_cond_vars] 或 None
            
        Returns:
            (correlation, p_value)
        """
        if Z is None or Z.shape[1] == 0:
            # 无条件：直接计算Pearson相关
            corr, pval = pearsonr(X, Y)
            return abs(corr), pval
        
        # 有条件：计算偏相关
        n = len(X)
        
        # 对X, Y, Z进行标准化
        X_std = (X - X.mean()) / (X.std() + 1e-10)
        Y_std = (Y - Y.mean()) / (Y.std() + 1e-10)
        Z_std = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-10)
        
        # 回归残差法计算偏相关
        # X对Z回归
        beta_xz = np.linalg.lstsq(Z_std, X_std, rcond=None)[0]
        residual_x = X_std - Z_std @ beta_xz
        
        # Y对Z回归
        beta_yz = np.linalg.lstsq(Z_std, Y_std, rcond=None)[0]
        residual_y = Y_std - Z_std @ beta_yz
        
        # 残差的相关系数
        corr = np.corrcoef(residual_x, residual_y)[0, 1]
        
        # Fisher's z变换计算p值
        if abs(corr) >= 0.9999:
            return abs(corr), 0.0
        
        z_score = 0.5 * np.log((1 + corr) / (1 - corr)) * np.sqrt(n - Z.shape[1] - 3)
        pval = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return abs(corr), pval
    
    def compute_association(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray = None
    ) -> Tuple[float, float]:
        """
        计算变量间的关联度和p值
        根据变量类型选择合适的方法
        
        Args:
            X: 变量X的数据 [n_samples]
            Y: 变量Y的数据 [n_samples]
            Z: 条件变量集的数据 [n_samples, n_cond_vars] 或 None
            
        Returns:
            (association, p_value)
            - association: 关联度 (越大越相关)
            - p_value: 显著性p值
        """
        if self.variable_type == "discrete":
            # 对于离散变量，使用卡方检验
            # 返回 1-pval 作为关联度（p值越小，关联度越大）
            X_int = X.astype(int)
            Y_int = Y.astype(int)
            
            if Z is None or (hasattr(Z, 'shape') and Z.shape[1] == 0):
                # 无条件
                try:
                    contingency_table = pd.crosstab(X_int, Y_int)
                    if contingency_table.size <= 1:
                        return 0.0, 1.0
                    chi2, pval, dof, expected = chi2_contingency(contingency_table)
                    # 使用 Cramér's V 作为关联度
                    n = len(X)
                    min_dim = min(contingency_table.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    return cramers_v, pval
                except:
                    return 0.0, 1.0
            else:
                # 有条件：使用条件独立性检验
                # 这里简化为返回1-pval作为关联度
                cond_indices = list(range(Z.shape[1])) if Z.ndim > 1 else [0]
                is_indep, pval = self.discrete_independence_test(
                    0, 1, set(), 
                    np.column_stack([X_int, Y_int] + [Z[:, i] for i in cond_indices])
                )
                association = 1 - pval
                return association, pval
        else:
            # 连续变量：使用偏相关
            return self.partial_correlation(X, Y, Z)
    
    def discrete_independence_test(
        self,
        X_idx: int,
        Y_idx: int,
        cond_set: Set[int],
        data: np.ndarray
    ) -> Tuple[bool, float]:
        """
        离散变量的条件独立性检验
        使用卡方检验(无条件)或分层卡方检验(有条件)
        
        Args:
            X_idx: 变量X的索引
            Y_idx: 变量Y的索引
            cond_set: 条件变量集的索引集合
            data: 数据 [n_samples, n_variables]
            
        Returns:
            (is_independent, p_value)
        """
        X = data[:, X_idx].astype(int)
        Y = data[:, Y_idx].astype(int)
        
        if len(cond_set) == 0:
            # 无条件：直接卡方检验
            try:
                contingency_table = pd.crosstab(X, Y)
                if contingency_table.size <= 1:
                    return True, 1.0
                chi2, pval, dof, expected = chi2_contingency(contingency_table)
                return pval > self.alpha, pval
            except:
                return True, 1.0
        else:
            # 有条件：分层卡方检验（Cochran-Mantel-Haenszel style）
            Z = data[:, list(cond_set)].astype(int)
            
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            
            # 按条件变量分层
            df = pd.DataFrame({
                'X': X,
                'Y': Y,
                **{f'Z{i}': Z[:, i] for i in range(Z.shape[1])}
            })
            
            z_cols = [f'Z{i}' for i in range(Z.shape[1])]
            grouped = df.groupby(z_cols)
            
            chi2_sum = 0
            dof_sum = 0
            valid_strata = 0
            
            for name, group in grouped:
                if len(group) < 5:  # 样本太少，跳过这层
                    continue
                
                try:
                    contingency_table = pd.crosstab(group['X'], group['Y'])
                    if contingency_table.size <= 1:
                        continue
                    
                    chi2, _, dof, _ = chi2_contingency(contingency_table)
                    chi2_sum += chi2
                    dof_sum += dof
                    valid_strata += 1
                except:
                    continue
            
            if valid_strata == 0:
                # 无法检验，保守地认为独立
                return True, 1.0
            
            # 合并统计量
            pval = 1 - stats.chi2.cdf(chi2_sum, dof_sum)
            return pval > self.alpha, pval
    
    def conditional_independence_test(
        self, 
        X_idx: int, 
        Y_idx: int, 
        cond_set: Set[int], 
        data: np.ndarray
    ) -> Tuple[bool, float]:
        """
        条件独立性检验: X ⊥ Y | cond_set
        根据变量类型选择合适的检验方法
        
        Args:
            X_idx: 变量X的索引
            Y_idx: 变量Y的索引
            cond_set: 条件变量集的索引集合
            data: 数据 [n_samples, n_variables]
            
        Returns:
            (is_independent, p_value)
        """
        if self.variable_type == "discrete":
            return self.discrete_independence_test(X_idx, Y_idx, cond_set, data)
        else:
            # 连续变量：使用偏相关
            X = data[:, X_idx]
            Y = data[:, Y_idx]
            
            if len(cond_set) == 0:
                Z = None
            else:
                Z = data[:, list(cond_set)]
            
            _, pval = self.partial_correlation(X, Y, Z)
            
            return pval > self.alpha, pval
    
    def MMPC_forward(
        self, 
        X_idx: int, 
        data: np.ndarray, 
        variable_names: List[str] = None
    ) -> List[int]:
        """
        MMPC前向阶段：找到候选PC集
        
        Args:
            X_idx: 目标变量索引
            data: 数据 [n_samples, n_variables]
            variable_names: 变量名列表（用于调试）
            
        Returns:
            候选PC集的索引列表
        """
        n_vars = data.shape[1]
        CPC = []  # Candidate Parents and Children
        candidates = set(range(n_vars)) - {X_idx}
        
        while candidates:
            # 1. 对每个候选变量，找到最小关联度
            max_min_assoc = -np.inf
            best_Y = None
            best_pval = 1.0
            
            for Y_idx in candidates:
                # 在所有CPC的子集上找最小关联
                min_assoc = np.inf
                min_pval = 1.0
                
                # 限制条件集大小，避免组合爆炸
                max_subset_size = min(len(CPC), self.max_cond_size)
                
                for subset_size in range(max_subset_size + 1):
                    for S in combinations(CPC, subset_size):
                        S_set = set(S)
                        assoc, pval = self.compute_association(
                            data[:, X_idx], 
                            data[:, Y_idx], 
                            data[:, list(S_set)] if S_set else None
                        )
                        
                        if assoc < min_assoc:
                            min_assoc = assoc
                            min_pval = pval
                
                # 2. 选择max-min关联度最大的变量
                if min_assoc > max_min_assoc:
                    max_min_assoc = min_assoc
                    best_Y = Y_idx
                    best_pval = min_pval
            
            # 3. 显著性检验
            if best_pval < self.alpha:
                CPC.append(best_Y)
                candidates.remove(best_Y)
            else:
                break  # 没有更多显著相关的变量
        
        return CPC
    
    def MMPC_backward(
        self, 
        X_idx: int, 
        CPC: List[int], 
        data: np.ndarray
    ) -> List[int]:
        """
        MMPC后向阶段：剪枝假阳性
        
        Args:
            X_idx: 目标变量索引
            CPC: 候选PC集
            data: 数据
            
        Returns:
            剪枝后的PC集
        """
        PC = CPC.copy()
        
        for Y_idx in CPC:
            if Y_idx not in PC:
                continue
            
            # 在剩余变量中寻找分离集
            remaining = set(PC) - {Y_idx}
            found_separator = False
            
            max_subset_size = min(len(remaining), self.max_cond_size)
            
            for subset_size in range(max_subset_size + 1):
                if found_separator:
                    break
                    
                for S in combinations(remaining, subset_size):
                    S_set = set(S)
                    is_indep, pval = self.conditional_independence_test(
                        X_idx, Y_idx, S_set, data
                    )
                    
                    if is_indep:
                        PC.remove(Y_idx)
                        found_separator = True
                        break
        
        return PC
    
    def build_skeleton(
        self, 
        data: np.ndarray, 
        variable_names: List[str] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        构建无向骨架图
        
        Args:
            data: 数据 [n_samples, n_variables]
            variable_names: 变量名列表
            verbose: 是否打印详细信息
            
        Returns:
            (skeleton_matrix, PC_sets)
            - skeleton_matrix: 无向邻接矩阵 [n_vars, n_vars]
            - PC_sets: 每个变量的PC集 {var_idx: [pc_indices]}
        """
        n_vars = data.shape[1]
        
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(n_vars)]
        
        if verbose:
            print("\n" + "="*70)
            print("BUILDING SKELETON USING MMHC")
            print("="*70)
            print(f"Variables: {n_vars}")
            print(f"Samples: {data.shape[0]}")
            print(f"Variable type: {self.variable_type}")
            print(f"Alpha: {self.alpha}")
            print(f"Max condition set size: {self.max_cond_size}")
            print("="*70 + "\n")
        
        # 为每个变量找PC集
        PC_sets = {}
        
        for X_idx in range(n_vars):
            if verbose:
                print(f"Processing {variable_names[X_idx]} ({X_idx+1}/{n_vars})...")
            
            # Forward phase
            CPC = self.MMPC_forward(X_idx, data, variable_names)
            
            # Backward phase
            PC = self.MMPC_backward(X_idx, CPC, data)
            
            PC_sets[X_idx] = PC
            
            if verbose and PC:
                pc_names = [variable_names[i] for i in PC]
                print(f"  → PC set: {pc_names}")
        
        # 构建无向邻接矩阵（对称性检查）
        skeleton = np.zeros((n_vars, n_vars), dtype=int)
        
        for X_idx in range(n_vars):
            for Y_idx in PC_sets[X_idx]: # 去掉了对称性检查
                skeleton[X_idx, Y_idx] = 1
                skeleton[Y_idx, X_idx] = 1
        
        n_edges = skeleton.sum() // 2
        
        if verbose:
            print("\n" + "-"*70)
            print(f"Skeleton built: {n_edges} undirected edges")
            print("-"*70)
            self._print_skeleton(skeleton, variable_names)
        
        return skeleton, PC_sets
    
    def _print_skeleton(self, skeleton: np.ndarray, variable_names: List[str]):
        """打印骨架图"""
        n_vars = skeleton.shape[0]
        edges = []
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if skeleton[i, j] == 1:
                    edges.append(f"{variable_names[i]} — {variable_names[j]}")
        
        if edges:
            print("Edges:")
            for edge in edges:
                print(f"  {edge}")
        else:
            print("No edges found.")
    
    def skeleton_to_constraint(
        self, 
        skeleton: np.ndarray, 
        variable_names: List[str]
    ) -> Dict:
        """
        将骨架转换为LLM可以理解的约束格式
        
        Returns:
            {
                'allowed_edges': [(parent, child), ...],  # 允许的有向边
                'forbidden_pairs': [(var1, var2), ...]    # 禁止任何方向的边对
            }
        """
        n_vars = skeleton.shape[0]
        
        # 允许的边：骨架中的无向边可以定向为任意方向
        allowed_edges = []
        forbidden_pairs = []
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if skeleton[i, j] == 1:
                    # 骨架中有边：两个方向都允许
                    allowed_edges.append((variable_names[i], variable_names[j]))
                    allowed_edges.append((variable_names[j], variable_names[i]))
                else:
                    # 骨架中无边：禁止
                    forbidden_pairs.append((variable_names[i], variable_names[j]))
        
        return {
            'allowed_edges': allowed_edges,
            'forbidden_pairs': forbidden_pairs,
            'skeleton_matrix': skeleton.tolist()
        }


def build_skeleton_from_data(
    data: np.ndarray,
    variable_names: List[str],
    alpha: float = 0.05,
    max_cond_size: int = 3,
    variable_type: str = "continuous",
    verbose: bool = True
) -> Dict:
    """
    便捷函数：从数据构建骨架并返回约束
    
    Args:
        data: 数据 [n_samples, n_variables]
        variable_names: 变量名列表
        alpha: 显著性水平
        max_cond_size: 最大条件集大小
        variable_type: 变量类型 ("continuous" 或 "discrete")
        verbose: 是否打印详细信息
        
    Returns:
        约束字典，可直接传给LLM
    """
    builder = SkeletonBuilder(alpha=alpha, max_cond_size=max_cond_size, variable_type=variable_type)
    skeleton, pc_sets = builder.build_skeleton(data, variable_names, verbose)
    constraints = builder.skeleton_to_constraint(skeleton, variable_names)
    
    return constraints


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 生成测试数据: X → Y → Z, X → Z
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.randn(n_samples)
    Y = 0.8 * X + np.random.randn(n_samples) * 0.3
    Z = 0.6 * Y + 0.4 * X + np.random.randn(n_samples) * 0.2
    
    data = np.column_stack([X, Y, Z])
    variable_names = ['X', 'Y', 'Z']
    
    # 构建骨架
    constraints = build_skeleton_from_data(
        data=data,
        variable_names=variable_names,
        alpha=0.05,
        max_cond_size=2,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("CONSTRAINTS FOR LLM:")
    print("="*70)
    print(f"Allowed edges: {len(constraints['allowed_edges'])}")
    for edge in constraints['allowed_edges']:
        print(f"  {edge[0]} → {edge[1]}")
    
    print(f"\nForbidden pairs: {len(constraints['forbidden_pairs'])}")
    for pair in constraints['forbidden_pairs'][:5]:  # 只显示前5个
        print(f"  {pair[0]} ⊥ {pair[1]}")

