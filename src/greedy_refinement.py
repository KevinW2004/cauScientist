"""
贪心图优化模块
使用MLP的log-likelihood作为评估标准，逐步优化因果图
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
from copy import deepcopy
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """临时抑制stdout和stderr输出"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class GreedyGraphRefiner:
    """
    贪心图优化器：
    1. 从LLM的图开始
    2. 逐个测试添加/删除/反转边
    3. 用model_fitting的log-likelihood判断是否接受
    4. 保留改进最大的修改
    """
    
    def __init__(self, 
                 max_modifications: int = 10,
                 min_improvement: float = 0.01,
                 eval_epochs: int = 30,
                 allow_add: bool = True,
                 allow_delete: bool = True,
                 allow_reverse: bool = True,
                 max_candidates_per_type: int = 50):  # 新增：每种操作最多测试的候选数
        """
        Args:
            max_modifications: 最大修改次数
            min_improvement: 最小LL改进阈值（低于此值停止）
            eval_epochs: 评估时的训练轮数（较少以加速）
            allow_add: 是否允许添加边
            allow_delete: 是否允许删除边
            allow_reverse: 是否允许反转边
            max_candidates_per_type: 每种操作类型最多测试的候选数（加速）
        """
        self.max_modifications = max_modifications
        self.min_improvement = min_improvement
        self.eval_epochs = eval_epochs
        self.allow_add = allow_add
        self.allow_delete = allow_delete
        self.allow_reverse = allow_reverse
        self.max_candidates_per_type = max_candidates_per_type
    
    def refine_graph(self,
                     initial_graph: Dict,
                     data: np.ndarray,
                     variable_names: List[str],
                     model_fitting_engine,
                     interventions: np.ndarray = None,
                     variable_type: str = None,
                     skeleton_constraints: Optional[Dict] = None,
                     verbose: bool = True,
                     seed: int = None) -> Tuple[Dict, Dict]:
        """
        贪心优化图结构
        
        Args:
            initial_graph: LLM生成的初始图
            data: 观测数据
            variable_names: 变量名列表
            model_fitting_engine: 模型拟合引擎
            interventions: 干预矩阵 [n_samples, n_variables]
            variable_type: 变量类型 ("continuous" 或 "discrete")
            skeleton_constraints: 骨架约束（可选）
            verbose: 是否打印详细信息
            seed: 随机种子
            
        Returns:
            (优化后的图, 优化信息)
        """
        current_graph = deepcopy(initial_graph)
        current_ll = self._evaluate_graph(current_graph, data, interventions, variable_type, model_fitting_engine)
        
        if verbose:
            print(f"\n[Greedy Refinement] Starting LL: {current_ll:.4f}")
        
        modifications = []
        allowed_edges = self._get_allowed_edges(skeleton_constraints, variable_names) if skeleton_constraints else None
        
        for step in range(self.max_modifications):
            if verbose:
                print(f"\n[Step {step+1}/{self.max_modifications}] Testing modifications...")
            
            # 生成并评估所有候选修改
            candidates = []
            
            if self.allow_add:
                add_candidates = self._test_add_edges(
                    current_graph, data, variable_names, model_fitting_engine,
                    current_ll, allowed_edges, interventions, variable_type, verbose
                )
                candidates.extend(add_candidates)
            
            if self.allow_delete:
                delete_candidates = self._test_delete_edges(
                    current_graph, data, model_fitting_engine,
                    current_ll, interventions, variable_type, verbose
                )
                candidates.extend(delete_candidates)
            
            if self.allow_reverse:
                reverse_candidates = self._test_reverse_edges(
                    current_graph, data, variable_names, model_fitting_engine,
                    current_ll, allowed_edges, interventions, variable_type, verbose
                )
                candidates.extend(reverse_candidates)
            
            if not candidates:
                if verbose:
                    print(f"[Step {step+1}] No improving modifications found. Stopping.")
                break
            
            # 选择改进最大的修改
            best_op, best_edge, best_ll, best_graph = max(candidates, key=lambda x: x[2])
            improvement = best_ll - current_ll
            
            if improvement < self.min_improvement:
                if verbose:
                    print(f"[Step {step+1}] Best improvement {improvement:.4f} < threshold {self.min_improvement}. Stopping.")
                break
            
            # 应用最佳修改
            current_graph = best_graph
            current_ll = best_ll
            
            modifications.append({
                'step': step + 1,
                'operation': best_op,
                'edge': best_edge,
                'll_improvement': improvement,
                'new_ll': current_ll
            })
            
            if verbose:
                print(f"[Step {step+1}] ✓ {best_op} {best_edge[0]}→{best_edge[1]}, "
                      f"LL: {current_ll:.4f} (Δ={improvement:+.4f})")
        
        refinement_info = {
            'initial_ll': current_ll - sum(m['ll_improvement'] for m in modifications),
            'final_ll': current_ll,
            'total_improvement': sum(m['ll_improvement'] for m in modifications),
            'num_modifications': len(modifications),
            'modifications': modifications
        }
        
        if verbose:
            print(f"\n[Greedy Refinement] Completed!")
            print(f"  Total modifications: {len(modifications)}")
            print(f"  Final LL: {current_ll:.4f}")
            print(f"  Total improvement: {refinement_info['total_improvement']:+.4f}")
        
        return current_graph, refinement_info
    
    def _test_add_edges(self, current_graph: Dict, data: np.ndarray,
                        variable_names: List[str], engine,
                        current_ll: float, allowed_edges: Optional[Set],
                        interventions: np.ndarray,
                        variable_type: str,
                        verbose: bool) -> List[Tuple]:
        """测试添加边（随机采样加速）"""
        candidates = []
        current_edges = self._get_current_edges(current_graph, variable_names)
        
        # 生成所有候选边
        all_candidate_edges = []
        for i, parent in enumerate(variable_names):
            for j, child in enumerate(variable_names):
                if i == j:
                    continue
                
                edge = (parent, child)
                
                # 跳过已存在的边
                if edge in current_edges:
                    continue
                
                # 检查skeleton约束
                if allowed_edges and edge not in allowed_edges:
                    continue
                
                all_candidate_edges.append(edge)
        
        # 随机采样以加速（如果候选太多）
        if len(all_candidate_edges) > self.max_candidates_per_type:
            import random
            sampled_edges = random.sample(all_candidate_edges, self.max_candidates_per_type)
            if verbose:
                print(f"  [Sampling] {len(sampled_edges)}/{len(all_candidate_edges)} candidate edges to ADD")
        else:
            sampled_edges = all_candidate_edges
        
        # 评估采样的候选边
        tested_count = 0
        for parent, child in sampled_edges:
            # 测试添加这条边
            test_graph = self._add_edge_to_graph(current_graph, parent, child)
            
            # 检查是否产生环
            if self._has_cycle(test_graph):
                continue
            
            # 评估
            test_ll = self._evaluate_graph(test_graph, data, interventions, variable_type, engine)
            tested_count += 1
            
            if test_ll > current_ll:
                candidates.append(('ADD', (parent, child), test_ll, test_graph))
                if verbose and len(candidates) <= 3:
                    print(f"  ADD {parent}→{child}: LL={test_ll:.4f} (Δ={test_ll-current_ll:+.4f})")
        
        if verbose:
            print(f"  [ADD] Tested {tested_count} edges, found {len(candidates)} improvements")
        
        return candidates
    
    def _test_delete_edges(self, current_graph: Dict, data: np.ndarray,
                           engine, current_ll: float, interventions: np.ndarray,
                           variable_type: str, verbose: bool) -> List[Tuple]:
        """测试删除边"""
        candidates = []
        current_edges = self._get_current_edges_from_graph(current_graph)
        
        tested_count = 0
        for edge in current_edges:
            parent, child = edge
            
            # 测试删除这条边
            test_graph = self._remove_edge_from_graph(current_graph, parent, child)
            
            # 评估
            test_ll = self._evaluate_graph(test_graph, data, interventions, variable_type, engine)
            tested_count += 1
            
            if test_ll > current_ll:
                candidates.append(('DELETE', edge, test_ll, test_graph))
                if verbose and len(candidates) <= 3:
                    print(f"  DELETE {parent}→{child}: LL={test_ll:.4f} (Δ={test_ll-current_ll:+.4f})")
        
        if verbose:
            print(f"  [DELETE] Tested {tested_count} edges, found {len(candidates)} improvements")
        
        return candidates
    
    def _test_reverse_edges(self, current_graph: Dict, data: np.ndarray,
                            variable_names: List[str], engine,
                            current_ll: float, allowed_edges: Optional[Set],
                            interventions: np.ndarray,
                            variable_type: str,
                            verbose: bool) -> List[Tuple]:
        """测试反转边"""
        candidates = []
        current_edges = self._get_current_edges_from_graph(current_graph)
        
        tested_count = 0
        for edge in current_edges:
            parent, child = edge
            reversed_edge = (child, parent)
            
            # 检查反转后是否在skeleton允许的边中
            if allowed_edges and reversed_edge not in allowed_edges:
                continue
            
            # 测试反转这条边
            test_graph = self._reverse_edge_in_graph(current_graph, parent, child)
            
            # 检查是否产生环
            if self._has_cycle(test_graph):
                continue
            
            # 评估
            test_ll = self._evaluate_graph(test_graph, data, interventions, variable_type, engine)
            tested_count += 1
            
            if test_ll > current_ll:
                candidates.append(('REVERSE', edge, test_ll, test_graph))
                if verbose and len(candidates) <= 3:
                    print(f"  REVERSE {parent}→{child} to {child}→{parent}: LL={test_ll:.4f} (Δ={test_ll-current_ll:+.4f})")
        
        if verbose:
            print(f"  [REVERSE] Tested {tested_count} edges, found {len(candidates)} improvements")
        
        return candidates
    
    def _evaluate_graph(self, graph: Dict, data: np.ndarray, interventions: np.ndarray, 
                        variable_type: str, engine) -> float:
        """用MLP评估图的log-likelihood（静默模式）"""
        try:
            # 使用suppress_output确保完全静默
            with suppress_output():
                results = engine.fit(
                    structured_graph=graph,
                    data=data,
                    interventions=interventions,
                    variable_type=variable_type,
                    num_epochs=self.eval_epochs,
                    learning_rate=0.01,
                    verbose=False,
                    seed=42
                )
            return results['log_likelihood']
        except Exception as e:
            # 只在真正出错时才打印（不抑制）
            print(f"Warning: Graph evaluation failed: {e}")
            return float('-inf')
    
    def _get_current_edges(self, graph: Dict, variable_names: List[str]) -> Set[Tuple]:
        """获取当前图中的所有边"""
        edges = set()
        for node in graph['nodes']:
            child = node['name']
            for parent in node.get('parents', []):
                edges.add((parent, child))
        return edges
    
    def _get_current_edges_from_graph(self, graph: Dict) -> List[Tuple]:
        """从图中提取所有边"""
        edges = []
        for node in graph['nodes']:
            child = node['name']
            for parent in node.get('parents', []):
                edges.append((parent, child))
        return edges
    
    def _add_edge_to_graph(self, graph: Dict, parent: str, child: str) -> Dict:
        """向图中添加一条边"""
        new_graph = deepcopy(graph)
        for node in new_graph['nodes']:
            if node['name'] == child:
                if parent not in node.get('parents', []):
                    node['parents'].append(parent)
                break
        new_graph['metadata']['num_edges'] = self._count_edges(new_graph)
        return new_graph
    
    def _remove_edge_from_graph(self, graph: Dict, parent: str, child: str) -> Dict:
        """从图中删除一条边"""
        new_graph = deepcopy(graph)
        for node in new_graph['nodes']:
            if node['name'] == child:
                if parent in node.get('parents', []):
                    node['parents'].remove(parent)
                break
        new_graph['metadata']['num_edges'] = self._count_edges(new_graph)
        return new_graph
    
    def _reverse_edge_in_graph(self, graph: Dict, parent: str, child: str) -> Dict:
        """反转图中的一条边"""
        new_graph = deepcopy(graph)
        # 删除原边
        for node in new_graph['nodes']:
            if node['name'] == child and parent in node.get('parents', []):
                node['parents'].remove(parent)
        # 添加反向边
        for node in new_graph['nodes']:
            if node['name'] == parent:
                if child not in node.get('parents', []):
                    node['parents'].append(child)
                break
        new_graph['metadata']['num_edges'] = self._count_edges(new_graph)
        return new_graph
    
    def _count_edges(self, graph: Dict) -> int:
        """计算图中的边数"""
        count = 0
        for node in graph['nodes']:
            count += len(node.get('parents', []))
        return count
    
    def _has_cycle(self, graph: Dict) -> bool:
        """检查图是否有环"""
        G = nx.DiGraph()
        for node in graph['nodes']:
            child = node['name']
            for parent in node.get('parents', []):
                G.add_edge(parent, child)
        
        try:
            nx.find_cycle(G)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    def _get_allowed_edges(self, skeleton_constraints: Dict, 
                           variable_names: List[str]) -> Set[Tuple]:
        """从skeleton约束中获取允许的边"""
        if not skeleton_constraints:
            return None
        
        allowed = set()
        for parent, child in skeleton_constraints.get('allowed_edges', []):
            allowed.add((parent, child))
        
        return allowed

