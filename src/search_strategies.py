"""
搜索策略模块 - 封装不同的图搜索方法
包含 Hill Climbing 和 MCTS
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import os
import json
import copy
import math
import random
from src.utils.metrics import _compute_metrics


class SearchStrategy(ABC):
    """搜索策略基类"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.iteration_history = []
    
    @abstractmethod
    def search(self, **kwargs) -> Dict:
        """执行搜索，返回结果字典"""
        pass


class HillClimbingStrategy(SearchStrategy):
    """爬山搜索策略（包含 refinement 调用）"""
    
    def __init__(self, pipeline, acceptance_tolerance: float = 0.0):
        super().__init__(pipeline)
        self.acceptance_tolerance = acceptance_tolerance
    
    def search(
        self,
        num_iterations: int = 3,
        early_stopping_patience: int = 3,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        temperature: float = 0.6,
        llm_model_name: str = "gpt-4o",
        max_tokens: int = 4096,
        verbose: bool = True,
        max_retries: int = 10,
        use_local_amendment: bool = True,
        llm_only: bool = False,
        choose_best: bool = False
    ) -> Dict:
        """Hill Climbing 搜索主循环"""
        
        print(f"\n{'='*70}")
        print(f"SEARCH STRATEGY: Hill Climbing")
        if llm_only:
            print(f"MODE: LLM-only (Stop after first successful global graph)")
        print(f"Acceptance Tolerance: {self.acceptance_tolerance}")
        print(f"Early Stopping Patience: {early_stopping_patience}")
        print(f"Choose Best Initial: {choose_best}")
        print(f"{'='*70}\n")
        
        # 初始化
        best_graph = None
        best_ll = float('-inf')
        current_graph = None
        
        t = 0
        error_memory = []
        failed_attempts = []
        consecutive_rejections = 0  # 连续被拒绝的次数
        
        # 主循环
        for i in range(num_iterations):
            print(f"\n{'🔄 '*35}")
            print(f"ITERATION {t}, Real iteration: {i+1}/{num_iterations}")
            if consecutive_rejections > 0:
                print(f"Consecutive rejections: {consecutive_rejections}/{early_stopping_patience}")
            print(f"{'🔄 '*35}")
            if error_memory:
                error_summary = "\n".join([f"Attempt {i+1}: {err}" 
                                            for i, err in enumerate(error_memory)])
                memory = f"⚠️ Previous Failed Attempts:\n{error_summary}\n\nPlease try different modifications."
            else:
                memory = None
            if current_graph is not None or (self.pipeline.use_baseline_reference==False):
                # ===== 新增：干预实验询问阶段 =====
                exp_reasoning = None
                candidate_operations = None
                if self.pipeline.use_intervention_test:
                    # 传入 edge_notes 以引导实验建议
                    current_edge_notes = None
                    if current_graph and 'metadata' in current_graph:
                        current_edge_notes = current_graph['metadata'].get('edge_notes', {})
                    
                    experiments, exp_reasoning, candidate_operations = self.pipeline.hypothesis_generator.propose_experiments(
                        variable_list=self.pipeline.variable_list,
                        domain_name=self.pipeline.domain_name,
                        domain_context=self.pipeline.domain_context,
                        previous_graph=current_graph,
                        num_experiments=self.pipeline.num_intervention_experiments,
                        model=llm_model_name,
                        temperature=temperature,
                        edge_notes=current_edge_notes  # 引导探索
                    )
                    if experiments:
                        new_evidence = self.pipeline.intervention_tester.run_experiments(experiments)
                        if new_evidence:
                            self.pipeline.accumulated_evidence.append(new_evidence)
                            self.pipeline.policy_verifier.update_evidence(new_evidence)
                            self.pipeline.policy_verifier.update_evidence(new_evidence) # 更新策略校验器
                            print(f"\n🔬 New Interventional Evidence Obtained:\n{new_evidence}")
                
                # 汇总所有历史证据
                interventional_evidence = "\n".join(self.pipeline.accumulated_evidence) if self.pipeline.accumulated_evidence else None
                if interventional_evidence and verbose:
                    print(f"  [Debug] Passing {len(self.pipeline.accumulated_evidence)} cumulative evidence reports to LLM")

                # ===== 获取历史推理和因果档案 =====
                # 优先级：实验动机推理 > 之前的修正推理
                previous_reasoning = exp_reasoning
                confirmed_edges = None
                edge_notes = None
                
                if current_graph and 'metadata' in current_graph:
                    if not previous_reasoning:
                        previous_reasoning = current_graph['metadata'].get('reasoning')
                    confirmed_edges = current_graph['metadata'].get('confirmed_edges', [])
                    edge_notes = current_graph['metadata'].get('edge_notes', {})

                # ===== 生成假设（传入证据和历史推理）=====
                structured_graph, validation_info = self.pipeline.hypothesis_generator.generate_hypothesis(
                    variable_list=self.pipeline.variable_list,
                    domain_name=self.pipeline.domain_name,
                    domain_context=self.pipeline.domain_context,
                    previous_graph=current_graph,
                    memory=memory,
                    iteration=t,
                    model=llm_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_local_amendment=use_local_amendment,
                    num_edge_operations=1,
                    skeleton_constraints=getattr(self.pipeline, 'skeleton_constraints', None),
                    failed_attempts=failed_attempts,
                    baseline_reference=getattr(self.pipeline, 'baseline_reference_text', None),
                    interventional_evidence=interventional_evidence,  # 传入证据
                    previous_reasoning=previous_reasoning,  # 传入历史推理
                    confirmed_edges=confirmed_edges,
                    edge_notes=edge_notes,
                    candidate_operations=candidate_operations  # 传入限定的候选操作
                )
                assert validation_info is not None, "validation_info cannot be None" # TODO:
                assert 'has_cycle' in validation_info, "has_cycle must be in validation_info"
                success_flag = validation_info['success']
                has_cycle_flag = validation_info['has_cycle']
                if has_cycle_flag: assert success_flag is False, "has_cycle must be False if success is False"
                if verbose and not success_flag:
                    print(f"  [Debug] validation_info: success={success_flag}, has_cycle={has_cycle_flag}")
                    if has_cycle_flag and 'cycle_path' in validation_info:
                        print(f"  [Debug] cycle_path: {validation_info['cycle_path']}")

                if (structured_graph is not None and 
                    validation_info is not None and 
                    validation_info.get('success') is True):  # 明确检查是 True，而不是默认值
                    
                    # ===== 强制策略校验 (EVIDENCE-FIRST POLICY) =====
                    proposed_ops = structured_graph['metadata'].get('proposed_operations')
                    
                    if proposed_ops:
                        policy_violations = self.pipeline.policy_verifier.verify_operations(proposed_ops)
                        
                        if policy_violations:
                            print(f"⚠️  EVIDENCE-FIRST POLICY VIOLATION DETECTED!")
                            for violation in policy_violations:
                                print(f"   - {violation}")
                            
                            # 标记为验证失败，并将违反策略的信息加入错误消息
                            validation_info['success'] = False
                            validation_info['error_messages'].extend(policy_violations)
                            # 强制进入失败处理流程
                        else:
                            pass
                    else:
                        # 初始生成或全局修正，没有具体操作列表，跳过针对操作的校验
                        pass

                # 检查验证或策略校验是否成功
                if structured_graph is None or validation_info is None or not validation_info.get('success'):
                    # 从error_messages数组中获取错误信息
                    error_messages = validation_info.get('error_messages', []) if validation_info else ["Unknown validation error"]
                    error_msg = '; '.join(error_messages)
                    print(f"❌ Validation failed: {error_msg}")
                    error_memory.append(error_msg)
                    
                    # 如果有提出的操作，也加入 failed_attempts，以便下一轮避坑
                    if structured_graph and 'metadata' in structured_graph:
                        proposed_ops = structured_graph['metadata'].get('proposed_operations', [])
                        if proposed_ops:
                            failed_ops = [(op['type'], op['parent'], op['child']) for op in proposed_ops]
                            failed_attempts.extend(failed_ops)
                            print(f"  [Debug] Added {len(failed_ops)} failed operations to failed_attempts")
                    
                    # 记录验证失败的迭代
                    self.iteration_history.append({
                        'iteration': t,
                        'graph': structured_graph if structured_graph is not None else None,
                        'accepted': False,
                        'best_ll': best_ll,
                        'current_ll': float('-inf'),
                        'metrics': {},
                        'results': {
                            'log_likelihood': float('-inf'),
                            'bic': None,
                            'num_edges': 0
                        },
                        'validation_failed': True,
                        'rejection_reason': f"Validation failed: {error_msg}",
                        'validation_info': validation_info
                    })
                    
                    # 保存验证失败的图（如果有的话）
                    if structured_graph is not None: # TODO: currently not used
                        graph_path = os.path.join(self.pipeline.output_dir, f"graph_t{t}_validation_failed.json")
                        with open(graph_path, 'w') as f:
                            json.dump(structured_graph, f, indent=2)
                    
                    t += 1
                    continue
            else:
                # 初始迭代 (t=0) 且启用 baseline_reference 时
                # 从基线结果中选择第一个方法作为比较对象
                baseline_graph = next(iter(self.pipeline.baseline_structured_graphs.values()))
                
                if choose_best:
                    print(f"\n[Choose Best] Comparing Baseline vs Global LLM hypothesis...")
                    
                    # 初始阶段也可以做干预测试
                    exp_reasoning = None
                    candidate_operations = None
                    if self.pipeline.use_intervention_test:
                        experiments, exp_reasoning, candidate_operations = self.pipeline.hypothesis_generator.propose_experiments(
                            variable_list=self.pipeline.variable_list,
                            domain_name=self.pipeline.domain_name,
                            domain_context=self.pipeline.domain_context,
                            previous_graph=None,
                            num_experiments=self.pipeline.num_intervention_experiments,
                            model=llm_model_name,
                            temperature=temperature,
                            edge_notes={} # 初始阶段为空
                        )
                        if experiments:
                            new_evidence = self.pipeline.intervention_tester.run_experiments(experiments)
                            if new_evidence:
                                self.pipeline.accumulated_evidence.append(new_evidence)
                                self.pipeline.policy_verifier.update_evidence(new_evidence) # 更新策略校验器
                                print(f"\n🔬 New Interventional Evidence Obtained:\n{new_evidence}")
                    
                    interventional_evidence = "\n".join(self.pipeline.accumulated_evidence) if self.pipeline.accumulated_evidence else None

                    global_graph, validation_info = self.pipeline.hypothesis_generator.generate_hypothesis(
                        variable_list=self.pipeline.variable_list,
                        domain_name=self.pipeline.domain_name,
                        domain_context=self.pipeline.domain_context,
                        previous_graph=None,
                        memory=None,
                        iteration=t,
                        model=llm_model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        use_local_amendment=False,
                        num_edge_operations=1,
                        skeleton_constraints=getattr(self.pipeline, 'skeleton_constraints', None),
                        failed_attempts=failed_attempts,
                        baseline_reference=getattr(self.pipeline, 'baseline_reference_text', None),
                        interventional_evidence=interventional_evidence,
                        previous_reasoning=exp_reasoning,
                        confirmed_edges=[],
                        edge_notes={},
                        candidate_operations=candidate_operations
                    )
                    
                    if validation_info and validation_info.get('success'):
                        # 评估两者
                        self._evaluate_graph(baseline_graph, num_epochs, learning_rate, verbose=False)
                        self._evaluate_graph(global_graph, num_epochs, learning_rate, verbose=False)
                        
                        baseline_bic = baseline_graph['metadata'].get('bic', float('inf'))
                        global_bic = global_graph['metadata'].get('bic', float('inf'))
                        
                        print(f"  - Baseline BIC: {baseline_bic:.2f}")
                        print(f"  - Global LLM BIC: {global_bic:.2f}")
                        
                        if global_bic < baseline_bic:
                            print(f"  🏆 Global LLM wins!")
                            structured_graph = global_graph
                        else:
                            print(f"  🏆 Baseline wins!")
                            structured_graph = baseline_graph
                    else:
                        print(f"  ⚠️ Global LLM generation failed validation, falling back to Baseline.")
                        structured_graph = baseline_graph
                else:
                    structured_graph = baseline_graph
                validation_info = None
            
            # 评估图
            current_ll = self._evaluate_graph(structured_graph, num_epochs, learning_rate, verbose=True)
            
            # 计算 metrics
            metrics = _compute_metrics(self.pipeline, structured_graph)
            structured_graph['metadata']['evaluation_metrics'] = metrics
            print(f"Evaluation metrics:", metrics)
            
            # 应用 refinements (NOTEARS, Greedy)
            if not llm_only:
                structured_graph, current_ll = self._apply_refinements(
                    structured_graph, current_ll, t, num_epochs, learning_rate, verbose
                )
            else:
                print(f"[LLM-only] Skipping refinements (NOTEARS/Greedy)")
            
            # Hill Climbing 决策
            if current_ll >= best_ll - self.acceptance_tolerance:
                # 接受
                if current_ll > best_ll:
                    best_ll = current_ll
                    best_graph = copy.deepcopy(structured_graph)
                
                current_graph = structured_graph
                error_memory = []
                failed_attempts = []
                consecutive_rejections = 0  # 重置重试计数
                
                print(f"\n✅ ACCEPTED (Hill Climbing)")
                print(f"  Current LL: {current_ll:.4f}")
                print(f"  Best LL: {best_ll:.4f}")
                
                # 记录历史 - 成功的迭代
                self.iteration_history.append({
                    'iteration': t,
                    'graph': structured_graph,
                    'accepted': True,
                    'best_ll': best_ll,
                    'current_ll': current_ll,
                    'metrics': metrics,
                    'results': {
                        'log_likelihood': current_ll,
                        'bic': structured_graph['metadata'].get('bic', None),
                        'num_edges': structured_graph['metadata']['num_edges']
                    }
                })
                
                # 保存图
                graph_path = os.path.join(self.pipeline.output_dir, f"graph_t{t}.json") # TODO: currently not used
                with open(graph_path, 'w') as f:
                    json.dump(structured_graph, f, indent=2)
                
                # LLM-only 模式：接受第一个成功的图（global图）后立即停止
                if llm_only:
                    print(f"\n✅ [LLM-only] First successful graph obtained. Terminating iteration.")
                    return {
                        'best_graph': best_graph,
                        'best_ll': best_ll,
                        'current_graph': current_graph,
                        'history': self.iteration_history
                    }

                t += 1
            else:
                # 拒绝
                consecutive_rejections += 1
                print(f"\n❌ REJECTED: LL = {current_ll:.4f} < {best_ll - self.acceptance_tolerance:.4f}")
                print(f"Consecutive rejections: {consecutive_rejections}/{early_stopping_patience}")
                
                assert 'proposed_operations' in structured_graph['metadata'], "currently not implemented no proposed operations"
                operation_proposed = [(operation['type'], operation['parent'], operation['child']) for operation in structured_graph['metadata']['proposed_operations']]
                error_msg = f"LL too low: {current_ll:.4f}, proposed operations: {operation_proposed}"
                error_memory.append(error_msg)
                failed_attempts.extend(operation_proposed)
                assert len(operation_proposed) == 1, "currently not implemented multiple proposed operations"
                
                # 记录历史 - 失败的迭代
                self.iteration_history.append({
                    'iteration': t,
                    'graph': structured_graph,
                    'accepted': False,
                    'best_ll': best_ll,
                    'current_ll': current_ll,
                    'metrics': metrics,
                    'results': {
                        'log_likelihood': current_ll,
                        'bic': structured_graph['metadata'].get('bic', None),
                        'num_edges': structured_graph['metadata']['num_edges']
                    },
                    'rejection_reason': error_msg,
                    'proposed_operations': operation_proposed
                })
                
                # 保存被拒绝的图（使用不同的命名以便区分）
                graph_path = os.path.join(self.pipeline.output_dir, f"graph_t{t}_rejected.json") # TODO: currently not used
                with open(graph_path, 'w') as f:
                    json.dump(structured_graph, f, indent=2)
                
                # 检查早停
                if consecutive_rejections >= early_stopping_patience:
                    print(f"\n🛑 EARLY STOPPING: {consecutive_rejections} consecutive rejections reached.")
                    break
                
                t += 1
        
        return {
            'best_graph': best_graph,
            'best_ll': best_ll,
            'current_graph': current_graph,
            'history': self.iteration_history
        }
    
    def _generate_graph(self, current_graph, iteration, llm_model_name,
                       temperature, max_tokens, max_retries, error_memory, verbose, use_local_amendment=True, 
                       interventional_evidence=None, previous_reasoning=None, confirmed_edges=None, edge_notes=None):
        """生成或修改图（带重试）"""
        
        retry_count = 0
        failed_initial_attempts = []
        
        while retry_count < max_retries:
            memory = None
            if error_memory:
                error_summary = "\n".join([f"Attempt {i+1}: {err}" 
                                            for i, err in enumerate(error_memory)])
                memory = f"⚠️ Previous Failed Attempts:\n{error_summary}\n\nPlease try different modifications."
            
            # 提取历史推理和档案 (如果没有从外部传入)
            if current_graph and 'metadata' in current_graph:
                if previous_reasoning is None:
                    previous_reasoning = current_graph['metadata'].get('reasoning')
                if confirmed_edges is None:
                    confirmed_edges = current_graph['metadata'].get('confirmed_edges', [])
                if edge_notes is None:
                    edge_notes = current_graph['metadata'].get('edge_notes', {})
            
            # 生成图
            structured_graph, validation_info = self.pipeline.hypothesis_generator.generate_hypothesis(
                variable_list=self.pipeline.variable_list,
                domain_name=self.pipeline.domain_name,
                domain_context=self.pipeline.domain_context,
                previous_graph=current_graph,
                memory=memory,
                iteration=iteration,
                model=llm_model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                use_local_amendment=use_local_amendment,
                num_edge_operations=1,
                skeleton_constraints=getattr(self.pipeline, 'skeleton_constraints', None),
                failed_attempts=failed_initial_attempts if failed_initial_attempts else None,
                baseline_reference=getattr(self.pipeline, 'baseline_reference_text', None),
                interventional_evidence=interventional_evidence,
                previous_reasoning=previous_reasoning,
                confirmed_edges=confirmed_edges,
                edge_notes=edge_notes
            )
            # 检查验证结果
            # Check both that graph exists and validation passed
            # 调试：打印validation_info的内容
            assert validation_info is not None, "validation_info cannot be None" # TODO:
            assert 'has_cycle' in validation_info, "has_cycle must be in validation_info"
            success_flag = validation_info['success']
            has_cycle_flag = validation_info['has_cycle']
            if has_cycle_flag: assert success_flag is False, "has_cycle must be False if success is False"
            if verbose and not success_flag:
                print(f"  [Debug] validation_info: success={success_flag}, has_cycle={has_cycle_flag}")
                if has_cycle_flag and 'cycle_path' in validation_info:
                    print(f"  [Debug] cycle_path: {validation_info['cycle_path']}")
            
            # 只有在明确标记为成功时才接受图
            # validation_info 必须存在，且 success 必须明确为 True
            if (structured_graph is not None and 
                validation_info is not None and 
                validation_info.get('success') is True):  # 明确检查是 True，而不是默认值
                return structured_graph
            else:
                # 从error_messages数组中获取错误信息
                error_messages = validation_info.get('error_messages', []) if validation_info else []
                error_msg = '; '.join(error_messages) if error_messages else ('Graph is None' if structured_graph is None else 'Unknown error')
                print(f"  [Retry {retry_count+1}] Validation failed: {error_msg}")
                
                retry_count += 1
        
        return None
    
    def _evaluate_graph(self, graph, num_epochs, learning_rate, verbose=False):
        """评估图的 log-likelihood"""
        fitting_results = self.pipeline.fitting_engine.fit(
            structured_graph=graph,
            data=self.pipeline.data,
            interventions=self.pipeline.interventions,
            variable_type=self.pipeline.variable_type,  # 传递变量类型
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            seed=42
        )
        graph['metadata']['log_likelihood'] = fitting_results['log_likelihood']
        graph['metadata']['num_parameters'] = fitting_results['num_parameters']
        graph['metadata']['bic'] = fitting_results['bic']  # 保存BIC
        return fitting_results['log_likelihood']
    
    def _apply_refinements(self, graph, current_ll, iteration, num_epochs, learning_rate, verbose):
        """应用 NOTEARS 和 Greedy refinements"""
        
        # NOTEARS refinement
        if self.pipeline.use_notears_refinement and iteration >= self.pipeline.notears_start_iter:
            graph, current_ll = self._apply_notears(graph, current_ll, num_epochs, learning_rate, verbose)
        
        # Greedy refinement
        if self.pipeline.use_greedy_refinement and iteration >= self.pipeline.greedy_start_iter:
            graph, current_ll = self._apply_greedy(graph, current_ll, num_epochs, learning_rate, verbose)
        
        return graph, current_ll
    
    def _apply_notears(self, graph, original_ll, num_epochs, learning_rate, verbose):
        """应用 NOTEARS refinement"""
        
        print(f"\n{'🔬'*35}")
        print(f"NOTEARS REFINEMENT")
        print(f"{'🔬'*35}")
        
        try:
            if self.pipeline.notears_use_mlp:
                # MLP 版本
                notears_graph, refinement_info = self.pipeline.notears_refiner.refine_graph(
                    initial_graph=graph,
                    data=self.pipeline.data,
                    variable_names=self.pipeline.variable_list,
                    engine=self.pipeline.fitting_engine
                )
                notears_ll = refinement_info['refined_ll']
            else:
                # Ridge 版本
                notears_graph, refinement_info = self.pipeline.notears_refiner.refine_graph(
                    initial_graph=graph,
                    data=self.pipeline.data,
                    variable_names=self.pipeline.variable_list,
                    locally_only=True
                )
                notears_ll = self._evaluate_graph(notears_graph, num_epochs, learning_rate, verbose=False)
            
            print(f"\nLLM LL: {original_ll:.4f}, NOTEARS LL: {notears_ll:.4f}, Δ: {notears_ll-original_ll:+.4f}")
            
            if notears_ll > original_ll:
                print(f"✅ NOTEARS improved!")
                from src.core.cma_pipeline import _compute_metrics
                metrics = _compute_metrics(self.pipeline, notears_graph)
                notears_graph['metadata']['evaluation_metrics'] = metrics
                notears_graph['metadata']['notears_refinement'] = refinement_info
                print(f"{'🔬'*35}\n")
                return notears_graph, notears_ll
            else:
                print(f"⚠️ Keeping LLM graph")
                print(f"{'🔬'*35}\n")
                return graph, original_ll
        
        except Exception as e:
            print(f"❌ NOTEARS failed: {e}")
            print(f"{'🔬'*35}\n")
            return graph, original_ll
    
    def _apply_greedy(self, graph, original_ll, num_epochs, learning_rate, verbose):
        """应用 Greedy refinement"""
        
        print(f"\n{'🎯'*35}")
        print(f"GREEDY REFINEMENT")
        print(f"{'🎯'*35}")
        
        try:
            skeleton_constraints = getattr(self.pipeline, 'skeleton_constraints', None)
            allowed_edges = skeleton_constraints.get('allowed_edges_set') if skeleton_constraints else None
            
            greedy_graph, greedy_info = self.pipeline.greedy_refiner.refine_graph(
                initial_graph=graph,
                data=self.pipeline.data,
                variable_names=self.pipeline.variable_list,
                model_fitting_engine=self.pipeline.fitting_engine,
                interventions=self.pipeline.interventions,
                variable_type=self.pipeline.variable_type,
                skeleton_constraints=skeleton_constraints,
                verbose=True,
                seed=42
            )
            
            greedy_ll = greedy_info['final_ll']
            
            print(f"\nLLM LL: {original_ll:.4f}, Greedy LL: {greedy_ll:.4f}, Δ: {greedy_ll-original_ll:+.4f}")
            
            if greedy_ll > original_ll:
                print(f"✅ Greedy improved!")
                from src.core.cma_pipeline import _compute_metrics
                metrics = _compute_metrics(self.pipeline, greedy_graph)
                greedy_graph['metadata']['evaluation_metrics'] = metrics
                greedy_graph['metadata']['greedy_refinement'] = greedy_info
                print(f"{'🎯'*35}\n")
                return greedy_graph, greedy_ll
            else:
                print(f"⚠️ Keeping LLM graph")
                print(f"{'🎯'*35}\n")
                return graph, original_ll
        
        except Exception as e:
            print(f"❌ Greedy failed: {e}")
            print(f"{'🎯'*35}\n")
            return graph, original_ll


# 未来的 MCTS 策略
class MCTSNode: # TODO:
    """MCTS 树节点"""
    
    def __init__(self, graph: Dict, parent: Optional['MCTSNode'] = None, 
                 ll: float = float('-inf'), iteration: int = 0):
        self.graph = graph  # 当前图结构
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累计奖励
        self.ll = ll  # 当前图的log-likelihood
        self.iteration = iteration  # 对应的迭代次数
        self.is_fully_expanded = False  # 是否已完全扩展
    
    def is_leaf(self) -> bool:
        """判断是否为叶节点"""
        return len(self.children) == 0
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """使用UCB1选择最佳子节点"""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # 优先选择未访问的节点
                return child
            
            # UCB1公式: exploitation + exploration
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def most_visited_child(self) -> 'MCTSNode':
        """返回访问次数最多的子节点（最终选择）"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


class MCTSStrategy(SearchStrategy):
    """蒙特卡洛树搜索策略"""
    
    def __init__(self, pipeline, num_simulations: int = 100, 
                 exploration_weight: float = 1.414, max_depth: int = 5):
        super().__init__(pipeline)
        self.num_simulations = num_simulations  # 每次迭代的模拟次数
        self.exploration_weight = exploration_weight  # UCB1探索权重
        self.max_depth = max_depth  # 最大搜索深度
        self.root = None  # 搜索树根节点
    
    def search(
        self,
        num_iterations: int = 3,
        early_stopping_patience: int = 3,  # MCTS 暂不强制使用，但保持接口一致
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        temperature: float = 0.6,
        llm_model_name: str = "gpt-4o",
        max_tokens: int = 4096,
        verbose: bool = True,
        max_retries: int = 10,
        use_local_amendment: bool = True,
        llm_only: bool = False,
        choose_best: bool = False
    ) -> Dict:
        """MCTS 搜索主循环"""
        
        print(f"\n{'='*70}")
        print(f"SEARCH STRATEGY: Monte Carlo Tree Search (MCTS)")
        if llm_only:
            print(f"MODE: LLM-only (Stop after first successful global graph)")
        print(f"Simulations per iteration: {self.num_simulations}")
        print(f"Exploration weight: {self.exploration_weight}")
        print(f"Max depth: {self.max_depth}")
        print(f"Choose Best Initial: {choose_best}")
        print(f"{'='*70}\n")
        
        best_graph = None
        best_ll = float('-inf')
        
        # 生成初始图
        print(f"🌱 Generating initial graph...")
        if choose_best and self.pipeline.use_baseline_reference:
            print(f"\n[Choose Best] Comparing Baseline vs Global LLM initial graph...")
            baseline_graph = next(iter(self.pipeline.baseline_structured_graphs.values()))
            global_graph = self._generate_initial_graph(
                llm_model_name, temperature, max_tokens, 
                max_retries, num_epochs, learning_rate, verbose
            )
            
            if global_graph is not None:
                # 评估两者
                self._evaluate_graph(baseline_graph, num_epochs, learning_rate, verbose=False)
                self._evaluate_graph(global_graph, num_epochs, learning_rate, verbose=False)
                
                baseline_bic = baseline_graph['metadata'].get('bic', float('inf'))
                global_bic = global_graph['metadata'].get('bic', float('inf'))
                
                print(f"  - Baseline BIC: {baseline_bic:.2f}")
                print(f"  - Global LLM BIC: {global_bic:.2f}")
                
                if global_bic < baseline_bic:
                    print(f"  🏆 Global LLM wins!")
                    initial_graph = global_graph
                else:
                    print(f"  🏆 Baseline wins!")
                    initial_graph = baseline_graph
            else:
                print(f"  ⚠️ Global LLM generation failed, falling back to Baseline.")
                initial_graph = baseline_graph
        else:
            initial_graph = self._generate_initial_graph(
                llm_model_name, temperature, max_tokens, 
                max_retries, num_epochs, learning_rate, verbose
            )
        
        if initial_graph is None:
            print("❌ Failed to generate initial graph")
            return {
                'best_graph': None,
                'best_ll': float('-inf'),
                'current_graph': None,
                'history': self.iteration_history
            }
        
        # 评估初始图
        initial_ll = self._evaluate_graph(initial_graph, num_epochs, learning_rate, verbose)
        best_graph = copy.deepcopy(initial_graph)
        best_ll = initial_ll
        
        # 记录初始图
        from src.core.cma_pipeline import _compute_metrics
        initial_metrics = _compute_metrics(self.pipeline, initial_graph)
        initial_graph['metadata']['evaluation_metrics'] = initial_metrics
        
        self.iteration_history.append({
            'iteration': 0,
            'graph': initial_graph,
            'accepted': True,
            'best_ll': best_ll,
            'current_ll': initial_ll,
            'metrics': initial_metrics,
            'results': {
                'log_likelihood': initial_ll,
                'bic': initial_graph['metadata'].get('bic', None),
                'num_edges': initial_graph['metadata']['num_edges']
            }
        })
        
        # self._save_graph(initial_graph, 0) # TODO: currently not used
        
        if self.pipeline.dataset is not None:
            self.pipeline._evaluate_against_ground_truth(initial_graph)
        
        # LLM-only 模式：获得第一个成功的图后立即停止
        if llm_only:
            print(f"\n✅ [LLM-only] Initial successful graph obtained. Terminating MCTS.")
            return {
                'best_graph': best_graph,
                'best_ll': best_ll,
                'current_graph': best_graph,
                'history': self.iteration_history
            }

        # 初始化MCTS根节点
        self.root = MCTSNode(initial_graph, parent=None, ll=initial_ll, iteration=0)
        self.root.visits = 1
        self.root.value = initial_ll
        
        print(f"✅ Initial graph: LL = {initial_ll:.4f}\n")
        
        # MCTS迭代
        for t in range(1, num_iterations):
            print(f"\n🔄 MCTS ITERATION {t}/{num_iterations-1}")
            print(f"Current best LL: {best_ll:.4f}")
            
            # 执行多次MCTS模拟
            for sim in range(self.num_simulations):
                if verbose and (sim + 1) % 10 == 0:
                    print(f"  Simulation {sim + 1}/{self.num_simulations}...")
                
                # MCTS四个步骤
                leaf = self._select(self.root)  # 选择
                child = self._expand(leaf, t, llm_model_name, temperature, 
                                    max_tokens, max_retries, num_epochs, 
                                    learning_rate, verbose)  # 扩展
                
                if child is not None:
                    reward = self._simulate(child)  # 模拟（评估）
                    self._backpropagate(child, reward)  # 回溯
            
            # 选择最佳子节点作为下一个根节点
            if self.root.children:
                best_child = self.root.most_visited_child()
                
                if best_child and best_child.ll > best_ll:
                    best_ll = best_child.ll
                    best_graph = copy.deepcopy(best_child.graph)
                    print(f"\n✅ NEW BEST: LL = {best_ll:.4f} (from {best_child.visits} visits)")
                
                # 更新根节点到最佳子节点
                self.root = best_child
                self.root.parent = None
                
                # 记录到历史
                metrics = _compute_metrics(self.pipeline, best_child.graph)
                best_child.graph['metadata']['evaluation_metrics'] = metrics
                
                self.iteration_history.append({
                    'iteration': t,
                    'graph': best_child.graph,
                    'accepted': True,
                    'best_ll': best_ll,
                    'current_ll': best_child.ll,
                    'metrics': metrics,
                    'results': {
                        'log_likelihood': best_child.ll,
                        'bic': best_child.graph['metadata'].get('bic', None),
                        'num_edges': best_child.graph['metadata']['num_edges']
                    }
                })
                
                # self._save_graph(best_child.graph, t) # TODO: currently not used
                
                if self.pipeline.dataset is not None:
                    self.pipeline._evaluate_against_ground_truth(best_child.graph)
            else:
                print(f"\n⚠️ No valid children found, stopping MCTS")
                break
        
        return {
            'best_graph': best_graph,
            'best_ll': best_ll,
            'current_graph': self.root.graph if self.root else best_graph,
            'history': self.iteration_history
        }
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：从根节点选择到叶节点"""
        current = node
        depth = 0
        
        while not current.is_leaf() and depth < self.max_depth:
            current = current.best_child(self.exploration_weight)
            depth += 1
        
        return current
    
    def _expand(self, node: MCTSNode, iteration: int, llm_model_name: str,
                temperature: float, max_tokens: int, max_retries: int,
                num_epochs: int, learning_rate: float, verbose: bool) -> Optional[MCTSNode]:
        """扩展阶段：为叶节点生成新的子节点"""
        
        if node.is_fully_expanded:
            return None
        
        # ===== 新增：干预实验询问阶段 =====
        all_evidence = None
        candidate_operations = None
        if self.pipeline.use_intervention_test:
            current_edge_notes = node.graph['metadata'].get('edge_notes', {}) if node.graph and 'metadata' in node.graph else {}
            experiments, _, candidate_operations = self.pipeline.hypothesis_generator.propose_experiments(
                variable_list=self.pipeline.variable_list,
                domain_name=self.pipeline.domain_name,
                domain_context=self.pipeline.domain_context,
                previous_graph=node.graph,
                num_experiments=self.pipeline.num_intervention_experiments,
                model=llm_model_name,
                temperature=temperature,
                edge_notes=current_edge_notes
            )
            if experiments:
                new_evidence = self.pipeline.intervention_tester.run_experiments(experiments)
                if new_evidence:
                    self.pipeline.accumulated_evidence.append(new_evidence)
                    self.pipeline.policy_verifier.update_evidence(new_evidence)
        
        all_evidence = "\n".join(self.pipeline.accumulated_evidence) if self.pipeline.accumulated_evidence else None

        # 使用LLM生成局部修改
        modified_graph, validation_info = self.pipeline.hypothesis_generator.generate_hypothesis(
            variable_list=self.pipeline.variable_list,
            domain_name=self.pipeline.domain_name,
            domain_context=self.pipeline.domain_context,
            previous_graph=node.graph,
            memory=None,
            iteration=iteration,
            model=llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            use_local_amendment=True,
            skeleton_constraints=getattr(self.pipeline, 'skeleton_constraints', None),
            interventional_evidence=all_evidence,
            candidate_operations=candidate_operations
        )

        # Check that graph exists and validation passed
        if modified_graph is None or not (validation_info and validation_info.get('success', True)):
            node.is_fully_expanded = True
            return None
        
        # ===== 强制策略校验 (EVIDENCE-FIRST POLICY) =====
        proposed_ops = modified_graph['metadata'].get('proposed_operations')
        
        if proposed_ops:
            policy_violations = self.pipeline.policy_verifier.verify_operations(proposed_ops)
            
            if policy_violations:
                print(f"⚠️  EVIDENCE-FIRST POLICY VIOLATION DETECTED in MCTS expansion!")
                for violation in policy_violations:
                    print(f"   - {violation}")
                # MCTS 中，如果违反策略，我们认为该扩展无效
                return None
        else:
            # 初始生成或全局修正，跳过针对操作的校验
            pass
        
        # 评估新图
        ll = self._evaluate_graph(modified_graph, num_epochs, learning_rate, verbose=False)
        
        # 应用refinement（如果启用）
        modified_graph, ll = self._apply_refinements(
            modified_graph, ll, iteration, num_epochs, learning_rate, verbose=False
        )
        
        # 创建子节点
        child = MCTSNode(modified_graph, parent=node, ll=ll, iteration=iteration)
        node.children.append(child)
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """模拟阶段：评估节点的价值（直接返回LL）"""
        # 对于因果图搜索，我们直接使用LL作为奖励
        # 不需要随机rollout，因为我们已经有了确定的评估指标
        return node.ll
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """回溯阶段：更新路径上所有节点的统计信息"""
        current = node
        
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent
    
    def _generate_initial_graph(self, llm_model_name: str, temperature: float,
                                max_tokens: int, max_retries: int, num_epochs: int,
                                learning_rate: float, verbose: bool) -> Optional[Dict]:
        """生成初始图（带重试）"""
        
        failed_attempts = []
        
        # 初始干预测试
        all_evidence = None
        candidate_operations = None
        if self.pipeline.use_intervention_test:
            experiments, _, candidate_operations = self.pipeline.hypothesis_generator.propose_experiments(
                variable_list=self.pipeline.variable_list,
                domain_name=self.pipeline.domain_name,
                domain_context=self.pipeline.domain_context,
                previous_graph=None,
                num_experiments=self.pipeline.num_intervention_experiments,
                model=llm_model_name,
                temperature=temperature,
                edge_notes={}
            )
            if experiments:
                new_evidence = self.pipeline.intervention_tester.run_experiments(experiments)
                if new_evidence:
                    self.pipeline.accumulated_evidence.append(new_evidence)
                    self.pipeline.policy_verifier.update_evidence(new_evidence)
        
        all_evidence = "\n".join(self.pipeline.accumulated_evidence) if self.pipeline.accumulated_evidence else None
        
        for retry in range(max_retries):
            try:
                structured_graph, validation_info = self.pipeline.hypothesis_generator.generate_hypothesis(
                    variable_list=self.pipeline.variable_list,
                    domain_name=self.pipeline.domain_name,
                    domain_context=self.pipeline.domain_context,
                    previous_graph=None,
                    memory=None,
                    iteration=0,
                    model=llm_model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_local_amendment=False,
                    skeleton_constraints=getattr(self.pipeline, 'skeleton_constraints', None),
                    failed_attempts=failed_attempts if failed_attempts else None,
                    baseline_reference=getattr(self.pipeline, 'baseline_reference_text', None),
                    interventional_evidence=all_evidence,
                    candidate_operations=candidate_operations
                )
                
                # Check both that graph exists and validation passed
                if structured_graph is not None and validation_info and validation_info.get('success', True):
                    return structured_graph
                else:
                    error_messages = validation_info.get('error_messages', []) if validation_info else []
                    error_msg = '; '.join(error_messages) if error_messages else ('Graph is None' if structured_graph is None else 'Unknown error')
                    print(f"  [Retry {retry+1}] Validation failed: {error_msg}")
                    
                    if structured_graph:
                        failed_attempts.append({
                            'graph': structured_graph,
                            'error': error_msg,
                            'cycle_path': validation_info.get('cycle_path')
                        })
                        
            except (TypeError, AttributeError) as e:
                # 参数错误或属性错误，立即抛出，不重试
                print(f"\n❌ FATAL ERROR: {type(e).__name__}: {str(e)}")
                print("This is a programming error, not a retry-able failure.")
                raise
            except Exception as e:
                # 其他错误，可以重试
                print(f"  [Retry {retry+1}] Exception: {str(e)}")
                if retry == max_retries - 1:
                    # 最后一次重试也失败了，抛出异常
                    print(f"\n❌ All {max_retries} retries failed.")
                    raise
        
        return None
    
    def _evaluate_graph(self, graph: Dict, num_epochs: int, 
                       learning_rate: float, verbose: bool = False) -> float:
        """评估图的log-likelihood"""
        from model_fitting import ModelFittingEngine
        
        engine = ModelFittingEngine(device=self.pipeline.device)
        
        results = engine.fit(
            structured_graph=graph,
            data=self.pipeline.data,
            interventions=self.pipeline.interventions,
            variable_type=self.pipeline.variable_type,  # 传递变量类型
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        return results['log_likelihood']
    
    def _apply_refinements(self, graph: Dict, ll: float, iteration: int,
                          num_epochs: int, learning_rate: float, 
                          verbose: bool) -> tuple:
        """应用NOTEARS和Greedy refinement（如果启用）"""
        
        # NOTEARS refinement
        if (self.pipeline.use_notears_refinement and 
            iteration >= self.pipeline.notears_start_iter):
            
            if verbose:
                print(f"\n🔧 Applying NOTEARS refinement...")
            
            refined_graph = self.pipeline.notears_refiner.refine_graph(
                initial_graph=graph,
                data=self.pipeline.data,
                num_epochs=num_epochs,
                lr=learning_rate
            )
            
            if refined_graph is not None:
                refined_ll = self._evaluate_graph(
                    refined_graph, num_epochs, learning_rate, verbose=False
                )
                
                if refined_ll > ll:
                    if verbose:
                        print(f"  ✅ NOTEARS improved LL: {ll:.4f} → {refined_ll:.4f}")
                    graph = refined_graph
                    ll = refined_ll
                else:
                    if verbose:
                        print(f"  ⚠️ NOTEARS did not improve (kept original)")
        
        # Greedy refinement
        if (self.pipeline.use_greedy_refinement and 
            iteration >= self.pipeline.greedy_start_iter):
            
            if verbose:
                print(f"\n🔧 Applying Greedy refinement...")
            
            refined_graph = self.pipeline.greedy_refiner.refine_graph(
                initial_graph=graph
            )
            
            if refined_graph is not None:
                refined_ll = self._evaluate_graph(
                    refined_graph, num_epochs, learning_rate, verbose=False
                )
                
                if refined_ll > ll:
                    if verbose:
                        print(f"  ✅ Greedy improved LL: {ll:.4f} → {refined_ll:.4f}")
                    graph = refined_graph
                    ll = refined_ll
                else:
                    if verbose:
                        print(f"  ⚠️ Greedy did not improve (kept original)")
        
        return graph, ll
    
    def _save_graph(self, graph: Dict, iteration: int):
        """保存图到JSON文件"""
        graph_path = os.path.join(self.pipeline.output_dir, f"graph_t{iteration}.json")
        with open(graph_path, 'w') as f:
            json.dump(graph, f, indent=2)

