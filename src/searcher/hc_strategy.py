from typing import Dict, List
import copy
import os
import json

from .search_strategy import SearchStrategy
from utils.metrics import compute_metrics

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
            metrics = compute_metrics(self.pipeline, structured_graph)
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
                from utils.metrics import compute_metrics
                metrics = compute_metrics(self.pipeline, notears_graph)
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
                metrics = compute_metrics(self.pipeline, greedy_graph)
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
