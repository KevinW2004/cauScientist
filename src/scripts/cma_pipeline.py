"""
主运行流程
CMA Pipeline - 完整版
同时支持:
1. 本地模型 / OpenAI API 切换
2. 数据加载功能
3. 真实图对比评估
"""
import json
import numpy as np
import os
from typing import Dict, List

from llm_hypothesis import LLMHypothesisGenerator
from post_processing import PostProcessor
from data_loader import CausalDataset, DOMAIN_CONTEXTS
from utils.score_functions import score_graph_with_bic
from transformers import AutoTokenizer

from utils import ConfigManager
from utils.metrics import compute_metrics
from llm_loader import LLMLoader, LLMLoaderFactory

os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

class CMAPipeline:
    """CMA完整流程管理器 - 支持本地模型和数据加载"""
    
    def __init__(
        self,
        # 数据参数 - 三选一
        domain_name: str = None,
        variable_list: List[str] = None,
        data: np.ndarray = None,
        dataset: CausalDataset = None,  # 新增: 直接传入数据集
        domain_context: str = "",
        use_observational_only: bool = True,  # 新增: 是否只用观测数据
    ):
        """
        初始化CMA流程
        
        Args:
            # 方式1: 手动指定数据
            domain_name: 领域名称
            variable_list: 变量列表
            data: 数据 [n_samples, n_variables]
            domain_context: 领域背景知识
            
            # 方式2: 传入CausalDataset对象
            dataset: CausalDataset对象(包含数据、真实图、变量名等)
            use_observational_only: 是否只使用观测数据(排除干预样本)
            
            # 输出配置
            output_dir: 输出目录
            device: 模型拟合设备
        """
        self.config = ConfigManager()
        config = self.config
        self.output_dir = config.get("experiment.output.dir", "./cma_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = config.get("llm.local.device", "cuda")
        self.llm_type = config.get("llm.type")
        
        # ===== 处理数据输入(两种方式) =====
        if dataset is not None:
            # 方式2: 使用CausalDataset
            self.dataset = dataset
            self.domain_name = dataset.domain_name
            self.variable_list = dataset.variable_names
            
            # 选择使用全部数据还是只用观测数据
            if use_observational_only and dataset.interventions is not None:
                self.data = dataset.get_observational_data()
                print(f"[Info] Using observational data only: {self.data.shape}")
            else:
                self.data = dataset.data
                print(f"[Info] Using all data: {self.data.shape}")
            
            # 使用预定义的领域背景
            if not domain_context:
                self.domain_context = DOMAIN_CONTEXTS.get(dataset.domain_name, "")
            else:
                self.domain_context = domain_context
                
        else:
            # 方式1: 手动指定
            if domain_name is None or variable_list is None or data is None:
                raise ValueError(
                    "Either provide 'dataset' OR all of ('domain_name', 'variable_list', 'data')"
                )
            
            self.dataset = None
            self.domain_name = domain_name
            self.variable_list = variable_list
            self.data = data
            self.domain_context = domain_context
        
        
        # 保存数据集信息
        if self.dataset:
            self._save_dataset_info()
        
        # ===== 集中加载LLM =====
        llm_type = config.get("llm.type")
        print("\n" + "="*70)
        print("INITIALIZING LLM BACKEND")
        print("="*70)
        print(f"LLM Type: {llm_type}")
        self.llm_loader: LLMLoader = LLMLoaderFactory.create_llm_loader(llm_type)
        self.llm_loader.load_model()

        # 统一注入 llm_loader 到 hypothesis generator 和 post processor
        self.hypothesis_generator = LLMHypothesisGenerator(llm_loader=self.llm_loader)
        # self.post_processor = PostProcessor(llm_loader=self.llm_loader)
        
        # 存储历史
        self.iteration_history = []
        
        print("✓ Pipeline initialized successfully!")
        print("="*70 + "\n")
    
    def _save_dataset_info(self):
        """保存数据集信息(仅当有dataset时)"""
        info = {
            "domain": self.dataset.domain_name,
            "n_variables": self.dataset.n_variables,
            "n_samples_total": self.dataset.n_samples,
            "n_samples_used": len(self.data),
            "variable_names": self.dataset.variable_names,
            "ground_truth_edges": self.dataset.get_ground_truth_edges(),
            "intervention_summary": self.dataset.get_intervention_summary()
        }
        
        info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"[Info] Dataset info saved to {info_path}")

    
    def run(
        self,
        verbose: bool = True
    ) -> Dict:
        """运行完整的CMA流程"""
        num_iterations = self.config.get("experiment.training.num_iterations", 3)
        
        print("\n" + "="*70)
        print(f"STARTING CMA PIPELINE: {self.domain_name.upper()}")
        print("="*70)
        
        # 打印数据集摘要
        if self.dataset:
            self.dataset.print_summary()
        else:
            print(f"Variables: {len(self.variable_list)}")
            print(f"Data shape: {self.data.shape}")
        
        print(f"LLM Type: {self.llm_type}")
        print(f"Iterations: {num_iterations}")
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        previous_graph = None
        previous_results = None
        memory = None
        best = None
        t = 0
        
        for i in range(num_iterations):
            print("\n" + "🔄 "*35)
            print(f"ITERATION {t}")
            print("🔄 "*35)
            
            # ===== 步骤1: 假设生成  =====
            structured_graph = self.hypothesis_generator.generate_hypothesis(
                variable_list=self.variable_list,
                domain_name=self.domain_name,
                domain_context=self.domain_context,
                previous_graph=previous_graph,
                memory=memory,
                iteration=t,
                num_edge_operations=3
            )
            if structured_graph is None:
                continue
            
            if verbose:
                self.hypothesis_generator.visualize_graph(structured_graph)
            
            # 保存假设
            graph_path = os.path.join(self.output_dir, f"graph_t{t}.json")
            with open(graph_path, 'w') as f:
                json.dump(structured_graph, f, indent=2)
            
            # ===== 步骤2: 使用标准 BIC 评分 =====
            fitting_results = score_graph_with_bic(
                structured_graph=structured_graph,
                data=self.data,
                variable_names=self.variable_list
            )
            
            # 将评分结果添加到图的元数据
            structured_graph['metadata']['log_likelihood'] = fitting_results['cv_log_likelihood']
            structured_graph['metadata']['bic'] = fitting_results['bic']
            structured_graph['metadata']['num_parameters'] = fitting_results['num_parameters']
            structured_graph['metadata']['method'] = fitting_results['method']
            
            # ===== 步骤3: 后处理 - 生成记忆 =====
            # memory = self.post_processor.generate_memory(
            #     current_graph=structured_graph,
            #     current_results=fitting_results,
            #     previous_graph=previous_graph,
            #     previous_results=previous_results,
            #     domain_name=self.domain_name,
            #     model=llm_model_name,
            #     temperature=temperature,
            #     max_tokens=max_tokens
            # )
            
            # print("\n" + "-"*70)
            # print("MEMORY (μ_t):")
            # print("-"*70)
            # print(memory)
            # print("-"*70)
            
            # # 保存记忆
            # memory_path = os.path.join(self.output_dir, f"memory_t{t}.txt")
            # self.post_processor.save_memory(memory, memory_path)
            
            # ===== 记录历史 =====
            self.iteration_history.append({
                'iteration': t,
                'graph': structured_graph,
                'results': fitting_results,
                # 'memory': memory,
                'metrics':compute_metrics(self, structured_graph)
            })
            
            # ===== 更新前一轮的信息 =====
            previous_graph = structured_graph
            previous_results = fitting_results
            
            # ===== 评估与真实图的差距(如果有) =====
            if self.dataset and verbose:
                self._evaluate_against_ground_truth(structured_graph)
            
            # ===== 提前终止检查 =====
            if t > 0:
                ll_change = (fitting_results['cv_log_likelihood'] - 
                           self.iteration_history[t-1]['results']['cv_log_likelihood'])
                
                if abs(ll_change) < 0.01:
                    print(f"\n⚠️  Convergence detected (ΔLL={ll_change:.4f}). Stopping early.")
                    break
            t += 1
        
        # ===== 生成最终报告 =====
        final_report = self._generate_final_report()
        
        report_path = os.path.join(self.output_dir, "final_report.txt")
        with open(report_path, 'w') as f:
            f.write(final_report)
        
        print("\n" + "="*70)
        print("CMA PIPELINE COMPLETED")
        print("="*70)
        print(final_report)
        print("="*70 + "\n")
        
        return self.iteration_history[-1] if len(self.iteration_history[-1])!= 0 else None
    
    def _evaluate_against_ground_truth(self, predicted_graph: Dict):
        """评估预测图与真实图的差距"""
        
        # 提取预测的边
        predicted_edges = set()
        for node in predicted_graph['nodes']:
            child = node['name']
            for parent in node.get('parents', []):
                parent_idx = self.variable_list.index(parent)
                child_idx = self.variable_list.index(child)
                predicted_edges.add((parent_idx, child_idx))
        
        # 提取真实的边
        true_edges = set()
        for i in range(self.dataset.n_variables):
            for j in range(self.dataset.n_variables):
                if self.dataset.ground_truth_graph[i, j] == 1:
                    true_edges.add((i, j))
        
        # 计算指标
        true_positive = len(predicted_edges & true_edges)
        false_positive = len(predicted_edges - true_edges)
        false_negative = len(true_edges - predicted_edges)
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "-"*70)
        print("EVALUATION AGAINST GROUND TRUTH:")
        print("-"*70)
        print(f"True Positive (correct edges): {true_positive}")
        print(f"False Positive (incorrect edges): {false_positive}")
        print(f"False Negative (missing edges): {false_negative}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        if false_positive > 0:
            print(f"\nIncorrect edges added:")
            for parent_idx, child_idx in (predicted_edges - true_edges):
                print(f"  {self.variable_list[parent_idx]} → {self.variable_list[child_idx]}")
        
        if false_negative > 0:
            print(f"\nMissing edges:")
            for parent_idx, child_idx in (true_edges - predicted_edges):
                print(f"  {self.variable_list[parent_idx]} → {self.variable_list[child_idx]}")
        
        print("-"*70)
    
    def _generate_final_report(self) -> str:
        """生成最终报告"""
        
        lines = [
            f"CMA Final Report: {self.domain_name}",
            "="*70,
        ]
        
        # 数据集信息
        if self.dataset:
            lines.extend([
                f"\nDataset Information:",
                f"  Variables: {self.dataset.n_variables}",
                f"  Samples used: {len(self.data)}",
                f"  Ground truth edges: {self.dataset.ground_truth_graph.sum()}",
            ])
        else:
            lines.extend([
                f"\nVariables: {len(self.variable_list)}",
                f"Data samples: {self.data.shape[0]}",
            ])
        
        lines.extend([
            f"\nTotal iterations: {len(self.iteration_history)}",
            "\n" + "-"*70,
            "Iteration Summary:",
            "-"*70
        ])
        
        for record in self.iteration_history:
            t = record['iteration']
            ll = record['results']['cv_log_likelihood']
            edges = record['graph']['metadata']['num_edges']
            lines.append(f"  t={t}: LL={ll:.4f}, Edges={edges}")
        
        # 最佳迭代
        if len(self.iteration_history) > 0:
            best_idx = max(range(len(self.iteration_history)), 
                        key=lambda i: self.iteration_history[i]['results']['cv_log_likelihood'])
            best_ll = self.iteration_history[best_idx]['results']['cv_log_likelihood']
        
            lines.extend([
                "\n" + "-"*70,
                f"Best iteration: t={best_idx} (LL={best_ll:.4f})",
                "-"*70,
                "\nFinal Causal Structure:"
            ])
            
            final_graph = self.iteration_history[-1]['graph']
            for node in final_graph['nodes']:
                parents = node.get('parents', [])
                if parents:
                    for parent in parents:
                        lines.append(f"  {parent} → {node['name']}")
                else:
                    lines.append(f"  {node['name']} (root)")
        
        # 如果有真实图,添加对比
        if self.dataset:
            lines.extend([
                "\n" + "-"*70,
                "Ground Truth Structure:"
            ])
            for edge in self.dataset.get_ground_truth_edges():
                lines.append(f"  {edge[0]} → {edge[1]}")
        
        return "\n".join(lines)
    
    def get_best_graph(self) -> Dict:
        """返回拟合度最好的图"""
        best_idx = max(range(len(self.iteration_history)),
                       key=lambda i: self.iteration_history[i]['results']['cv_log_likelihood'])
        return self.iteration_history[best_idx]['graph']
    