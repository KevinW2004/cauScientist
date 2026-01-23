"""
主运行流程
CMA Pipeline -CMA完整流程管理器
"""

import json
import numpy as np
import os
from typing import Dict, List, Optional

from core.llm_hypothesis import LLMHypothesisGenerator
from reflection.post_processing import PostProcessor
from data_loader import DOMAIN_CONTEXTS
from utils.score_functions import score_graph_with_bic

from utils import ConfigManager, visualize_causal_graph
from utils.metrics import compute_metrics
from llm_loader import LLMLoader, LLMLoaderFactory
from schemas import StructuredGraph, CausalDataset
from searcher import SearchStrategy, SearcherFactory

os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

class CMAPipeline:
    def __init__(
        self,
        dataset: CausalDataset
    ):
        """
        初始化CMA流程
        
        Args:
            dataset: CausalDataset对象
        """
        # 加载配置管理器
        self.config = ConfigManager()
        config = self.config

        # 设置输出目录
        self.output_dir = config.get("experiment.output.dir", "./cma_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置LLM类型
        self.llm_type = config.get("llm.type")

        # 加载数据集
        self.dataset = dataset
        self.domain_name = dataset.domain_name
        self.variable_list = dataset.variable_names
        self.data = dataset.data
        self.domain_context = DOMAIN_CONTEXTS.get(dataset.domain_name, "")

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
        """保存数据集信息"""
        assert self.dataset is not None, "Dataset must be provided to save dataset info."
        info = {
            "domain": self.dataset.domain_name,
            "n_variables": self.dataset.n_variables,
            "n_samples_total": self.dataset.n_samples,
            "n_samples_used": len(self.data),
            "variable_names": self.dataset.variable_names,
            "ground_truth_edges": self.dataset.get_ground_truth_edges(),
        }

        info_path = os.path.join(self.output_dir, f"{self.domain_name}_dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"[Info] Dataset info saved to {info_path}")

    def run(
        self,
        verbose: bool = True
    ) -> None:
        """运行完整的CMA流程"""
        num_iterations = self.config.get("training.num_iterations")

        print("\n" + "="*70)
        print(f"STARTING CMA PIPELINE: {self.domain_name.upper()}")
        print("="*70)

        # 打印数据集摘要
        self.dataset.print_summary()

        print(f"LLM Type: {self.llm_type}")
        print(f"Iterations: {num_iterations}")
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")

        previous_graph = None
        previous_results = None
        memory = None
        best = None

        # 1. 生成初始图，并创建搜索策略
        print("🔄 " * 35)
        print("GENERATING INITIAL HYPOTHESIS GRAPH & SEARCH STRATEGY")
        print("🔄 " * 35)
        strategy_name = self.config.get("strategy", "linear")

        initial_graph = self.hypothesis_generator.generate_initial_hypothesis(
            variable_list=self.variable_list,
            domain_name=self.domain_name,
            domain_context=self.domain_context
        )
        if initial_graph is None:
            print("Error: Failed to generate initial hypothesis graph.")
            return
        visualize_causal_graph(initial_graph)

        self.searcher: SearchStrategy = SearcherFactory.create_searcher(
            strategy_name=strategy_name,
            initial_graph=initial_graph
        )

        print(f"Initial hypothesis graph generated. {strategy_name} search strategy initialized.")

        # 2. 循环：
        #   获取需要修改图（由searcher提供）；
        #   如果 metadata 中 is_final_graph 为 True，则 continue;
        #   使用 hypothesis_generator 生成新假设图（列表）；
        #   使用 score_functions 评分新假设图；
        #   将评分上升的图加入 searcher;
        for t in range(1, num_iterations+1):
            print("\n" + "🔄 "*35)
            print(f"ITERATION {t}")
            print("🔄 "*35)

            # 获取需要修改的图
            current_graph = self.searcher.search()
            if current_graph.metadata.is_final_graph:
                print(f"Iteration {t}: Graph marked as final by LLM. Skipping modification.")
                continue

            # 生成新假设图
            new_graph = self.hypothesis_generator.generate_next_hypothesis(
                variable_list=self.variable_list,
                domain_name=self.domain_name,
                domain_context=self.domain_context,
                previous_graph=current_graph,
                memory=memory,
                iteration=t,
                num_edge_operations=self.config.get("training.num_edge_operations")
            )

            if new_graph is None:
                print(f"Iteration {t}: No new hypothesis generated.")
                continue

            visualize_causal_graph(new_graph)

            # 评分新图
            fitting_results = score_graph_with_bic(
                structured_graph=new_graph,
                data=self.data,
                variable_names=self.variable_list
            )

            # 将评分结果添加到图的元数据
            new_graph.metadata.log_likelihood = fitting_results['cv_log_likelihood']
            new_graph.metadata.bic = fitting_results['bic']
            new_graph.metadata.num_parameters = fitting_results['num_parameters']

            # 将新图和评分结果加入搜索器
            self.searcher.update([new_graph])

        # ===== 3. 生成最终报告 =====
        final_report = self._generate_final_report()

        report_path = os.path.join(self.output_dir, "final_report.txt")
        with open(report_path, 'w') as f:
            f.write(final_report)

        print("\n" + "="*70)
        print("CMA PIPELINE COMPLETED")
        print("="*70)
        print(final_report)
        print("="*70 + "\n")

        # for i in range(num_iterations):
        #     print("\n" + "🔄 "*35)
        #     print(f"ITERATION {t}")
        #     print("🔄 "*35)

        #     # ===== 步骤1: 假设生成  =====
        #     structured_graph = self.hypothesis_generator.generate_hypothesis(
        #         variable_list=self.variable_list,
        #         domain_name=self.domain_name,
        #         domain_context=self.domain_context,
        #         previous_graph=previous_graph,
        #         memory=memory,
        #         iteration=t,
        #         num_edge_operations=3
        #     )
        #     if structured_graph is None:
        #         continue

        #     if verbose:
        #         self.hypothesis_generator.visualize_graph(structured_graph)

        #     # 保存假设（使用 Pydantic 的 model_dump 转为字典再序列化）
        #     graph_path = os.path.join(self.output_dir, f"graph_t{t}.json")
        #     with open(graph_path, 'w') as f:
        #         json.dump(structured_graph.model_dump(mode='python'), f, indent=2)

        #     # ===== 步骤2: 使用标准 BIC 评分 =====
        #     fitting_results = score_graph_with_bic(
        #         structured_graph=structured_graph,
        #         data=self.data,
        #         variable_names=self.variable_list
        #     )

        #     # 将评分结果添加到图的元数据
        #     structured_graph.metadata.log_likelihood = fitting_results['cv_log_likelihood']
        #     structured_graph.metadata.bic = fitting_results['bic']
        #     structured_graph.metadata.num_parameters = fitting_results['num_parameters']

        #     # ===== 记录历史 =====
        #     self.iteration_history.append({
        #         'iteration': t,
        #         'graph': structured_graph,
        #         'results': fitting_results,
        #         # 'memory': memory,
        #         'metrics':compute_metrics(self, structured_graph)
        #     })

        #     # ===== 更新前一轮的信息 =====
        #     previous_graph = structured_graph
        #     previous_results = fitting_results

        #     # ===== 评估与真实图的差距(如果有) =====
        #     if self.dataset and verbose:
        #         self._evaluate_against_ground_truth(structured_graph)

        #     # ===== 提前终止检查 =====
        #     if t > 0:
        #         ll_change = (fitting_results['cv_log_likelihood'] -
        #                    self.iteration_history[t-1]['results']['cv_log_likelihood'])

        #         if abs(ll_change) < 0.01:
        #             print(f"\n⚠️  Convergence detected (ΔLL={ll_change:.4f}). Stopping early.")
        #             break
        #     t += 1

        # # ===== 生成最终报告 =====
        # final_report = self._generate_final_report()

        # report_path = os.path.join(self.output_dir, "final_report.txt")
        # with open(report_path, 'w') as f:
        #     f.write(final_report)

        # print("\n" + "="*70)
        # print("CMA PIPELINE COMPLETED")
        # print("="*70)
        # print(final_report)
        # print("="*70 + "\n")

        # return self.iteration_history[-1] if len(self.iteration_history) > 0 else None

    def _evaluate_against_ground_truth(self, predicted_graph: StructuredGraph):
        """评估预测图与真实图的差距"""

        assert self.dataset is not None, "Dataset must be provided to evaluate against ground truth"

        # 提取预测的边
        predicted_edges = set()
        for node in predicted_graph.nodes:
            child = node.name
            for parent in node.parents:
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
            graph: StructuredGraph = record['graph']
            edges = graph.metadata.num_edges
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

            final_graph: StructuredGraph = self.iteration_history[-1]['graph']
            for node in final_graph.nodes:
                parents = node.parents
                if parents:
                    for parent in parents:
                        lines.append(f"  {parent} → {node.name}")
                else:
                    lines.append(f"  {node.name} (root)")

        # 如果有真实图,添加对比
        if self.dataset:
            lines.extend([
                "\n" + "-"*70,
                "Ground Truth Structure:"
            ])
            for edge in self.dataset.get_ground_truth_edges():
                lines.append(f"  {edge[0]} → {edge[1]}")

        return "\n".join(lines)
