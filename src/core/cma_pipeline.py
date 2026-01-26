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

from utils import ConfigManager, visualize_causal_graph, visualize_graph
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
        memory = None

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
        # 初始图评分
        fitting_results = score_graph_with_bic(
            structured_graph=initial_graph,
            data=self.data,
            variable_names=self.variable_list
        )
        initial_graph.metadata.log_likelihood = fitting_results['cv_log_likelihood']
        initial_graph.metadata.bic = fitting_results['bic']
        initial_graph.metadata.num_parameters = fitting_results['num_parameters']

        self.searcher: SearchStrategy = SearcherFactory.create_searcher(
            strategy_name=strategy_name,
            initial_graph=initial_graph
        )

        print(f"Initial hypothesis graph generated. {strategy_name} search strategy initialized.")
        print(f"Initial Graph Score - LL: {fitting_results['cv_log_likelihood']:.4f}, BIC: {fitting_results['bic']:.4f}")

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
            
            # 生成新假设图列表
            new_graphs, is_final_graph = self.hypothesis_generator.generate_next_hypothesis(
                variable_list=self.variable_list,
                domain_name=self.domain_name,
                domain_context=self.domain_context,
                previous_graph=current_graph,
                memory=memory,
                iteration=t,
                num_edge_operations=self.config.get("training.num_edge_operations")
            )

            # 如果 LLM 认为当前图已经足够好，标记并跳过
            if is_final_graph:
                print(f"Iteration {t}: LLM indicates current graph is final. Marking and skipping modification.")
                self.searcher.mark_as_final()
                continue

            if not new_graphs:
                print(f"Iteration {t}: No new hypothesis generated.")
                continue

            # 对每个候选图进行可视化和评分
            for idx, new_graph in enumerate(new_graphs, 1):
                print(f"\n  Candidate {idx}/{len(new_graphs)}:")
                visualize_causal_graph(new_graph, filename=f"iteration_{t}-{idx}.html")

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

                print(f"    Score - LL: {fitting_results['cv_log_likelihood']:.4f}, BIC: {fitting_results['bic']:.4f}")
                # 只有评分提升的图才加入搜索器
                if previous_graph is not None and previous_graph.metadata.log_likelihood is not None:
                    if new_graph.metadata.log_likelihood < previous_graph.metadata.log_likelihood:
                        print("    (Rejected: LL did not improve over previous graph)")
                        new_graphs.remove(new_graph)

            # 将新图列表加入搜索器
            self.searcher.update(new_graphs)

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

        res_lins = ""
        res_lins += "\n" + "-"*70
        res_lins += "EVALUATION AGAINST GROUND TRUTH:\n"
        res_lins += "-"*70 + "\n"
        res_lins += f"True Positive (correct edges): {true_positive}\n"
        res_lins += f"False Positive (incorrect edges): {false_positive}\n"
        res_lins += f"False Negative (missing edges): {false_negative}\n"
        res_lins += f"Precision: {precision:.3f}\n"
        res_lins += f"Recall: {recall:.3f}\n"
        res_lins += f"F1 Score: {f1:.3f}\n"

        if false_positive > 0:
            res_lins += f"\nIncorrect edges added:\n"
            for parent_idx, child_idx in (predicted_edges - true_edges):
                res_lins += f"  {self.variable_list[parent_idx]} → {self.variable_list[child_idx]}\n"

        if false_negative > 0:
            res_lins += f"\nMissing edges:\n"
            for parent_idx, child_idx in (true_edges - predicted_edges):
                res_lins += f"  {self.variable_list[parent_idx]} → {self.variable_list[child_idx]}\n"

        return res_lins

    def _generate_final_report(self) -> str:
        """生成最终报告"""

        lines = [
            f"CMA Final Report: {self.domain_name}",
            "="*70,
        ]

        # 数据集信息
        lines.extend([
            f"\nDataset Information:",
            f"  Variables: {self.dataset.n_variables}",
            f"  Samples used: {len(self.data)}",
            f"  Ground truth edges: {self.dataset.ground_truth_graph.sum()}",
        ])

        lines.extend([
            "-"*70,
            "Iteration Summary:",
        ])

        # 最佳迭代
        best_graph = self.searcher.best_graph()
        lines.extend([
            "\n" + "-"*70,
            "Best Found Structure:"
        ])
        for node in best_graph.nodes:
            parents = node.parents
            if parents:
                for parent in parents:
                    lines.append(f"  {parent} → {node.name}")
            else:
                lines.append(f"  {node.name} (root)")

        # 如果有真实图,添加对比
        lines.extend([
            "\n" + "-"*70,
            "Ground Truth Structure:"
        ])
        for edge in self.dataset.get_ground_truth_edges():
            lines.append(f"  {edge[0]} → {edge[1]}")
        if self.dataset.ground_truth_graph is not None:
            lines.append(self._evaluate_against_ground_truth(best_graph))
        

        return "\n".join(lines)
