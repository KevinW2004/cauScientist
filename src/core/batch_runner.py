"""
还是原始版本，没有修改适配
"""
import os
import json

from .cma_pipeline import CMAPipeline
from utils.metrics import compute_metrics
from data_loader import DataLoader

# ========== 批量实验运行器 ==========
class BatchExperimentRunner:
    """批量运行CMA实验"""
    
    def __init__(
        self,
        csv_config_path: str,
        base_output_dir: str = "./cma_experiments",
        # LLM配置
        llm_type: str = "openai",
        llm_model_path: str = None,
        openai_base_url: str = None,
        openai_api_key: str = None
    ):
        self.csv_config_path = csv_config_path
        self.base_output_dir = base_output_dir
        self.llm_type = llm_type
        self.llm_model_path = llm_model_path
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        
        os.makedirs(base_output_dir, exist_ok=True)
    
    def run_all_experiments(
        self,
        split: str = "test",
        num_iterations: int = 3,
        num_epochs: int = 50,
        device: str = "cpu",  # 新增 device 参数
        **kwargs
    ):
        """运行所有实验"""
        
        # 加载所有数据集
        print(f"Loading datasets from {self.csv_config_path}...")
        datasets = DataLoader.load_all_from_csv(self.csv_config_path, split=split)
        
        print(f"\n{'='*70}")
        print(f"BATCH EXPERIMENT: {len(datasets)} datasets loaded")
        print(f"LLM Type: {self.llm_type}")
        if self.llm_type == "local":
            print(f"Model Path: {self.llm_model_path}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        
        results_summary = []
        
        for idx, dataset in enumerate(datasets):
            print(f"\n{'#'*70}")
            print(f"EXPERIMENT {idx+1}/{len(datasets)}: {dataset.domain_name}")
            print(f"{'#'*70}\n")
            
            # 创建输出目录
            output_dir = os.path.join(self.base_output_dir, f"{idx:02d}_{dataset.domain_name}")
            
            # 运行CMA
            try:
                pipeline = CMAPipeline(
                    dataset=dataset,
                )
                
                final_result = pipeline.run(
                    **kwargs
                )
                
                # 提取评估指标(如果有ground truth)
                # final_graph = final_result['graph']
                best_graph = pipeline.get_best_graph()
                final_graph = best_graph
                metrics = compute_metrics(pipeline, final_graph)
                iteration_improvement_LL = [iteration_i['results']['cv_log_likelihood'] for iteration_i in pipeline.iteration_history]
                iteration_improvement_SHD = [iteration_i['metrics']['shd'] for iteration_i in pipeline.iteration_history]
                
                results_summary.append({
                    "experiment_id": idx,
                    "domain": dataset.domain_name,
                    "status": "success",
                    "final_ll": final_result['results']['cv_log_likelihood'] if final_result else None,
                    "num_edges_predicted": final_graph.metadata.num_edges,
                    "num_edges_true": int(dataset.ground_truth_graph.sum()),
                    "metrics": metrics,
                    "output_dir": output_dir,
                    "iteration_improvement_LL": iteration_improvement_LL,
                    "iteration_improvement_SHD": iteration_improvement_SHD
                })
                
                print(f"\n✅ Experiment {idx+1} completed successfully!")
                
            except Exception as e:
                print(f"\n❌ Experiment {idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                
                results_summary.append({
                    "experiment_id": idx,
                    "domain": dataset.domain_name,
                    "status": "failed",
                    "error": str(e),
                    "output_dir": output_dir
                })
                exit()
        
        # 保存总结
        summary_path = os.path.join(self.base_output_dir, "experiments_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 打印统计信息
        self._print_summary(results_summary)
        
        return results_summary
    
    def _print_summary(self, results_summary: list[dict]):
        """打印总结信息"""
        
        print(f"\n{'='*70}")
        print("BATCH EXPERIMENTS SUMMARY")
        print(f"{'='*70}\n")
        
        success_count = sum(1 for r in results_summary if r['status'] == 'success')
        failed_count = len(results_summary) - success_count
        
        print(f"Total experiments: {len(results_summary)}")
        print(f"  Success: {success_count}")
        print(f"  Failed: {failed_count}")
        
        if success_count > 0:
            print(f"\n{'-'*70}")
            print("Successful Experiments:")
            print(f"{'-'*70}")
            print(f"{'ID':<5} {'Domain':<15} {'iteration_improvement_SHD':<30} {'iteration_improvement_LL':<30} {'LL':<12} {'Edges':<10} {'F1':<8} {'Precision':<10} {'Recall':<8} {'SHD':<10}")
            print(f"{'-'*70}")
            
            for r in results_summary:
                if r['status'] == 'success':
                    metrics = r.get('metrics', {})
                    iteration_improvement_SHD_list = r['iteration_improvement_SHD']
                    iteration_improvement_SHD_str = '[' + ', '.join(f"{x:.1f}" for x in iteration_improvement_SHD_list) + ']'
                    iteration_improvement_LL_list = r['iteration_improvement_LL']
                    iteration_improvement_LL_str = '[' + ', '.join(f"{x:.1f}" for x in iteration_improvement_LL_list) + ']'
                    print(f"{r['experiment_id']:<5} "
                          f"{r['domain']:<15} "
                          f"{iteration_improvement_SHD_str:<30} "
                          f"{iteration_improvement_LL_str:<30} "
                          f"{r['final_ll']:<12.4f} "
                          f"{r['num_edges_predicted']}/{r['num_edges_true']:<8} "
                          f"{metrics.get('f1_score', 0):<8.4f} "
                          f"{metrics.get('precision', 0):<10.4f} "
                          f"{metrics.get('recall', 0):<8.4f}"
                          f"{metrics.get('shd', 0):<10.4f}"
                          )
        
        if failed_count > 0:
            print(f"\n{'-'*70}")
            print("Failed Experiments:")
            print(f"{'-'*70}")
            for r in results_summary:
                if r['status'] == 'failed':
                    print(f"  {r['experiment_id']}: {r['domain']} - {r['error'][:50]}")
        
        print(f"\n{'='*70}\n")
