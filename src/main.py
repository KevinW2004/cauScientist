"""项目主入口"""
import numpy as np
from argparse import ArgumentParser

from utils import ConfigManager
from data_loader import DataLoader
from core.cma_pipeline import CMAPipeline
from core.batch_runner import BatchExperimentRunner


def run_single_experiment():
    config = ConfigManager()
    dataset = None

    data_paths = {
        "fp_data": config.get("experiment.single.fp_data"),
        "fp_graph": config.get("experiment.single.fp_graph"),
    }
    if not all(data_paths.values()):
        raise ValueError("Missing data paths in config: experiment.single.fp_data/fp_graph")
    dataset = DataLoader.load_from_paths(**data_paths)

    pipeline = CMAPipeline(
        dataset=dataset
    )

    result = pipeline.run(verbose=True)
    best_graph = pipeline.get_best_graph()
    print(f"\n✅ Experiment completed!")
    print(f"Best graph at iteration {best_graph.metadata.iteration}")
    print(f"Log-likelihood: {best_graph.metadata.log_likelihood:.4f}\n")
    return result


def run_batch_experiment():
    config = ConfigManager()
    runner = BatchExperimentRunner(
        csv_config_path=config.get("experiment.csv_path"),
        base_output_dir=config.get("experiment.output.dir"),
        llm_type=config.get("llm.type"),
        llm_model_path=config.get("llm.local.model_path") if config.get("llm.type") == "local" else None,
        openai_base_url=config.get("llm.openai.base_url") if config.get("llm.type") == "openai" else None,
        openai_api_key=config.get("llm.openai.api_key") if config.get("llm.type") == "openai" else None,
    )
    return runner.run_all_experiments(
        split=config.get("experiment.split", "test"),
        num_iterations=config.get("training.num_iterations"),
        num_epochs=config.get("training.num_epochs"),
        device=config.get("llm.local.device"),
    )


# ========== 使用示例 ==========
if __name__ == "__main__":
    parser = ArgumentParser(description="Causal Scientist Main Entry Point")
    parser.add_argument("--config", type=str,
                        help="Configuration file name in /config", default=None)
    config_file = parser.parse_args().config
                        
    config = ConfigManager(config_file)
    
    # ========== 批量实验模式 ==========
    if config.get("experiment.mode") == "batch":
        run_batch_experiment()
    
    # ========== 单个实验模式 ==========
    else:
        run_single_experiment()