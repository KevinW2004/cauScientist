"""项目主入口"""
import numpy as np

from utils import ConfigManager
from data_loader import DataLoader
from scripts.cma_pipeline import CMAPipeline
from scripts.batch_runner import BatchExperimentRunner


def run_single_experiment():
    config = ConfigManager()
    use_synthetic = config.get("experiment.single.use_synthetic_data")
    domain_name = config.get("experiment.single.domain_name")
    variable_list = config.get("experiment.single.variable_list")

    dataset = None
    if not use_synthetic:
        data_paths = {
            "fp_data": config.get("experiment.single.data.fp_data"),
            "fp_graph": config.get("experiment.single.data.fp_graph"),
            "fp_regime": config.get("experiment.single.data.fp_regime"),
        }
        if not all(data_paths.values()):
            raise ValueError("Missing data paths in config: experiment.single.data.fp_data/fp_graph/fp_regime")
        dataset = DataLoader.load_from_paths(**data_paths)

    if use_synthetic:
        num_samples = config.get("experiment.single.num_samples")
        np.random.seed(42)
        synthetic_data = np.random.randn(num_samples, len(variable_list)).astype(np.float32)
        pipeline = CMAPipeline(
            domain_name=domain_name,
            variable_list=variable_list,
            data=synthetic_data,
            domain_context=config.get("experiment.single.domain_context", "Synthetic data experiment")
        )
    else:
        pipeline = CMAPipeline(
            dataset=dataset
        )

    result = pipeline.run(verbose=True)
    best_graph = pipeline.get_best_graph()
    print(f"\n✅ Experiment completed!")
    print(f"Best graph at iteration {best_graph['metadata']['iteration']}")
    print(f"Log-likelihood: {best_graph['metadata']['log_likelihood']:.4f}\n")
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
    config = ConfigManager()
    
    # ========== 批量实验模式 ==========
    if config.get("experiment.mode") == "batch":
        run_batch_experiment()
    
    # ========== 单个实验模式 ==========
    else:
        run_single_experiment()