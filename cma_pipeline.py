"""
CMA Pipeline - 完整版
同时支持:
1. 本地模型 / OpenAI API 切换
2. 数据加载功能
3. 真实图对比评估
"""

import json
import numpy as np
import os
from typing import Dict, List, Optional

from llm_hypothesis import LLMHypothesisGenerator
from model_fitting import ModelFittingEngine
from post_processing import PostProcessor
from data_loader import DataLoader, CausalDataset, DOMAIN_CONTEXTS
from skeleton_builder import SkeletonBuilder, _skeleton_to_graph_format
from transformers import AutoTokenizer
from metrics import _compute_metrics
from search_strategies import HillClimbingStrategy, MCTSStrategy
from baseline_reference import load_baseline_reference_from_predict
from intervention_utils import InterventionTester, EvidencePolicyVerifier

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
        
        # 骨架构建参数
        use_skeleton: bool = False,  # 是否使用MMHC骨架
        skeleton_alpha: float = 0.05,  # 骨架构建的显著性水平
        skeleton_max_cond_size: int = 3,  # 骨架构建的最大条件集大小
        
        # 基线参考参数
        use_baseline_reference: bool = False,  # 是否使用传统方法的参考信息
        baseline_predict_dir: str = "predict",  # predict 目录路径
        baseline_methods: List[str] = None,  # 要加载的基线方法 ['corr', 'invcov', 'notears']
        baseline_top_k: int = 10,  # 每个方法显示top-k个关系
        baseline_threshold: float = 0.5,  # 基线方法的阈值百分位
        choose_best: bool = False,  # 是否在初始阶段比较基线和LLM
        
        # 干预测试参数
        use_intervention_test: bool = False,  # 是否允许LLM主动发起干预实验
        num_intervention_experiments: int = 3, # 每轮允许的最大实验数
        
        # 经典算法增强参数
        use_notears_refinement: bool = False,  # 是否使用NOTEARS优化
        notears_use_mlp: bool = False,  # NOTEARS是否使用MLP作为score（推荐）
        notears_alpha: float = 0.001,  # NOTEARS L2正则化（仅Ridge版本）
        notears_threshold: float = 0.15,  # NOTEARS边权重阈值
        notears_poly_degree: int = 2,  # 多项式阶数（仅Ridge版本）
        notears_start_iter: int = 0,  # 从第几轮迭代开始使用NOTEARS
        use_greedy_refinement: bool = False,  # 是否使用贪心优化（推荐）
        greedy_max_modifications: int = 10,  # 贪心优化的最大修改次数
        greedy_min_improvement: float = 0.01,  # 贪心优化的最小LL改进阈值
        greedy_eval_epochs: int = 15,  # 贪心评估时的训练轮数（降低以加速）
        greedy_max_candidates: int = 30,  # 每种操作最多测试的候选数（加速）
        greedy_start_iter: int = 0,  # 从第几轮迭代开始使用贪心优化
        
        # MCTS参数
        mcts_simulations: int = 50,  # MCTS每次迭代的模拟次数
        mcts_exploration_weight: float = 1.414,  # MCTS UCB1探索权重
        mcts_max_depth: int = 5,  # MCTS最大搜索深度
        
        # 输出参数
        output_dir: str = "./cma_output",
        device: str = 'cpu',
        
        # LLM配置
        llm_type: str = "openai",  # "openai" 或 "local"
        llm_model_path: str = None,  # 本地模型路径
        openai_base_url: str = None,  # OpenAI API URL
        openai_api_key: str = None,   # OpenAI API key
        
        # 预加载的模型（用于批量实验复用）
        shared_tokenizer = None,
        shared_model = None
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
            
            # 骨架构建配置
            use_skeleton: 是否使用MMHC算法构建统计骨架作为约束
            skeleton_alpha: 独立性检验的显著性水平
            skeleton_max_cond_size: 条件集的最大大小
            
            # 输出配置
            output_dir: 输出目录
            device: 模型拟合设备
            
            # LLM配置
            llm_type: "openai" 或 "local"
            llm_model_path: 本地模型路径(llm_type="local"时必需)
            openai_base_url: OpenAI API base URL
            openai_api_key: OpenAI API key
        """
        
        # ===== 处理数据输入(两种方式) =====
        assert dataset is not None, "dataset is required"
        self.dataset = dataset
        self.domain_name = dataset.domain_name
        self.variable_list = dataset.variable_names
        self.variable_type = dataset.variable_type  # 存储变量类型 (continuous/discrete)
        
        # 使用全部数据，并保存干预信息以便在模型拟合时精确处理
        # 不再使用简单的"全有或全无"策略，而是在拟合时针对每个变量使用未被干预的样本
        self.data = dataset.data
        self.interventions = dataset.interventions  # 保存干预信息
        
        if dataset.interventions is not None:
            n_intervened = (dataset.interventions.sum(axis=1) > 0).sum()
            n_observational = len(dataset.data) - n_intervened
            print(f"[Info] Using all data with intervention-aware fitting:")
            print(f"  - Total samples: {len(dataset.data)}")
            print(f"  - Observational samples: {n_observational}")
            print(f"  - Samples with interventions: {n_intervened}")
            print(f"  - During fitting, each variable will only use samples where it was NOT intervened")
        else:
            print(f"[Info] Using data: {self.data.shape} (no interventions)")
        
        # 使用预定义的领域背景
        if not domain_context:
            self.domain_context = DOMAIN_CONTEXTS.get(dataset.domain_name, "")
        else:
            self.domain_context = domain_context
        
        self.output_dir = output_dir
        self.device = device
        self.llm_type = llm_type
        self.use_skeleton = use_skeleton
        self.skeleton_constraints = None
        
        # ===== 基线参考配置 =====
        self.use_baseline_reference = use_baseline_reference
        self.baseline_predict_dir = baseline_predict_dir
        self.baseline_methods = baseline_methods or ['corr', 'invcov']
        self.baseline_top_k = baseline_top_k
        self.baseline_threshold = baseline_threshold
        self.baseline_reference_text = None
        self.baseline_structured_graphs = {}  # 初始为空字典
        self.choose_best = choose_best
        
        # ===== 干预测试配置 =====
        self.use_intervention_test = use_intervention_test
        self.num_intervention_experiments = num_intervention_experiments
        self.accumulated_evidence = []
        self.intervention_tester = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据集信息
        if self.dataset:
            self._save_dataset_info()
        
        # ===== 构建统计骨架(如果启用) =====
        if use_skeleton:
            print("\n" + "="*70)
            print("BUILDING STATISTICAL SKELETON")
            print("="*70)
            
            # 获取变量类型
            variable_type = self.dataset.variable_type if self.dataset else "continuous"
            
            skeleton_builder = SkeletonBuilder(
                alpha=skeleton_alpha,
                max_cond_size=skeleton_max_cond_size,
                variable_type=variable_type
            )
            
            skeleton, pc_sets = skeleton_builder.build_skeleton(
                data=self.data,
                variable_names=self.variable_list,
                verbose=True
            )
            
            self.skeleton_constraints = skeleton_builder.skeleton_to_constraint(
                skeleton, self.variable_list
            )
            skeleton_graph = _skeleton_to_graph_format(skeleton, self.variable_list)
            skeleton_metrics = _compute_metrics(self, skeleton_graph)
            
            # 打印骨架质量
            print("\n" + "-"*70)
            print("SKELETON QUALITY METRICS:", skeleton_metrics)
            print("-"*70)
            
            # 保存骨架信息
            skeleton_info = {
                'skeleton_matrix': skeleton.tolist(),
                'pc_sets': {self.variable_list[k]: [self.variable_list[v] for v in vals] 
                           for k, vals in pc_sets.items()},
                'allowed_edges': self.skeleton_constraints['allowed_edges'],
                'forbidden_pairs': self.skeleton_constraints['forbidden_pairs'],
                'n_edges': int(skeleton.sum() // 2),
                'alpha': skeleton_alpha,
                'max_cond_size': skeleton_max_cond_size,
                'variable_type': variable_type,
                'metrics': skeleton_metrics  # 添加指标
            }
            
            skeleton_path = os.path.join(self.output_dir, "skeleton_info.json")
            with open(skeleton_path, 'w') as f:
                json.dump(skeleton_info, f, indent=2)
            
            print(f"\n✓ Skeleton saved to {skeleton_path}")
            print(f"  Allowed edges: {len(self.skeleton_constraints['allowed_edges'])}")
            print(f"  Forbidden pairs: {len(self.skeleton_constraints['forbidden_pairs'])}")
            print("="*70 + "\n")
            exit()
        
        # ===== 加载基线参考（如果启用）=====
        if self.use_baseline_reference:
            self.baseline_structured_graphs = load_baseline_reference_from_predict(
                dataset_name=self.domain_name,
                variable_list=self.variable_list,
                predict_dir=self.baseline_predict_dir,
                methods=self.baseline_methods,
                top_k=self.baseline_top_k,
                threshold=self.baseline_threshold
            )
            # 生成供LLM阅读的文本描述
            self.baseline_reference_text = self._format_baseline_reference_text()

        # ===== 初始化NOTEARS优化器（支持MLP score）=====
        self.use_notears_refinement = use_notears_refinement
        self.notears_start_iter = notears_start_iter
        self.notears_use_mlp = notears_use_mlp  # 使用传入的参数
        
        if self.use_notears_refinement:
            if self.notears_use_mlp:
                # 使用MLP-based NOTEARS（推荐）
                from classical_refinement import NOTEARSMLPRefiner
                self.notears_refiner = NOTEARSMLPRefiner(
                    w_threshold=notears_threshold,
                    max_iter=100,
                    h_tol=1e-8,
                    rho_max=1e+16,
                    w_lr=0.001,
                    hidden_dims=[10, 1],
                    device=self.device
                )
                print("\n" + "="*70)
                print("NOTEARS-MLP REFINEMENT ENABLED (Official Implementation)")
                print("="*70)
                print(f"Using continuous optimization with MLP")
                print(f"Weight threshold: {notears_threshold}")
                print(f"Max iterations: 100")
                print(f"Learning rate: 0.001")
                print(f"Hidden dims: [10, 1]")
                print(f"Start from iteration: {notears_start_iter}")
                print("="*70 + "\n")
            else:
                # 使用传统的多项式Ridge回归NOTEARS
                from classical_refinement import NOTEARSRefiner
                self.notears_refiner = NOTEARSRefiner(
                    alpha=notears_alpha,
                    w_threshold=notears_threshold,
                    max_iter=50,
                    poly_degree=notears_poly_degree
                )
                print("\n" + "="*70)
                print("NOTEARS REFINEMENT ENABLED (Polynomial Ridge Regression)")
                print("="*70)
                print(f"Alpha (L2 regularization): {notears_alpha}")
                print(f"Threshold: {notears_threshold}")
                print(f"Polynomial degree: {notears_poly_degree} ({'linear' if notears_poly_degree==1 else 'nonlinear'})")
                print(f"Start from iteration: {notears_start_iter}")
                print("="*70 + "\n")
        
        # ===== 初始化贪心优化器（推荐）=====
        self.use_greedy_refinement = use_greedy_refinement
        self.greedy_start_iter = greedy_start_iter
        if self.use_greedy_refinement:
            from greedy_refinement import GreedyGraphRefiner
            self.greedy_refiner = GreedyGraphRefiner(
                max_modifications=greedy_max_modifications,
                min_improvement=greedy_min_improvement,
                eval_epochs=greedy_eval_epochs,
                max_candidates_per_type=greedy_max_candidates,
                allow_add=True,
                allow_delete=True,
                allow_reverse=True
            )
            print("\n" + "="*70)
            print("GREEDY GRAPH REFINEMENT ENABLED (MLP-based)")
            print("="*70)
            print(f"Max modifications per iteration: {greedy_max_modifications}")
            print(f"Min LL improvement threshold: {greedy_min_improvement}")
            print(f"Evaluation epochs: {greedy_eval_epochs}")
            print(f"Max candidates per operation type: {greedy_max_candidates}")
            print(f"Start from iteration: {greedy_start_iter}")
            print("="*70 + "\n")
        
        # ===== 存储MCTS参数 =====
        self.mcts_simulations = mcts_simulations
        self.mcts_exploration_weight = mcts_exploration_weight
        self.mcts_max_depth = mcts_max_depth
        
        if llm_type == "local":
            # 检查是否提供了预加载的模型
            assert shared_model is not None, "shared_model must be provided when llm_type='local'"
            self.tokenizer = shared_tokenizer
            self.model = shared_model
            
            # 使用共享的模型初始化各模块
            self.hypothesis_generator = LLMHypothesisGenerator(
                model_type="local",
                shared_tokenizer=self.tokenizer,
                shared_model=self.model
            )
            
            self.post_processor = PostProcessor(
                model_type="local",
                tokenizer=self.tokenizer,
                model=self.model
            )
            
        else:  # openai
            print(f"Using OpenAI-compatible API")
            if openai_base_url:
                print(f"Base URL: {openai_base_url}")
            
            # OpenAI不需要预加载模型
            self.tokenizer = None
            self.model = None
            
            self.hypothesis_generator = LLMHypothesisGenerator(
                model_type="openai",
                base_url=openai_base_url,
                api_key=openai_api_key
            )
            
            self.post_processor = PostProcessor(
                model_type="openai",
                base_url=openai_base_url,
                api_key=openai_api_key
            )
        
        # 模型拟合引擎（不需要LLM）
        self.fitting_engine = ModelFittingEngine(device=device)
        
        # ===== 干预测试引擎 =====
        # 注意：已经在 __init__ 前期初始化了 self.use_intervention_test 和 self.accumulated_evidence
        self.num_intervention_experiments = num_intervention_experiments
        self.policy_verifier = EvidencePolicyVerifier() # 始终初始化，用于验证 LLM 的决策
        if self.dataset.interventions is not None:
            self.intervention_tester = InterventionTester(self.dataset)
        else:
            self.use_intervention_test = False
            self.intervention_tester = None
            if use_intervention_test:
                print("[Warning] Intervention test requested but no intervention data found. Disabling.")
        
        # 存储历史
        self.iteration_history = []
        
        print("✓ Pipeline initialized successfully!")
        print("="*70 + "\n")

    def _format_baseline_reference_text(self) -> str:
        """将结构化基线图转换为供LLM阅读的文本描述"""
        if not self.baseline_structured_graphs:
            return ""
            
        text = "\n" + "="*50 + "\n"
        text += "📊 STATISTICAL BASELINE REFERENCE (from traditional methods):\n"
        text += "="*50 + "\n"
        text += "The following causal relationships were suggested by traditional algorithms.\n"
        text += "Use them as hints, but rely on your domain knowledge and interventional evidence.\n\n"
        
        for method, graph in self.baseline_structured_graphs.items():
            method_name = method.upper()
            text += f"[{method_name} Algorithm]:\n"
            edges = []
            for node in graph['nodes']:
                for parent in node.get('parents', []):
                    edges.append(f"  - {parent} → {node['name']}")
            
            if edges:
                text += "\n".join(edges) + "\n"
            else:
                text += "  (no edges predicted)\n"
            text += "\n"
            
        return text
    
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
        # with open(info_path, 'w') as f:
        #     json.dump(info, f, indent=2)
        # print(f"[Info] Dataset info saved to {info_path}") # TODO: currently not used
    
    def run(
        self,
        num_iterations: Optional[int] = None,
        iterations_per_node: float = 3.0,  # 每节点建议迭代次数
        early_stopping_patience: int = 3,   # 连续多少次改进失败则停止
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        temperature: float = 0.6,
        llm_model_name: str = "gpt-4o",
        max_tokens: int = 4096,
        verbose: bool = True,
        use_hill_climbing: bool = True,  # 保持原有参数（向后兼容）
        use_mcts: bool = False,  # MCTS策略开关
        acceptance_tolerance: float = 0.0,
        max_retries: int = 10,
        use_local_amendment: bool = True,
        llm_only: bool = False,
        choose_best: bool = False,
        use_intervention_test: Optional[bool] = None  # 新增
    ) -> Dict:
        """运行完整的CMA流程（内部已重构，支持策略扩展）"""
        
        if use_intervention_test is not None:
            self.use_intervention_test = use_intervention_test
        
        # 自动计算迭代次数
        if num_iterations is None:
            num_iterations = int(len(self.variable_list) * iterations_per_node)
            print(f"[Info] num_iterations is None, auto-calculated: {len(self.variable_list)} nodes * {iterations_per_node} = {num_iterations}")
        
        # 内部转换：根据参数选择策略
        if use_mcts:
            search_strategy = "mcts"
        else:
            search_strategy = "hill_climbing"
        
        print("\n" + "="*70)
        print(f"STARTING CMA PIPELINE: {self.domain_name.upper()}")
        print(f"Total iterations: {num_iterations}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Choose best initial (Baseline vs LLM): {choose_best}")
        print("="*70)
        
        if search_strategy == "hill_climbing":
            strategy = HillClimbingStrategy(
                pipeline=self,
                acceptance_tolerance=acceptance_tolerance
            )
        elif search_strategy == "mcts":
            strategy = MCTSStrategy(
                pipeline=self,
                num_simulations=self.mcts_simulations,
                exploration_weight=self.mcts_exploration_weight,
                max_depth=self.mcts_max_depth
            )
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
        
        # ===== 执行搜索 =====
        results = strategy.search(
            num_iterations=num_iterations,
            early_stopping_patience=early_stopping_patience,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            llm_model_name=llm_model_name,
            max_tokens=max_tokens,
            verbose=verbose,
            max_retries=max_retries,
            use_local_amendment=use_local_amendment,
            llm_only=llm_only,
            choose_best=choose_best
        )
        
        # ===== 更新历史记录 =====
        self.iteration_history = strategy.iteration_history

    def get_best_graph(self) -> Dict:
        """返回拟合度最好的图（支持爬山策略）"""
        if not self.iteration_history:
            return None, float('-inf')
        
        best_idx = max(range(len(self.iteration_history)),
                       key=lambda i: self.iteration_history[i]['results']['log_likelihood'])
        return self.iteration_history[best_idx]['graph'], self.iteration_history[best_idx]['results']['log_likelihood']
    
    def get_accepted_graphs(self) -> List[Dict]:
        """返回所有被接受的图（仅爬山策略有效）"""
        return [h for h in self.iteration_history if h.get('accepted', True)]


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
        
        # 预加载的模型（用于批量实验复用）
        self.shared_tokenizer = None
        self.shared_model = None
        
        os.makedirs(base_output_dir, exist_ok=True)
    
    def _preload_local_model(self):
        """预加载本地模型（只加载一次，供所有实验复用）"""
        from vllm import LLM
        from transformers import AutoTokenizer
        
        print(f"Loading local model from {self.llm_model_path}... This may take a few minutes...")
        
        # 使用 vLLM 加载模型
        self.shared_model = LLM(
            model=self.llm_model_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        
        # 加载tokenizer
        self.shared_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
        
        print(f"\n✓ Model pre-loaded successfully!")
        print("="*70 + "\n")
    
    def run_all_experiments(
        self,
        split: str = "test",
        num_runs: int = 1,  # 新增: 每个数据集跑多少次以进行显著性测试
        num_iterations: Optional[int] = None,
        iterations_per_node: float = 3.0,
        early_stopping_patience: int = 3,
        num_epochs: int = 50,
        device: str = "cpu",
        use_skeleton: bool = False,
        skeleton_alpha: float = 0.05,
        skeleton_max_cond_size: int = 3,
        use_notears_refinement: bool = False,  # NOTEARS优化
        notears_use_mlp: bool = False,  # NOTEARS是否使用MLP作为score
        notears_alpha: float = 0.001,  # L2正则化（仅Ridge版本）
        notears_threshold: float = 0.15,  # 边权重阈值
        notears_poly_degree: int = 2,  # 多项式阶数（仅Ridge版本）
        notears_start_iter: int = 0,
        use_greedy_refinement: bool = False,  # 贪心优化（推荐）
        greedy_max_modifications: int = 10,
        greedy_min_improvement: float = 0.01,
        greedy_eval_epochs: int = 15,
        greedy_max_candidates: int = 30,
        greedy_start_iter: int = 0,
        mcts_simulations: int = 50,  # MCTS参数
        mcts_exploration_weight: float = 1.414,
        mcts_max_depth: int = 5,
        llm_only: bool = False,
        choose_best: bool = False,
        use_intervention_test: bool = False,
        num_intervention_experiments: int = 3,
        **kwargs
    ):
        """运行所有实验"""
        
        # 加载所有数据集
        print(f"Loading datasets from {self.csv_config_path}...")
        datasets = DataLoader.load_all_from_csv(self.csv_config_path, split=split)
        
        print(f"\n{'='*70}")
        print(f"BATCH EXPERIMENT: {len(datasets)} datasets loaded")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        
        # 预加载本地模型（如果使用local模式）
        if self.llm_type == "local":
            self._preload_local_model()
        
        results_summary = []
        
        for idx, dataset in enumerate(datasets):
            print(f"\n{'#'*70}")
            print(f"EXPERIMENT {idx+1}/{len(datasets)}: {dataset.domain_name}")
            print(f"Running {num_runs} independent trials for significance testing")
            print(f"{'#'*70}\n")
            
            # 存储该数据集的所有运行结果
            dataset_run_results = []
            
            for run_idx in range(num_runs):
                print(f"\n>>> Trial {run_idx+1}/{num_runs}")
                
                # 为每个运行创建子目录
                trial_output_dir = os.path.join(self.base_output_dir, f"{idx:02d}_{dataset.domain_name}", f"run_{run_idx:02d}")
                os.makedirs(trial_output_dir, exist_ok=True)
                
                # 保存完整配置
                config_path = os.path.join(trial_output_dir, "config.json")
                # 收集所有配置
                full_config = {
                    "base_output_dir": self.base_output_dir,
                    "llm_type": self.llm_type,
                    "llm_model_path": self.llm_model_path,
                    "dataset": dataset.domain_name,
                    "run_idx": run_idx,
                    "split": split,
                    "num_runs": num_runs,
                    "num_iterations": num_iterations,
                    "iterations_per_node": iterations_per_node,
                    "early_stopping_patience": early_stopping_patience,
                    "num_epochs": num_epochs,
                    "device": device,
                    "use_skeleton": use_skeleton,
                    "skeleton_alpha": skeleton_alpha,
                    "skeleton_max_cond_size": skeleton_max_cond_size,
                    "use_notears_refinement": use_notears_refinement,
                    "notears_use_mlp": notears_use_mlp,
                    "use_greedy_refinement": use_greedy_refinement,
                    "greedy_max_modifications": greedy_max_modifications,
                    "greedy_min_improvement": greedy_min_improvement,
                    "greedy_eval_epochs": greedy_eval_epochs,
                    "greedy_max_candidates": greedy_max_candidates,
                    "greedy_start_iter": greedy_start_iter,
                    "mcts_simulations": mcts_simulations,
                    "mcts_exploration_weight": mcts_exploration_weight,
                    "mcts_max_depth": mcts_max_depth,
                    "choose_best": choose_best,
                    "use_intervention_test": use_intervention_test,
                    "num_intervention_experiments": num_intervention_experiments,
                    **kwargs
                }
                with open(config_path, 'w') as f:
                    json.dump(full_config, f, indent=2, ensure_ascii=False)
                
                # 运行CMA
                try:
                    pipeline = CMAPipeline(
                        dataset=dataset,
                        output_dir=trial_output_dir,
                        use_observational_only=True,
                        device=device,
                        use_skeleton=use_skeleton,
                        skeleton_alpha=skeleton_alpha,
                        skeleton_max_cond_size=skeleton_max_cond_size,
                        use_notears_refinement=use_notears_refinement,  # NOTEARS参数
                        notears_use_mlp=notears_use_mlp,
                        notears_alpha=notears_alpha,
                        notears_threshold=notears_threshold,
                        notears_poly_degree=notears_poly_degree,
                        notears_start_iter=notears_start_iter,
                        use_greedy_refinement=use_greedy_refinement,  # 贪心参数
                        greedy_max_modifications=greedy_max_modifications,
                        greedy_min_improvement=greedy_min_improvement,
                        greedy_eval_epochs=greedy_eval_epochs,
                        greedy_max_candidates=greedy_max_candidates,
                        greedy_start_iter=greedy_start_iter,
                        mcts_simulations=mcts_simulations,  # MCTS参数
                        mcts_exploration_weight=mcts_exploration_weight,
                        mcts_max_depth=mcts_max_depth,
                        use_baseline_reference=kwargs.get('use_baseline_reference', False),  # 基线参考参数
                        baseline_predict_dir=kwargs.get('baseline_predict_dir', 'predict'),
                        baseline_methods=kwargs.get('baseline_methods', ['corr', 'invcov']),
                        baseline_top_k=kwargs.get('baseline_top_k', 10),
                        baseline_threshold=kwargs.get('baseline_threshold', 0.5),
                        choose_best=choose_best,
                        use_intervention_test=use_intervention_test,
                        num_intervention_experiments=num_intervention_experiments,
                        llm_type=self.llm_type,
                        llm_model_path=self.llm_model_path,
                        openai_base_url=self.openai_base_url,
                        openai_api_key=self.openai_api_key,
                        shared_tokenizer=self.shared_tokenizer,
                        shared_model=self.shared_model
                    )
                    
                    # 移除 baseline 参数，因为它们已经在 __init__ 中使用了
                    run_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['use_baseline_reference', 'baseline_predict_dir', 
                                             'baseline_methods', 'baseline_top_k', 
                                             'baseline_threshold', 'choose_best']}
                    
                    pipeline.run(
                        num_iterations=num_iterations,
                        iterations_per_node=iterations_per_node,
                        early_stopping_patience=early_stopping_patience,
                        num_epochs=num_epochs,
                        llm_only=llm_only,
                        choose_best=choose_best,
                        **run_kwargs
                    )
                    
                    # 提取评估指标(如果有ground truth)
                    final_graph, best_ll = pipeline.get_best_graph()
                    metrics = _compute_metrics(pipeline, final_graph)
                    
                    # 保存该次运行的详细历史和结果到本地目录
                    trial_history_path = os.path.join(trial_output_dir, "run_history.json")
                    with open(trial_history_path, 'w') as f:
                        json.dump(pipeline.iteration_history, f, indent=2)
                    
                    trial_summary_path = os.path.join(trial_output_dir, "run_summary.json")
                    trial_summary = {
                        "run_id": run_idx,
                        "domain": dataset.domain_name,
                        "final_ll": best_ll,
                        "final_bic": final_graph['metadata'].get('bic'),
                        "num_edges_predicted": final_graph['metadata']['num_edges'],
                        "num_edges_true": int(dataset.ground_truth_graph.sum()),
                        "metrics": metrics,
                        "status": "success"
                    }
                    with open(trial_summary_path, 'w') as f:
                        json.dump(trial_summary, f, indent=2)
                    
                    # 提取骨架指标（如果使用了骨架）
                    skeleton_metrics = None
                    skeleton_info_path = os.path.join(trial_output_dir, "skeleton_info.json")
                    if os.path.exists(skeleton_info_path):
                        with open(skeleton_info_path, 'r') as f:
                            skeleton_info = json.load(f)
                            skeleton_metrics = skeleton_info.get('metrics', {})
                    
                    dataset_run_results.append({
                        "run_id": run_idx,
                        "status": "success",
                        "final_ll": best_ll,
                        "num_edges_predicted": final_graph['metadata']['num_edges'],
                        "num_edges_true": int(dataset.ground_truth_graph.sum()),
                        "metrics": metrics,
                        "skeleton_metrics": skeleton_metrics,
                        "iteration_history": pipeline.iteration_history
                    })
                    
                    print(f"✅ Trial {run_idx+1} completed")
                    
                except Exception as e:
                    print(f"❌ Trial {run_idx+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 记录失败信息到本地目录
                    trial_summary_path = os.path.join(trial_output_dir, "run_summary.json")
                    with open(trial_summary_path, 'w') as f:
                        json.dump({
                            "run_id": run_idx,
                            "domain": dataset.domain_name,
                            "status": "failed",
                            "error": str(e)
                        }, f, indent=2)
                        
                    dataset_run_results.append({
                        "run_id": run_idx,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # --- 汇总该数据集的所有运行结果 ---
            success_runs = [r for r in dataset_run_results if r['status'] == 'success']
            if success_runs:
                # 提取指标进行统计
                metrics_to_stats = ['f1_score', 'shd', 'precision', 'recall']
                stats_summary = {}
                
                for m in metrics_to_stats:
                    vals = [r['metrics'].get(m, 0) for r in success_runs]
                    stats_summary[f"{m}_mean"] = float(np.mean(vals))
                    stats_summary[f"{m}_std"] = float(np.std(vals))
                
                # 最终LL统计
                ll_vals = [r['final_ll'] for r in success_runs if r['final_ll'] != float('-inf')]
                if ll_vals:
                    stats_summary["ll_mean"] = float(np.mean(ll_vals))
                    stats_summary["ll_std"] = float(np.std(ll_vals))
                
                results_summary.append({
                    "experiment_id": idx,
                    "domain": dataset.domain_name,
                    "status": "success",
                    "num_runs": num_runs,
                    "success_runs": len(success_runs),
                    "stats": stats_summary,
                    # 为了向后兼容，保留一个“代表性”结果（取第一个成功的运行）
                    "final_ll": success_runs[0]['final_ll'],
                    "num_edges_predicted": success_runs[0]['num_edges_predicted'],
                    "num_edges_true": success_runs[0]['num_edges_true'],
                    "metrics": success_runs[0]['metrics'],
                    "skeleton_metrics": success_runs[0]['skeleton_metrics'],
                    "output_dir": os.path.join(self.base_output_dir, f"{idx:02d}_{dataset.domain_name}"),
                    "all_runs": dataset_run_results
                })
            else:
                results_summary.append({
                    "experiment_id": idx,
                    "domain": dataset.domain_name,
                    "status": "failed",
                    "error": "All runs failed",
                    "output_dir": os.path.join(self.base_output_dir, f"{idx:02d}_{dataset.domain_name}")
                })
        
        # 保存总结
        summary_path = os.path.join(self.base_output_dir, "experiments_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 打印统计信息
        self._print_summary(results_summary)
        
        return results_summary
    
    def _print_summary(self, results_summary: List[Dict]):
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
            # 表格1: 每个实验的迭代历史
            for r in results_summary:
                if r['status'] == 'success':
                    print(f"\n{'-'*70}")
                    print(f"Iteration History - Experiment {r['experiment_id']} ({r['domain']}):")
                    print(f"{'-'*70}")
                    print(f"{'Iter':<6} {'Status':<8} {'SHD':<8} {'LL':<12} {'BIC':<12} {'Edges':<8} {'F1':<8} {'Precision':<10} {'Recall':<8}")
                    print(f"{'-'*70}")
                    
                    iteration_history = r.get('iteration_history', [])
                    for iteration in iteration_history:
                        # 检查是否被接受
                        accepted = iteration.get('accepted', True)
                        status_str = "✓" if accepted else "✗"
                        
                        shd = iteration['metrics'].get('shd', None) if iteration.get('metrics') else None
                        ll = iteration['results'].get('log_likelihood', None)
                        bic = iteration['results'].get('bic', None)
                        edges = iteration['results'].get('num_edges', iteration.get('graph', {}).get('metadata', {}).get('num_edges', 0) if iteration.get('graph') else 0)
                        f1 = iteration['metrics'].get('f1_score', None) if iteration.get('metrics') else None
                        precision = iteration['metrics'].get('precision', None) if iteration.get('metrics') else None
                        recall = iteration['metrics'].get('recall', None) if iteration.get('metrics') else None
                        
                        # to
                        shd_str = f"{shd:.1f}" if shd is not None else "N/A"
                        ll_str = f"{ll:.4f}" if ll is not None and ll != float('-inf') else "N/A"
                        bic_str = f"{bic:.2f}" if bic is not None else "N/A"
                        edges_str = f"{edges}" if edges is not None else "N/A"
                        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                        precision_str = f"{precision:.4f}" if precision is not None else "N/A"
                        recall_str = f"{recall:.4f}" if recall is not None else "N/A"
                        
                        print(f"{iteration['iteration']:<6} "
                              f"{status_str:<8} "
                              f"{shd_str:<8} "
                              f"{ll_str:<12} "
                              f"{bic_str:<12} "
                              f"{edges_str:<8} "
                              f"{f1_str:<8} "
                              f"{precision_str:<10} "
                              f"{recall_str:<8}")
            
            # 表格2: 最终结果 (包含统计信息)
            print(f"\n{'-'*70}")
            print("Final Results (Mean ± Std across trials):")
            print(f"{'-'*70}")
            print(f"{'ID':<5} {'Domain':<15} {'F1-Score':<20} {'SHD':<15} {'Precision':<15} {'Recall':<15}")
            print(f"{'-'*70}")
            
            for r in results_summary:
                if r['status'] == 'success':
                    s = r.get('stats', {})
                    f1_str = f"{s.get('f1_score_mean', 0):.4f}±{s.get('f1_score_std', 0):.4f}"
                    shd_str = f"{s.get('shd_mean', 0):.2f}±{s.get('shd_std', 0):.2f}"
                    prec_str = f"{s.get('precision_mean', 0):.4f}±{s.get('precision_std', 0):.4f}"
                    rec_str = f"{s.get('recall_mean', 0):.4f}±{s.get('recall_std', 0):.4f}"
                    
                    print(f"{r['experiment_id']:<5} "
                          f"{r['domain']:<15} "
                          f"{f1_str:<20} "
                          f"{shd_str:<15} "
                          f"{prec_str:<15} "
                          f"{rec_str:<15}")
            
            # 表格3: 骨架质量（如果有）
            has_skeleton = any(r.get('skeleton_metrics') for r in results_summary if r['status'] == 'success')
            if has_skeleton:
                print(f"\n{'-'*70}")
                print("Skeleton Quality (before LLM):")
                print(f"{'-'*70}")
                print(f"{'ID':<5} {'Domain':<15} {'Skeleton F1':<12} {'Skeleton SHD':<12} {'Precision':<12} {'Recall':<12}")
                print(f"{'-'*70}")
                
                for r in results_summary:
                    if r['status'] == 'success' and r.get('skeleton_metrics'):
                        skel = r['skeleton_metrics']
                        print(f"{r['experiment_id']:<5} "
                              f"{r['domain']:<15} "
                              f"{skel.get('f1_score', 0):<12.4f} "
                              f"{skel.get('shd', 0):<12} "
                              f"{skel.get('precision', 0):<12.4f} "
                              f"{skel.get('recall', 0):<12.4f}")
        
        if failed_count > 0:
            print(f"\n{'-'*70}")
            print("Failed Experiments:")
            print(f"{'-'*70}")
            for r in results_summary:
                if r['status'] == 'failed':
                    print(f"  {r['experiment_id']}: {r['domain']} - {r['error'][:50]}")
        
        print(f"\n{'='*70}\n")

# ========== 使用示例 ==========
if __name__ == "__main__":
    
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='CMA Pipeline - Causal Discovery')
    parser.add_argument('--mode', type=str, default='batch', choices=['single', 'batch', 'llm-only'],
                       help='运行模式: single(单个实验), batch(批量实验) 或 llm-only(仅LLM生成一次)')
    parser.add_argument('--llm_type', type=str, default='local', choices=['local', 'openai'],
                       help='LLM类型: local 或 openai')
    parser.add_argument('--model_path', type=str, 
                       default='/mnt/shared-storage-user/safewt-share/HuggingfaceModels/Qwen3-14B',
                       help='本地模型路径(llm_type=local时使用)')
    parser.add_argument('--openai_url', type=str,
                       default='http://35.220.164.252:3888/v1/',
                       help='OpenAI API URL')
    parser.add_argument('--openai_key', type=str,
                       default='sk-x1DLgF9tW1t2IwCrUFyCfIIYFookGgO4qseCxb9vefNHQPcp',
                       help='OpenAI API key')
    parser.add_argument('--csv_path', type=str,
                       default='/mnt/shared-storage-user/pengbo/created/projects/CDLLM/Test-1213/real_test.csv',
                       help='批量实验的CSV配置文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='./cma_experiments',
                       help='输出目录')
    parser.add_argument('--num_iterations', type=int, default=None,
                       help='CMA迭代次数（默认为None，由iterations_per_node计算）')
    parser.add_argument('--iterations_per_node', type=float, default=1.0,
                       help='当num_iterations为None时，每节点分配的迭代次数')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='早停耐心值：连续多少次图修改未被接受则停止迭代')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='显著性测试的独立运行次数')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='模型拟合的epoch数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备: cpu 或 cuda')
    parser.add_argument('--use_hill_climbing', action='store_true',
                       help='启用爬山策略(基于LL接受/拒绝图修改)')
    parser.add_argument('--acceptance_tolerance', type=float, default=0.0,
                       help='爬山策略的接受范围: new_ll >= best_ll - tolerance')
    
    # MCTS参数
    parser.add_argument('--use_mcts', action='store_true',
                       help='使用MCTS搜索策略（与use_hill_climbing互斥）')
    parser.add_argument('--mcts_simulations', type=int, default=50,
                       help='MCTS每次迭代的模拟次数')
    parser.add_argument('--mcts_exploration_weight', type=float, default=1.414,
                       help='MCTS的UCB1探索权重（sqrt(2)≈1.414）')
    parser.add_argument('--mcts_max_depth', type=int, default=5,
                       help='MCTS的最大搜索深度')
    parser.add_argument('--use_skeleton', action='store_true',
                       help='启用MMHC骨架构建，用统计方法缩小搜索空间')
    parser.add_argument('--skeleton_alpha', type=float, default=0.05,
                       help='骨架构建的独立性检验显著性水平')
    parser.add_argument('--skeleton_max_cond_size', type=int, default=3,
                       help='骨架构建的最大条件集大小')
    parser.add_argument('--verbose', action='store_true',
                       help='')
    parser.add_argument('--max_retries', type=int, default=10,
                       help='最大重试次数')
    
    # NOTEARS参数
    parser.add_argument('--use_notears_refinement', action='store_true',
                       help='使用NOTEARS优化')
    parser.add_argument('--notears_use_mlp', action='store_true',
                       help='NOTEARS使用MLP作为score（推荐，更准确）')
    
    # 贪心优化参数（推荐）
    parser.add_argument('--use_greedy_refinement', action='store_true',
                       help='使用贪心图优化（推荐）')
    parser.add_argument('--greedy_max_modifications', type=int, default=10,
                       help='贪心优化的最大修改次数')
    parser.add_argument('--greedy_min_improvement', type=float, default=0.01,
                       help='贪心优化的最小LL改进阈值')
    parser.add_argument('--greedy_eval_epochs', type=int, default=15,
                       help='贪心评估时的训练轮数（降低以加速）')
    parser.add_argument('--greedy_max_candidates', type=int, default=30,
                       help='每种操作最多测试的候选数（加速大图）')
    parser.add_argument('--greedy_start_iter', type=int, default=0,
                       help='从第几轮迭代开始使用贪心优化')
    
    # 基线参考参数
    parser.add_argument('--use_baseline_reference', action='store_true',
                       help='使用传统方法的预测结果作为LLM参考')
    parser.add_argument('--baseline_predict_dir', type=str, default='predict',
                       help='predict目录路径（包含预先计算的预测结果）')
    parser.add_argument('--baseline_methods', type=str, nargs='+', default=['corr', 'invcov'],
                       help='要加载的基线方法列表，如: corr invcov notears')
    parser.add_argument('--baseline_top_k', type=int, default=10,
                       help='每个方法显示top-k个最强关系')
    parser.add_argument('--baseline_threshold', type=float, default=0.5,
                       help='筛选阈值的百分位数（0-100）')
    parser.add_argument('--use_local_amendment', action='store_true',
                       help='使用本地修正')
    parser.add_argument('--choose_best', action='store_true',
                       help='在初始阶段比较基线方法和全局LLM生成的结果，选择BIC更好的那个')
    parser.add_argument('--use_intervention_test', action='store_true',
                       help='启用干预实验验证逻辑')
    parser.add_argument('--num_intervention_experiments', type=int, default=3,
                       help='每轮允许提出的最大干预实验数')
    
    args = parser.parse_args()
    
    # 根据模型和基线方法构建输出路径
    model_tag = "openai"
    if args.llm_type == 'local' and args.model_path:
        model_tag = os.path.basename(args.model_path.rstrip('/'))
    elif args.llm_type == 'openai':
        model_tag = "openai"
    
    baseline_tag = "no_baseline"
    if args.use_baseline_reference:
        # 如果 baseline_methods 是列表，将其排序并连接
        if isinstance(args.baseline_methods, list):
            baseline_tag = "_".join(sorted(args.baseline_methods))
        else:
            baseline_tag = str(args.baseline_methods)
    if args.mode == 'llm-only':
        baseline_tag = "llm-only"
    
    # 更新 output_dir
    args.output_dir = os.path.join(args.output_dir, f"{model_tag}_{baseline_tag}")

    # ========== 批量实验模式 ==========
    if args.mode in ['batch', 'llm-only']:
        print("\n" + "="*80)
        print(f"CMA {'LLM-ONLY' if args.mode == 'llm-only' else 'BATCH'} EXPERIMENTS")
        print("="*80)
        print(f"Configuration:")
        print(f"  CSV Path: {args.csv_path}")
        print(f"  Output Dir: {args.output_dir}")
        print(f"  LLM Type: {args.llm_type}")
        if args.llm_type == 'local':
            print(f"  Model Path: {args.model_path}")
        else:
            print(f"  API URL: {args.openai_url}")
        
        if args.mode == 'llm-only':
            print(f"  Mode: LLM-only (Stop after first successful global graph)")
        else:
            print(f"  Iterations: {args.num_iterations if args.num_iterations else 'Auto (per node)'}")
            if args.num_iterations is None:
                print(f"  Iterations per node: {args.iterations_per_node}")
            print(f"  Early stopping patience: {args.early_stopping_patience}")
        
        print(f"  Trials per dataset: {args.num_runs}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Device: {args.device}")
        print(f"  Hill Climbing: {args.use_hill_climbing}")
        if args.use_hill_climbing:
            print(f"  Acceptance Tolerance: {args.acceptance_tolerance}")
        print(f"  Use Skeleton: {args.use_skeleton}")
        if args.use_skeleton:
            print(f"  Skeleton Alpha: {args.skeleton_alpha}")
            print(f"  Skeleton Max Cond Size: {args.skeleton_max_cond_size}")
        print(f"  Use Baseline Reference: {args.use_baseline_reference}")
        if args.use_baseline_reference:
            print(f"  Baseline Predict Dir: {args.baseline_predict_dir}")
            print(f"  Baseline Methods: {args.baseline_methods}")
            print(f"  Baseline Top-K: {args.baseline_top_k}")
            print(f"  Baseline Threshold Percentile: {args.baseline_threshold}")
        print(f"  Choose Best Initial: {args.choose_best}")
        print("="*80 + "\n")
        
        # 验证文件
        if not os.path.exists(args.csv_path):
            print(f"❌ Error: CSV file not found: {args.csv_path}")
            exit(1)
        
        if args.llm_type == 'local' and not os.path.exists(args.model_path):
            print(f"❌ Error: Model path not found: {args.model_path}")
            exit(1)
        
        # 创建批量运行器
        runner = BatchExperimentRunner(
            csv_config_path=args.csv_path,
            base_output_dir=args.output_dir,
            llm_type=args.llm_type,
            llm_model_path=args.model_path if args.llm_type == 'local' else None,
            openai_base_url=args.openai_url if args.llm_type == 'openai' else None,
            openai_api_key=args.openai_key if args.llm_type == 'openai' else None
        )
        
        # 运行批量实验
        runner.run_all_experiments(
            split='test',
            num_runs=args.num_runs,
            num_iterations=args.num_iterations,
            iterations_per_node=args.iterations_per_node,
            early_stopping_patience=args.early_stopping_patience,
            num_epochs=args.num_epochs,
            device=args.device,
            learning_rate=0.01,
            temperature=0.6,
            use_hill_climbing=args.use_hill_climbing,
            acceptance_tolerance=args.acceptance_tolerance,
            verbose=args.verbose,
            max_retries=args.max_retries,
            use_skeleton=args.use_skeleton,
            skeleton_alpha=args.skeleton_alpha,
            skeleton_max_cond_size=args.skeleton_max_cond_size,
            use_notears_refinement=args.use_notears_refinement,
            notears_use_mlp=args.notears_use_mlp,
            use_greedy_refinement=args.use_greedy_refinement,
            greedy_max_modifications=args.greedy_max_modifications,
            greedy_min_improvement=args.greedy_min_improvement,
            greedy_eval_epochs=args.greedy_eval_epochs,
            greedy_max_candidates=args.greedy_max_candidates,
            greedy_start_iter=args.greedy_start_iter,
            use_mcts=args.use_mcts,
            mcts_simulations=args.mcts_simulations,
            mcts_exploration_weight=args.mcts_exploration_weight,
            mcts_max_depth=args.mcts_max_depth,
            llm_only=(args.mode == 'llm-only'),
            choose_best=args.choose_best,
            # 基线参考参数
            use_baseline_reference=args.use_baseline_reference,
            baseline_predict_dir=args.baseline_predict_dir,
            baseline_methods=args.baseline_methods,
            baseline_top_k=args.baseline_top_k,
            baseline_threshold=args.baseline_threshold,
            use_local_amendment=args.use_local_amendment,
            use_intervention_test=args.use_intervention_test,
            num_intervention_experiments=args.num_intervention_experiments
        )
        
        print(f"\n✅ All experiments completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Summary: {args.output_dir}/experiments_summary.json\n")