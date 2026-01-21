"""
Data Loader Module
加载因果发现数据集 弃用 interventions 相关功能
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from ..utils.config_manager import ConfigManager


class CausalDataset:
    """因果发现数据集"""
    
    def __init__(
        self,
        data: np.ndarray,
        ground_truth_graph: np.ndarray,
        interventions: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        domain_name: str = "unknown",
        variable_type: str = "continuous"
    ):
        """
        Args:
            data: 观测数据 [n_samples, n_variables]
            ground_truth_graph: 真实的因果图邻接矩阵 [n_variables, n_variables]
            interventions: 干预矩阵 [n_samples, n_variables], 1表示该变量被干预
            variable_names: 变量名列表
            domain_name: 领域名称
            variable_type: 变量类型 ("continuous" 或 "discrete")
        """
        self.data = data
        self.ground_truth_graph = ground_truth_graph
        self.interventions = interventions
        self.domain_name = domain_name
        self.variable_type = variable_type
        
        self.n_samples, self.n_variables = data.shape
        
        # 变量名
        if variable_names is None:
            self.variable_names = [f"X{i}" for i in range(self.n_variables)]
        else:
            self.variable_names = variable_names
        
        assert len(self.variable_names) == self.n_variables
        
    def get_observational_data(self) -> np.ndarray:
        """获取纯观测数据(排除干预样本)"""
        if self.interventions is None:
            return self.data
        
        # 找到没有任何干预的样本
        no_intervention_mask = (self.interventions.sum(axis=1) == 0)
        return self.data[no_intervention_mask]
    
    def get_intervention_summary(self) -> Dict:
        """获取干预信息摘要"""
        if self.interventions is None:
            return {"has_interventions": False}
        
        total_samples = len(self.interventions)
        intervened_samples = (self.interventions.sum(axis=1) > 0).sum()
        observational_samples = total_samples - intervened_samples
        
        # 统计每个变量被干预的次数
        intervention_counts = self.interventions.sum(axis=0)
        
        summary = {
            "has_interventions": True,
            "total_samples": total_samples,
            "observational_samples": int(observational_samples),
            "intervened_samples": int(intervened_samples),
            "intervention_per_variable": {
                self.variable_names[i]: int(intervention_counts[i])
                for i in range(self.n_variables)
            }
        }
        
        return summary
    
    def get_ground_truth_edges(self) -> List[Tuple[str, str]]:
        """获取真实的因果边"""
        edges = []
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                if self.ground_truth_graph[i, j] == 1:
                    edges.append((self.variable_names[i], self.variable_names[j]))
        return edges
    
    def print_summary(self):
        """打印数据集摘要"""
        print("\n" + "="*70)
        print(f"DATASET SUMMARY: {self.domain_name}")
        print("="*70)
        print(f"Variables: {self.n_variables}")
        print(f"  {', '.join(self.variable_names)}")
        print(f"\nData shape: {self.data.shape}")
        print(f"Data range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        
        print(f"\nGround Truth Graph:")
        print(f"  Total edges: {self.ground_truth_graph.sum()}")
        
        if self.interventions is not None:
            summary = self.get_intervention_summary()
            print(f"\nInterventions:")
            print(f"  Observational samples: {summary['observational_samples']}")
            print(f"  Intervened samples: {summary['intervened_samples']}")
            print(f"  Interventions per variable:")
            for var, count in summary['intervention_per_variable'].items():
                if count > 0:
                    print(f"    {var}: {count}")
        
        print("="*70 + "\n")


class DataLoader:
    """数据加载器"""
    
    # 预定义的变量名
    VARIABLE_NAMES = {
        "earthquake": ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'],
        "cancer": ['Pollution', 'Smoker', 'Cancer', 'Xray', 'Dyspnoea'],
        "asia": ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp'],
        "sachs": ['Plcg', 'PKC', 'PKA', 'Jnk', 'P38', 'Raf', 'Mek', 'Erk', 'Akt', 'PIP3', 'PIP2'], # 可能是alarm的
        # "sachs": ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk'],
        "child": ['BirthAsphyxia', 'Disease', 'LVH', 'LVHreport', 'DuctFlow', 
                  'CardiacMixing', 'HypDistrib', 'LungParench', 'HypoxiaInO2', 'CO2', 
                  'LowerBodyO2', 'RUQO2', 'CO2Report', 'LungFlow', 'ChestXray', 
                  'XrayReport', 'Sick', 'Grunting', 'GruntingReport', 'Age'],
        "alarm": ['HYPOVOLEMIA', 'LVFAILURE', 'HISTORY', 'LVEDVOLUME', 'CVP', 'PCWP', 
                  'STROKEVOLUME', 'ERRLOWOUTPUT', 'ERRCAUTER', 'INSUFFANESTH', 
                  'ANAPHYLAXIS', 'TPR', 'KINKEDTUBE', 'FIO2', 'PULMEMBOLUS', 'PAP', 
                  'INTUBATION', 'SHUNT', 'DISCONNECT', 'MINVOLSET', 'VENTMACH', 
                  'VENTTUBE', 'PRESS', 'VENTLUNG', 'MINVOL', 'VENTALV', 'PVSAT', 
                  'SAO2', 'ARTCO2', 'EXPCO2', 'CATECHOL', 'HR', 'HRBP', 'HREKG', 
                  'HRSAT', 'CO', 'BP'],
        "insurance": ['GoodStudent', 'Age', 'SocioEcon', 'RiskAversion', 'VehicleYear', 'ThisCarDam', 'RuggedAuto', 'Accident', 'MakeModel', 'DrivQuality', 'Mileage', 'Antilock', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost', 'Theft', 'CarValue', 'HomeBase', 'AntiTheft', 'PropCost', 'OtherCarCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag', 'ILiCost', 'DrivHist'],
        "water": ['C_NI_12_00', 'CKNI_12_00', 'CBODD_12_00', 'CKND_12_00', 'CNOD_12_00', 'CBODN_12_00', 'CKNN_12_00', 'CNON_12_00', 'C_NI_12_15', 'CKNI_12_15', 'CBODD_12_15', 'CKND_12_15', 'CNOD_12_15', 'CBODN_12_15', 'CKNN_12_15', 'CNON_12_15', 'C_NI_12_30', 'CKNI_12_30', 'CBODD_12_30', 'CKND_12_30', 'CNOD_12_30', 'CBODN_12_30', 'CKNN_12_30', 'CNON_12_30', 'C_NI_12_45', 'CKNI_12_45', 'CBODD_12_45', 'CKND_12_45', 'CNOD_12_45', 'CBODN_12_45', 'CKNN_12_45', 'CNON_12_45']
    }
    
    @staticmethod
    def load_from_paths(
        fp_data: str,
        fp_graph: str,
        fp_regime: str = "None",
        domain_name: str = None
    ) -> CausalDataset:
        """
        从文件路径加载数据
        
        Args:
            fp_data: 数据文件路径 (.npy)
            fp_graph: 图邻接矩阵路径 (.npy)
            fp_regime: 干预regime文件路径 (.csv)
            domain_name: 领域名称(如果None,从路径推断)
        """
        
        # 加载数据
        data = np.load(fp_data)
        label = np.load(fp_graph)
        
        # 推断领域名称
        if domain_name is None:
            domain_name = DataLoader._infer_domain_name(fp_data)
        
        # 加载干预信息
        interventions = None
        if fp_regime != "None" and os.path.exists(fp_regime):
            interventions = DataLoader._load_interventions(fp_regime, data, label)
        
        # 获取变量名和类型
        variable_names = DataLoader.VARIABLE_NAMES.get(domain_name)
        variable_type = DOMAIN_VARIABLE_TYPES.get(domain_name, "continuous")
        
        dataset = CausalDataset(
            data=data,
            ground_truth_graph=label,
            interventions=interventions,
            variable_names=variable_names,
            domain_name=domain_name,
            variable_type=variable_type
        )
        
        return dataset
    
    @staticmethod
    def load_from_csv_config(csv_path: str, row_index: int = 0) -> CausalDataset:
        """
        从CSV配置文件加载数据
        
        Args:
            csv_path: CSV配置文件路径
            row_index: 要加载的行索引
        """
        df = pd.read_csv(csv_path)
        row = df.iloc[row_index]
        
        return DataLoader.load_from_paths(
            fp_data=row['fp_data'],
            fp_graph=row['fp_graph'],
            fp_regime=row['fp_regime']
        )
    
    @staticmethod
    def _load_interventions(fp_regime: str, data: np.ndarray, label: np.ndarray) -> np.ndarray:
        """加载干预信息"""
        with open(fp_regime) as f:
            lines = [line.strip() for line in f.readlines()]
        
        # 解析regime
        regimes = [
            tuple(sorted(int(x) for x in line.split(","))) if len(line) > 0 else ()
            for line in lines
        ]
        
        assert len(regimes) == len(data), f"Regimes ({len(regimes)}) and data ({len(data)}) length mismatch"
        
        # 转换为干预矩阵
        interv = np.zeros((len(data), len(label)))
        for i, regime in enumerate(regimes):
            for node in regime:
                interv[i, node] = 1
        
        return interv
    
    @staticmethod
    def _infer_domain_name(fp_data: str) -> str:
        """从文件路径推断领域名称"""
        path_lower = fp_data.lower()
        
        for domain in DataLoader.VARIABLE_NAMES.keys():
            if domain in path_lower:
                return domain
        
        # 无法推断,使用默认名称
        return "unknown"
    
    @staticmethod
    def load_all_from_csv(csv_path: str, split: str = "test") -> List[CausalDataset]:
        """
        从CSV加载所有指定split的数据集
        
        Args:
            csv_path: CSV配置文件路径
            split: 'test', 'train', 或 'val'
        """
        df = pd.read_csv(csv_path)
        df_split = df[df['split'] == split]
        
        datasets = []
        for idx, row in df_split.iterrows():
            try:
                dataset = DataLoader.load_from_paths(
                    fp_data=row['fp_data'],
                    fp_graph=row['fp_graph'],
                    fp_regime=row['fp_regime']
                )
                datasets.append(dataset)
                print(f"✓ Loaded: {dataset.domain_name}")
            except Exception as e:
                print(f"✗ Failed to load {row['fp_data']}: {e}")
        
        return datasets


# ========== 新增：从配置文件加载数据 ==========
    @staticmethod
    def load_from_config(config_manager: ConfigManager = None) -> CausalDataset:
        """
        从 ConfigManager 加载单个数据集
        
        Args:
            config_manager: ConfigManager 实例，如果为None则创建新实例
        
        Returns:
            加载的数据集
        """
        if config_manager is None:
            config_manager = ConfigManager()
        
        # 现在直接用 get() 即可，路径已经在加载时转换为绝对路径
        fp_data = config_manager.get("experiment.single.data.fp_data")
        fp_graph = config_manager.get("experiment.single.data.fp_graph")
        
        # 检查路径是否有效
        if not fp_data or not fp_graph:
            raise ValueError(
                "数据路径未配置。请在 config/default.toml 或 config/secret.toml 中设置 "
                "experiment.single.data.fp_data 和 experiment.single.data.fp_graph"
            )
        
        # 检查文件是否存在
        if not os.path.exists(fp_data):
            raise FileNotFoundError(f"数据文件不存在: {fp_data}")
        if not os.path.exists(fp_graph):
            raise FileNotFoundError(f"图文件不存在: {fp_graph}")
        
        # 获取领域名称
        domain_name = config_manager.get("experiment.single.domain_name", "unknown")
        
        return DataLoader.load_from_paths(
            fp_data=fp_data,
            fp_graph=fp_graph,
            fp_regime="None",  # 配置文件暂不支持干预数据
            domain_name=domain_name
        )


# ========== 领域背景知识 ==========
DOMAIN_CONTEXTS = {
    "sachs": """Protein signaling networks in human immune cells.""",
    
    "cancer": """Lung cancer risk factors.""",
    
    "asia": """Tuberculosis diagnosis.""",
    
    "earthquake": """Classic burglar alarm.""",
    
    "child": """Congenital heart disease diagnosis.""",
    
    "alarm": """Logical Alarm Reduction."""
}

# 数据集变量类型配置
DOMAIN_VARIABLE_TYPES = {
    "sachs": "continuous",      # 蛋白质浓度是连续值
    "cancer": "discrete",        # 贝叶斯网络，离散变量
    "asia": "discrete",          # 贝叶斯网络，离散变量
    "earthquake": "discrete",    # 贝叶斯网络，离散变量
    "child": "discrete",         # 贝叶斯网络，离散变量
    "alarm": "discrete",         # 贝叶斯网络，离散变量
}


# ========== 测试代码 ==========
if __name__ == "__main__":
    
    # 测试1: 从单个路径加载
    print("Test 1: Load single dataset")
    print("-"*70)
    
    dataset = DataLoader.load_from_paths(
        fp_data="/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/datasets/bnlearn_intv/data_n1000_earthquake/data_interv1.npy",
        fp_graph="/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/datasets/bnlearn_intv/data_n1000_earthquake/DAG1.npy",
        fp_regime="/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/datasets/bnlearn_intv/data_n1000_earthquake/intervention1.csv"
    )
    
    dataset.print_summary()
    
    print("Ground truth edges:")
    for edge in dataset.get_ground_truth_edges():
        print(f"  {edge[0]} → {edge[1]}")
    
    # 测试2: 从CSV加载
    print("\n\nTest 2: Load from CSV config")
    print("-"*70)
    
    # 创建临时CSV文件
    csv_content = """fp_data,fp_graph,fp_regime,split
/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/datasets/bnlearn_intv/data_n1000_cancer/data_interv1.npy,/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/datasets/bnlearn_intv/data_n1000_cancer/DAG1.npy,/mnt/shared-storage-user/safewt-share/pengbo/CausalDiscovery/data/datasets/bnlearn_intv/data_n1000_cancer/intervention1.csv,test"""
    
    with open("temp_config.csv", "w") as f:
        f.write(csv_content)
    
    dataset2 = DataLoader.load_from_csv_config("temp_config.csv", row_index=0)
    dataset2.print_summary()
    
    # 测试3: 获取纯观测数据
    print("\n\nTest 3: Get observational data")
    print("-"*70)
    
    obs_data = dataset.get_observational_data()
    print(f"Original data shape: {dataset.data.shape}")
    print(f"Observational data shape: {obs_data.shape}")