"""
Data Loader Module (Refactored for LLM-based Causal Discovery)
加载 数据 和 真实因果图 (Ground Truth), 
移除了所有 Intervention/Regime 相关逻辑。
"""

import numpy as np
import pandas as pd
import os

# ==== 运行测试用：将 src 目录添加到搜索路径 ====
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ==========================================

from schemas import CausalDataset

class DataLoader:
    """数据加载器"""
    
    # 预定义的变量名 (LLM 需要理解这些语义)
    VARIABLE_NAMES = {
        "earthquake": ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls'],
        "cancer": ['Pollution', 'Smoker', 'Cancer', 'Xray', 'Dyspnoea'],
        "asia": ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp'],
        "sachs": ['Plcg', 'PKC', 'PKA', 'Jnk', 'P38', 'Raf', 'Mek', 'Erk', 'Akt', 'PIP3', 'PIP2'],
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
    }
    
    # 数据集变量类型
    DOMAIN_VARIABLE_TYPES = {
        "sachs": "continuous",
        "cancer": "discrete",
        "asia": "discrete",
        "earthquake": "discrete",
        "child": "discrete",
        "alarm": "discrete",
    }

    @staticmethod
    def load_from_paths(
        fp_data: str,
        fp_graph: str,
        domain_name: str = None
    ) -> CausalDataset:
        """
        从文件路径加载
        Args:
            fp_data: 数据文件路径 (.npy)
            fp_graph: 图邻接矩阵路径 (.npy)
            domain_name: 领域名称 (可选, 若无则从路径推断)
        """
        if not os.path.exists(fp_data):
            raise FileNotFoundError(f"Data file not found: {fp_data}")
        if not os.path.exists(fp_graph):
            raise FileNotFoundError(f"Graph file not found: {fp_graph}")

        # 加载 numpy 数据
        data = np.load(fp_data)
        ground_truth = np.load(fp_graph)
        
        # 推断领域名称
        if domain_name is None:
            domain_name = DataLoader._infer_domain_name(fp_data)
        
        # 获取元数据
        variable_names = DataLoader.VARIABLE_NAMES.get(domain_name)
        variable_type = DataLoader.DOMAIN_VARIABLE_TYPES.get(domain_name, "continuous")
        
        # 创建数据集对象
        dataset = CausalDataset(
            data=data,
            ground_truth_graph=ground_truth,
            variable_names=variable_names,
            domain_name=domain_name,
            variable_type=variable_type
        )
        
        return dataset
    
    @staticmethod
    def load_from_csv_config(csv_path: str, row_index: int = 0) -> CausalDataset:
        """从CSV配置加载"""
        df = pd.read_csv(csv_path)
        row = df.iloc[row_index]
        
        return DataLoader.load_from_paths(
            fp_data=row['fp_data'],
            fp_graph=row['fp_graph']
        )
    
    @staticmethod
    def _infer_domain_name(fp_data: str) -> str:
        """从路径字符串中猜测数据集名字"""
        path_lower = fp_data.lower()
        for domain in DataLoader.VARIABLE_NAMES.keys():
            if domain in path_lower:
                return domain
        return "unknown"


# ========== 领域背景知识 (Prompt Engineering 用) ==========
# 可以在后续 Agent 代码中引用这个字典
DOMAIN_CONTEXTS = {
    "sachs": "Protein signaling networks in human immune cells.",
    "cancer": "Lung cancer risk factors and symptoms.",
    "asia": "Medical diagnosis for lung diseases including tuberculosis and cancer.",
    "earthquake": "A home alarm system responding to burglaries and earthquakes.",
    "child": "Diagnosis of congenital heart disease in infants.",
    "alarm": "Anesthesia patient monitoring system in an operating room."
}

# ========== 测试代码 ==========
# 记得解除最上面的路径添加注释
if __name__ == "__main__":
    from utils import ConfigManager
    config = ConfigManager()
    test_data_path = config.get("experiment.single.fp_data")
    test_graph_path = config.get("experiment.single.fp_graph")

    print("Test: Load dataset")
    print(f"Data path: {os.path.abspath(test_data_path)}")
    print(f"Graph path: {os.path.abspath(test_graph_path)}")
    ds = DataLoader.load_from_paths(
        fp_data=test_data_path,
        fp_graph=test_graph_path
    )
    
    ds.print_summary()