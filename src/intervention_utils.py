import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from src.data_loader.data_loader import CausalDataset
from scipy import stats

class InterventionTester:
    """
    基于干预数据的因果关系验证器
    """
    
    def __init__(self, dataset: CausalDataset):
        self.dataset = dataset
        self.variable_names = dataset.variable_names
        self.n_vars = dataset.n_variables
        self.is_categorical = dataset.variable_type == "discrete"

    def test_intervention_effect(self, treatment_var: str, target_var: str, alpha: float = 0.05) -> Dict:
        """
        测试 do(treatment_var) 对 target_var 的影响
        """
        if treatment_var not in self.variable_names or target_var not in self.variable_names:
            return {"error": "Invalid variable names"}
            
        t_idx = self.variable_names.index(treatment_var)
        y_idx = self.variable_names.index(target_var)
        
        # 1. 获取观测数据样本 (无任何干预)
        obs_mask = (self.dataset.interventions.sum(axis=1) == 0)
        obs_data = self.dataset.data[obs_mask, y_idx]
        
        # 2. 获取 do(treatment_var) 的样本 (仅对 treatment_var 进行干预)
        # 注意：有些样本可能同时干预了多个变量，这里我们取包含 treatment_var 被干预的样本
        intv_mask = (self.dataset.interventions[:, t_idx] == 1)
        intv_data = self.dataset.data[intv_mask, y_idx]
        
        if len(intv_data) == 0:
            return {
                "treatment": treatment_var,
                "target": target_var,
                "success": False,
                "reason": f"No interventional data for {treatment_var}"
            }
            
        # 3. 统计测试
        if self.is_categorical:
            # 离散变量：使用卡方检验或比较分布差异
            # 这里简单起见，比较分布的差异显著性
            # 构建频率表
            all_vals = np.unique(self.dataset.data[:, y_idx])
            obs_counts = np.array([np.sum(obs_data == v) for v in all_vals])
            intv_counts = np.array([np.sum(intv_data == v) for v in all_vals])
            
            # 归一化为概率
            obs_prob = obs_counts / len(obs_data) if len(obs_data) > 0 else np.ones_like(all_vals)/len(all_vals)
            intv_prob = intv_counts / len(intv_data)
            
            # 使用 TVD (Total Variation Distance) 衡量差异
            tvd = 0.5 * np.sum(np.abs(obs_prob - intv_prob))
            
            # 简单启发式判断显著性 (后续可以改用更严谨的检验)
            is_significant = tvd > 0.1 
            
            sig_label = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
            implication = "SUPPORTS ANCESTRAL path (check for intermediate nodes before adding direct edge)" if is_significant else "CONTRADICTS existence / SUPPORTS deletion"
            
            return {
                "treatment": treatment_var,
                "target": target_var,
                "success": True,
                "is_significant": is_significant,
                "metric_name": "TVD",
                "metric_value": round(float(tvd), 4),
                "obs_mean": round(float(np.mean(obs_data)), 4) if len(obs_data) > 0 else None,
                "intv_mean": round(float(np.mean(intv_data)), 4),
                "description": f"RESULT: [{sig_label}] do({treatment_var}) affects {target_var} (TVD={tvd:.4f}). This {implication}."
            }
        else:
            # 连续变量：使用 t-test
            t_stat, p_val = stats.ttest_ind(obs_data, intv_data, equal_var=False)
            is_significant = p_val < alpha
            
            sig_label = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
            implication = "SUPPORTS ANCESTRAL path (check for intermediate nodes before adding direct edge)" if is_significant else "CONTRADICTS existence / SUPPORTS deletion"
            
            return {
                "treatment": treatment_var,
                "target": target_var,
                "success": True,
                "is_significant": is_significant,
                "metric_name": "p-value",
                "metric_value": round(float(p_val), 4),
                "obs_mean": round(float(np.mean(obs_data)), 4),
                "intv_mean": round(float(np.mean(intv_data)), 4),
                "description": f"RESULT: [{sig_label}] do({treatment_var}) affects {target_var} mean (p={p_val:.4f}). This {implication}."
            }

    def test_conditional_independence(self, treatment_var: str, target_var: str, conditioning_set: List[str], alpha: float = 0.05) -> Dict:
        """
        测试 do(treatment_var) 对 target_var 的影响在给定 conditioning_set 时是否消失
        用于区分直接原因和间接原因
        """
        # 0. 变量合法性检查
        if treatment_var not in self.variable_names or target_var not in self.variable_names:
            return {"error": f"Invalid variable names: {treatment_var} or {target_var}"}

        # 逻辑修复：剔除 conditioning_set 中的 treatment_var 和 target_var
        # 并且只保留确实存在于数据集中的变量 (防幻觉)
        filtered_z = [z for z in conditioning_set if z in self.variable_names and z != treatment_var and z != target_var]
        
        if not filtered_z:
            return self.test_intervention_effect(treatment_var, target_var, alpha)
            
        t_idx = self.variable_names.index(treatment_var)
        y_idx = self.variable_names.index(target_var)
        z_indices = [self.variable_names.index(z) for z in filtered_z]
        
        # 获取 do(X) 样本
        intv_mask = (self.dataset.interventions[:, t_idx] == 1)
        # 获取观测样本
        obs_mask = (self.dataset.interventions.sum(axis=1) == 0)
        
        # 这里为了简化，我们只取 conditioning_set 的第一种常见组合进行测试，或者对结果取平均
        # 实际更严谨的方法是分层测试
        
        # 提取数据
        intv_data = self.dataset.data[intv_mask]
        obs_data = self.dataset.data[obs_mask]
        
        # 找出共同的 Z 组合
        # 简单起见，我们取 Z 的众数或随机一个 level
        # 如果变量很多，这步会很复杂。在 Asia/Alarm 数据集中，Z 通常只有 1-2 个变量。
        
        # 记录每个 level 下的显著性
        results_per_level = []
        
        # 仅对离散变量支持较好
        if self.is_categorical:
            # 找到 Z 的所有组合
            z_data_intv = intv_data[:, z_indices]
            unique_z_levels = np.unique(z_data_intv, axis=0)
            
            for level in unique_z_levels[:3]: # 最多测3个组合，防止太慢
                level_mask_intv = np.all(intv_data[:, z_indices] == level, axis=1)
                level_mask_obs = np.all(obs_data[:, z_indices] == level, axis=1)
                
                sub_intv = intv_data[level_mask_intv, y_idx]
                sub_obs = obs_data[level_mask_obs, y_idx]
                
                if len(sub_intv) > 10 and len(sub_obs) > 10:
                    # 在该 level 下计算效应
                    all_vals = np.unique(self.dataset.data[:, y_idx])
                    p_obs = np.array([np.sum(sub_obs == v) for v in all_vals]) / len(sub_obs)
                    p_intv = np.array([np.sum(sub_intv == v) for v in all_vals]) / len(sub_intv)
                    tvd = 0.5 * np.sum(np.abs(p_obs - p_intv))
                    results_per_level.append(tvd)
            
            if not results_per_level:
                #  fallback to marginal if conditional data is sparse
                marginal_res = self.test_intervention_effect(treatment_var, target_var, alpha)
                if "error" in marginal_res:
                    return {"error": f"Insufficient data for conditional testing and {marginal_res['error']}"}
                
                return {
                    "treatment": treatment_var,
                    "target": target_var,
                    "conditioning_set": filtered_z,
                    "is_significant": marginal_res["is_significant"],
                    "description": f"INSUFFICIENT DATA to condition on {filtered_z}. FALLBACK to Marginal: {marginal_res['description']}"
                }
                
            avg_tvd = np.mean(results_per_level)
            is_significant = avg_tvd > 0.1
            
            sig_label = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
            path_type = "DIRECT / not fully mediated by current set" if is_significant else "INDIRECT / fully mediated by current set"
            
            return {
                "treatment": treatment_var,
                "target": target_var,
                "conditioning_set": filtered_z,
                "is_significant": is_significant,
                "avg_tvd": round(float(avg_tvd), 4),
                "description": f"RESULT: [{sig_label}] Given {filtered_z}, do({treatment_var}) affects {target_var} (avg_TVD={avg_tvd:.4f}). Link is likely {path_type}."
            }
        else:
            # 连续变量的简易条件独立性检查 (偏相关或残差回归)
            # 这里先返回未实现，或者使用简单的线性残差
            return {"error": "Conditional intervention testing for continuous data not yet fully implemented"}

    def run_experiments(self, experiments: List[Dict]) -> str:
        """
        运行一系列实验并返回文本报告
        experiments: [{"treatment": "A", "target": "B", "conditioning_set": ["C"]}]
        """
        report = []
        for exp in experiments:
            t = exp.get("treatment")
            target = exp.get("target")
            z = exp.get("conditioning_set", [])
            
            # 防御性检查：确保 treatment 和 target 存在
            if t not in self.variable_names or target not in self.variable_names:
                invalid_vars = [v for v in [t, target] if v not in self.variable_names]
                report.append(f"- Experiment {t} -> {target}: Skipped (Variables not found: {', '.join(invalid_vars)})")
                continue

            if z:
                # 过滤 conditioning_set，只保留存在的变量
                valid_z = [v for v in z if v in self.variable_names]
                res = self.test_conditional_independence(t, target, valid_z)
            else:
                res = self.test_intervention_effect(t, target)
            
            if "error" in res:
                report.append(f"- Experiment {t} -> {target}: Error - {res['error']}")
            else:
                report.append(f"- {res['description']}")
                
        return "\n".join(report)


class EvidencePolicyVerifier:
    """
    强制执行“证据优先”策略的验证器
    """
    def __init__(self):
        self.evidence_cache = {} # (treatment, target) -> {is_significant: bool, metric: float, is_conditional: bool, original_text: str}

    def update_evidence(self, report: str):
        """解析实验报告并更新缓存"""
        if not report:
            return
            
        lines = report.split('\n')
        for line in lines:
            # 匹配模式: RESULT: [SIGNIFICANT] do(X) affects Y (TVD=0.1234)
            # 或者: RESULT: [NOT SIGNIFICANT] do(X) affects Y (TVD=0.0123)
            # 或者条件干预: RESULT: [SIGNIFICANT] Given ['Z'], do(X) affects Y (avg_TVD=0.1234)
            
            sig_match = re.search(r'RESULT: \[(SIGNIFICANT|NOT SIGNIFICANT)\]', line)
            do_match = re.search(r'do\((\w+)\) affects (\w+)', line)
            tvd_match = re.search(r'(?:TVD|avg_TVD|p|p-value)=([\d.]+)', line)
            
            if sig_match and do_match:
                is_significant = sig_match.group(1) == "SIGNIFICANT"
                treatment = do_match.group(1)
                target = do_match.group(2)
                metric = float(tvd_match.group(1)) if tvd_match else None
                is_conditional = "Given" in line or "condition on" in line
                
                key = (treatment, target)
                # 只有当新证据是条件性的（更强），或者之前没有该路径证据时才更新
                if key not in self.evidence_cache or is_conditional:
                    self.evidence_cache[key] = {
                        "is_significant": is_significant,
                        "metric": metric,
                        "is_conditional": is_conditional,
                        "original_text": line.strip()
                    }

    def verify_operations(self, operations: List[Dict]) -> List[str]:
        """验证操作是否违反证据优先策略"""
        if operations is None:
            return []
        violations = []
        for op in operations:
            op_type = op.get('type')
            p = op.get('parent')
            c = op.get('child')
            
            evidence = self.evidence_cache.get((p, c))
            if not evidence:
                continue
                
            is_sig = evidence["is_significant"]
            is_cond = evidence["is_conditional"]
            
            if op_type == 'DELETE':
                # 规则：严禁删除具有显著边缘干预效应的边（除非是条件干预证明了它是间接的）
                # 注意：如果是条件干预结果为不显著，则不违反删除规则
                if is_sig and not is_cond:
                    violations.append(
                        f"STRICT VIOLATION: Cannot DELETE {p} → {c}. "
                        f"Interventional evidence shows a SIGNIFICANT effect ({evidence['original_text']}). "
                        f"This link must be preserved unless you can prove it is indirect via conditional testing."
                    )
                elif is_sig and is_cond:
                    # 如果条件干预仍然显著，说明不是完全中介，通常也不应删除
                    violations.append(
                        f"STRICT VIOLATION: Cannot DELETE {p} → {c}. "
                        f"Even after conditioning, the effect remains SIGNIFICANT ({evidence['original_text']})."
                    )
            
            elif op_type == 'ADD':
                # 规则：禁止添加明确显示没有干预效应的边
                if not is_sig:
                    violations.append(
                        f"STRICT VIOLATION: Cannot ADD {p} → {c}. "
                        f"Interventional evidence shows NO significant effect ({evidence['original_text']})."
                    )
            
            elif op_type == 'REVERSE':
                # REVERSE p->c 意味着我们要建立 c->p 并删除 p->c
                # 1. 检查删除 p->c 是否合规 (显著则不能删)
                if is_sig and not is_cond:
                    violations.append(
                        f"STRICT VIOLATION: Cannot REVERSE {p} → {c}. "
                        f"The current direction {p} → {c} is supported by SIGNIFICANT evidence ({evidence['original_text']}). "
                        f"Reversing it would involve deleting a significant link."
                    )
                
                # 2. 检查添加 c->p 是否合规 (c->p 必须有证据支持或至少不被反对)
                rev_evidence = self.evidence_cache.get((c, p))
                if rev_evidence and not rev_evidence["is_significant"]:
                    violations.append(
                        f"STRICT VIOLATION: Cannot REVERSE {p} → {c}. "
                        f"The proposed direction {c} → {p} is explicitly contradicted by evidence ({rev_evidence['original_text']})."
                    )

        return violations

