"""
LLM Hypothesis Generation Module
"""

import json
from typing import Dict, List, Tuple, Optional
from utils.llm.parse_response import parse_and_normalize_response, parse_edge_operations, apply_edge_operations, create_structured_graph
from utils.llm.prompts import construct_system_prompt, construct_initial_prompt, construct_amendment_prompt, construct_local_amendment_prompt, construct_experiment_proposal_prompt
from utils.config_manager import ConfigManager
from llm_loader import LLMLoader


class LLMHypothesisGenerator:
    """
    LLM假设生成器 - 使用统一的 LLMLoader 接口
    """
    
    def __init__(self, llm_loader: LLMLoader):
        self.llm_loader = llm_loader
        self.config = ConfigManager()

    def propose_experiments(
        self,
        variable_list: List[str],
        domain_name: str,
        domain_context: str = "",
        previous_graph: Optional[Dict] = None,
        edge_notes: Dict[str, str] = None
    ) -> Tuple[List[Dict], str, List[Dict]]:
        """
        让 LLM 提出干预实验建议和候选操作
        返回: (experiments_list, reasoning_string, candidate_operations)
        """
        system_prompt = construct_system_prompt(domain_name)
        user_prompt = construct_experiment_proposal_prompt(
            variable_list, domain_name, domain_context, previous_graph, num_experiments,
            edge_notes=edge_notes
        )
        
        print(f"\n[Intervention] Requesting candidates and experiments from LLM...")
        response_text = self._call_llm(system_prompt, user_prompt)
        
        num_experiments = self.config.get("experiment.num_experiments", 5)

        try:
            # 解析 JSON 响应
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                experiments = data.get("experiments", [])
                candidate_operations = data.get("candidate_operations", [])
                reasoning = data.get("reasoning", "")
                
                if candidate_operations:
                    print(f"  📋 Proposed {len(candidate_operations)} candidate operations")
                if reasoning:
                    print(f"  🧠 Reasoning: {reasoning[:200]}...")
                    
                return experiments[:num_experiments], reasoning, candidate_operations
        except Exception as e:
            print(f"❌ Failed to parse experiment proposal: {e}")
            
        return [], "", []

    def generate_hypothesis(
        self, 
        variable_list: List[str],
        domain_name: str,
        domain_context: str = "",
        previous_graph: Optional[Dict] = None,
        memory: Optional[str] = None,
        iteration: int = 0,
        num_edge_operations: int = 3,
        skeleton_constraints: Optional[Dict] = None,
        failed_attempts: Optional[List[Dict]] = None,
        baseline_reference: Optional[str] = None,
        interventional_evidence: Optional[str] = None,
        previous_reasoning: Optional[str] = None,
        confirmed_edges: List[str] = None,
        edge_notes: Dict[str, str] = None,
        candidate_operations: List[Dict] = None
    ) -> tuple:
        """
        生成因果图假设
        
        Args:
            variable_list: 变量列表
            domain_name: 领域名称
            domain_context: 领域背景知识
            previous_graph: 上一轮的因果图
            memory: 记忆(上一轮的反馈)
            iteration: 当前迭代次数
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大token数
            use_local_amendment: 是否使用局部修正（而非全局修正）
            num_edge_operations: 局部修正时操作的边数
            skeleton_constraints: 骨架约束
            failed_attempts: 失败的初始尝试历史
            baseline_reference: 传统方法的参考信息
            interventional_evidence: 干预实验的统计结果
            previous_reasoning: 上一轮的推理逻辑或实验动机
            confirmed_edges: 已确认存在的边列表
            edge_notes: 对特定边的推理笔记
            candidate_operations: 局部修正阶段限定的操作候选
        """
        
        is_initial = (iteration == 0) or (previous_graph is None)

        system_prompt = construct_system_prompt(domain_name)
        if previous_graph is None: # TODO:
            response_type = "global"
            valid_ops=None
            if is_initial:
                print(f"\n[Iteration {iteration}] Generating INITIAL hypothesis...")
                if baseline_reference:
                    print(f"  📊 Using statistical baseline reference")
                user_prompt = construct_initial_prompt(
                    variable_list, domain_name, domain_context,
                    skeleton_constraints,
                    failed_attempts,
                    baseline_reference,
                    interventional_evidence
                )
            else:
                print(f"\n[Iteration {iteration}] Performing GLOBAL amendment...")
                user_prompt = construct_amendment_prompt(
                    variable_list, domain_name, domain_context,
                    previous_graph, memory,
                    interventional_evidence,
                    previous_reasoning,
                    confirmed_edges,
                    edge_notes
                )
        else:
            print(f"\n[Iteration {iteration}] Performing LOCAL amendment (n={num_edge_operations})...")
            response_type = "local"
            valid_ops = self._get_valid_operations(previous_graph, variable_list, exclude_operations=failed_attempts, skeleton_constraints=skeleton_constraints)
            user_prompt = construct_local_amendment_prompt(
                variable_list, domain_name, domain_context,
                previous_graph, memory, num_edge_operations, exclude_operations=failed_attempts,
                skeleton_constraints=skeleton_constraints, 
                interventional_evidence=interventional_evidence,
                previous_reasoning=previous_reasoning,
                confirmed_edges=confirmed_edges,
                edge_notes=edge_notes,
                candidate_operations=candidate_operations
                )

        print(user_prompt)
        response_text = self._call_llm(system_prompt, user_prompt)
        print(response_text)
        structured_graph, validation_info = self.parse_response_to_graph(response_text, variable_list, response_type, previous_graph, domain_name, iteration, num_edge_operations=num_edge_operations, valid_ops=valid_ops)
        
        # 保存prompt信息（仅当图创建成功时）
        if structured_graph is not None:
            structured_graph['metadata']['prompts'] = {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'llm_response': response_text
            }
        # 返回结果
        return structured_graph, validation_info
    
    def parse_response_to_graph(self, response_text: str, variable_list: List[str], response_type: str, previous_graph: Optional[Dict] = None, domain_name: str = None, iteration: int = 0, num_edge_operations: int = 1, valid_ops: Optional[Dict] = None) -> Dict:
        if response_type == "global":

            causal_graph, validation_info = parse_and_normalize_response(response_text, variable_list)
            if causal_graph is None:
                return None, validation_info
            structured_graph, validation_info = create_structured_graph(
                causal_graph, variable_list, domain_name, iteration, previous_graph, validation_info
            )
        elif response_type == "local":
            operations, overall_reasoning, confirmed_edges, edge_notes = parse_edge_operations(response_text, num_edge_operations=num_edge_operations)
            proposed_operations = operations.copy() if operations else []
            
            # 先应用操作并获取基础图结构
            causal_graph, validation_info = apply_edge_operations(
                    previous_graph, operations, variable_list, valid_ops=valid_ops
                )
            
            # 只有当 validation_info 标记为成功时（包括空操作），才更新档案和推理
            # 这样可以防止解析彻底失败或严重错误操作时，错误的档案被保存
            if validation_info.get('success', False):
                causal_graph['reasoning'] = overall_reasoning
                causal_graph['confirmed_edges'] = confirmed_edges
                causal_graph['edge_notes'] = edge_notes
            
            structured_graph, validation_info = create_structured_graph(
                causal_graph, variable_list, domain_name, iteration, previous_graph, validation_info,
                proposed_operations
            )
        return structured_graph, validation_info
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        调用LLM并返回响应文本
        """
        temperature = self.config.get("training.temperature", 0.7)
        return self.llm_loader.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
    
    
    def _get_valid_operations(
        self,
        previous_graph: Dict,
        variable_list: List[str],
        exclude_operations: List[tuple] = None,
        skeleton_constraints: Optional[Dict] = None
    ) -> Dict[str, List[tuple]]:
        """
        计算所有合法的边操作
        
        Args:
            exclude_operations: 要排除的操作列表 [(op_type, parent, child), ...]
            skeleton_constraints: 骨架约束 {'allowed_edges': [(p, c), ...], 'forbidden_pairs': [...]}
        
        Returns:
            {
                'can_add': [(parent, child), ...],      # 可以添加的边（不存在且不会创环）
                'can_delete': [(parent, child), ...],   # 可以删除的边（存在的边）
                'can_reverse': [(parent, child), ...]   # 可以反转的边（存在且反转不会创环）
            }
        """
        if exclude_operations is None:
            exclude_operations = []
        
        # 构建排除集合，方便快速查找
        excluded_adds = {(p, c) for op_type, p, c in exclude_operations if op_type == 'ADD'}
        excluded_deletes = {(p, c) for op_type, p, c in exclude_operations if op_type == 'DELETE'}
        excluded_reverses = {(p, c) for op_type, p, c in exclude_operations if op_type == 'REVERSE'}
        
        # 如果有骨架约束，构建允许的边集合
        allowed_edges_set = None
        if skeleton_constraints:
            allowed_edges_set = set(skeleton_constraints.get('allowed_edges', []))
        
        nodes = previous_graph['nodes']
        
        # 构建当前边的集合
        existing_edges = set()
        node_parents = {}
        for node in nodes:
            node_name = node['name']
            parents = node.get('parents', [])
            node_parents[node_name] = parents.copy()
            for parent in parents:
                existing_edges.add((parent, node_name))
        
        # 确保所有变量都在node_parents中
        for var in variable_list:
            if var not in node_parents:
                node_parents[var] = []
        
        # 1. 可删除的边：所有存在的边（排除已失败的）
        can_delete = [edge for edge in existing_edges if edge not in excluded_deletes]
        
        # 2. 可添加的边：不存在且不会创环的边（排除已失败的，应用骨架约束）
        can_add = []
        for parent in variable_list:
            for child in variable_list:
                if parent == child:
                    continue  # 跳过自环
                if (parent, child) in existing_edges:
                    continue  # 跳过已存在的边
                if (parent, child) in excluded_adds:
                    continue  # 跳过已失败的操作
                
                # 应用骨架约束
                if allowed_edges_set is not None and (parent, child) not in allowed_edges_set:
                    continue  # 跳过不在骨架中的边
                
                # 检查添加这条边是否会创环
                # 方法：检查是否已经存在从child到parent的路径
                if not self._would_create_cycle(child, parent, node_parents):
                    can_add.append((parent, child))
        
        # 3. 可反转的边：存在且反转后不会创环的边（排除已失败的，应用骨架约束）
        can_reverse = []
        for parent, child in existing_edges:
            if (parent, child) in excluded_reverses:
                continue  # 跳过已失败的操作
            
            # 应用骨架约束：反转后的边也必须在允许列表中
            if allowed_edges_set is not None and (child, parent) not in allowed_edges_set:
                continue  # 跳过反转后不在骨架中的边
            
            # 检查反转后（child → parent）是否会创环
            # 需要暂时移除原边，添加反向边，然后检查
            temp_parents = {k: v.copy() for k, v in node_parents.items()}
            temp_parents[child].remove(parent)  # 移除原边
            temp_parents[parent].append(child)  # 添加反向边
            
            # 检查是否有环
            if not self._has_cycle_in_parents(temp_parents):
                can_reverse.append((parent, child))
        
        return {
            'can_add': can_add,
            'can_delete': can_delete,
            'can_reverse': can_reverse
        }
    
    def _would_create_cycle(
        self,
        start: str,
        target: str,
        node_parents: Dict[str, List[str]]
    ) -> bool:
        """
        检查从start到target是否已经存在路径（BFS）
        
        在图中添加 target→start 边时，如果已经存在 start...→target 的路径，
        则会形成环。
        
        Args:
            start: 路径起点
            target: 路径终点
            node_parents: {node: [parent1, parent2, ...]} 表示 parent→node 的边
        
        Returns:
            True if 存在从start到target的路径（会创建环）
            False if 不存在路径（不会创建环）
        """
        if start == target:
            return True
        
        # 构建正向图：child→parents 转换为 parent→children
        children_map = {}
        for node, parents in node_parents.items():
            for parent in parents:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(node)
        
        # BFS从start开始，沿着边的方向前进
        visited = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current == target:
                return True  # 找到路径，会创建环
            
            # 添加所有子节点到队列（沿着边的方向）
            for child in children_map.get(current, []):
                if child not in visited:
                    queue.append(child)
        
        return False  # 没找到路径，不会创建环
    
    def _has_cycle_in_parents(self, node_parents: Dict[str, List[str]]) -> bool:
        """
        检查给定的父节点关系中是否有环（使用DFS）
        
        Args:
            node_parents: {node: [parent1, parent2, ...]} 表示 parent→node 的边
        
        Returns:
            True if 图中存在环
            False if 图是DAG
        """
        # 构建正向图：parent→children
        children_map = {}
        all_nodes = set(node_parents.keys())
        
        for node, parents in node_parents.items():
            for parent in parents:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(node)
                all_nodes.add(parent)
        
        # 确保所有节点都在children_map中
        for node in all_nodes:
            if node not in children_map:
                children_map[node] = []
        
        # DFS环检测（正向遍历）
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True  # 发现环
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            # 遍历子节点（正向）
            for child in children_map.get(node, []):
                if dfs(child):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # 检查所有节点
        for node in all_nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def visualize_graph(self, structured_graph: Dict):
        """文本可视化因果图"""
        print("\n" + "="*60)
        print(f"CAUSAL GRAPH - {structured_graph['metadata']['domain'].upper()}")
        print("="*60)
        print(f"Iteration: {structured_graph['metadata']['iteration']}")
        print(f"Variables: {structured_graph['metadata']['num_variables']}")
        print(f"Edges: {structured_graph['metadata']['num_edges']}")
        
        # 显示变化
        if structured_graph['metadata'].get('changes'):
            changes = structured_graph['metadata']['changes']
            print(f"\nChanges from previous iteration:")
            print(f"  Added: {changes['num_added']} edges")
            print(f"  Removed: {changes['num_removed']} edges")
            
            if changes['added_edges']:
                for parent, child in changes['added_edges']:
                    print(f"  + {parent} → {child}")
            if changes['removed_edges']:
                for parent, child in changes['removed_edges']:
                    print(f"  - {parent} → {child}")
        
        # print("\nReasoning:")
        # reasoning = structured_graph['metadata']['reasoning']
        # print(reasoning[:300] + "..." if len(reasoning) > 300 else reasoning)
        
        print("\n" + "-"*60)
        print("CAUSAL RELATIONSHIPS:")
        print("-"*60)
        
        # 显示边
        edges = []
        root_nodes = []
        
        for node in structured_graph['nodes']:
            parents = node.get('parents', [])
            if parents:
                for parent in parents:
                    edges.append(f"  {parent} → {node['name']}")
            else:
                root_nodes.append(node['name'])
        
        if root_nodes:
            print("\nRoot Nodes (no parents):")
            for node in root_nodes:
                print(f"  • {node}")
        
        if edges:
            print("\nCausal Edges:")
            for edge in sorted(edges):
                print(edge)
        
        print("="*60 + "\n")