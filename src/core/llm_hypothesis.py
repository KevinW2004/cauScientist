import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

from llm_loader import LLMLoader
from utils import ConfigManager
from utils import visualize_causal_graph
from schemas.causal_graph import *
from utils.llm import construct_initial_prompt, construct_system_prompt, construct_local_amendment_prompt

class LLMHypothesisGenerator:
    """
    LLM假设生成器 - 使用统一的 LLMLoader 接口
    """
    
    def __init__(self, llm_loader: LLMLoader):
        self.llm_loader = llm_loader
        self.config = ConfigManager()
        
    
    def generate_next_hypothesis(
        self, 
        variable_list: List[str],
        domain_name: str,
        domain_context: str = "",
        previous_graph: Optional[StructuredGraph] = None,
        memory: Optional[str] = None,
        iteration: int = 0,
        num_edge_operations: int = 3
    ) -> StructuredGraph | None:
        """
        生成下一步因果图修改假设
        
        Args:
            variable_list: 变量列表
            domain_name: 领域名称
            domain_context: 领域背景知识
            previous_graph: 上一轮的因果图
            memory: 记忆(上一轮的反馈)
            iteration: 当前迭代轮次
            num_edge_operations: 允许提出的最大操作数
            
        Returns:
            结构化的因果图字典 | None
        """

        if previous_graph is None:
            raise ValueError("previous_graph must not be None in local amendment")
        print(f"\n[Iteration {iteration}] Performing LOCAL amendment (n={num_edge_operations})...")
        return self._local_amendment(
            variable_list, domain_name, domain_context,
            previous_graph, memory, iteration, num_edge_operations
        )
    
    def generate_initial_hypothesis(
        self,
        variable_list: List[str],
        domain_name: str,
        domain_context: str,
    ) -> StructuredGraph | None:
        """生成初始因果图假设"""
        system_prompt = construct_system_prompt(domain_name)
        user_prompt = construct_initial_prompt(
            variable_list, domain_name, domain_context
        )
        
        # 调用LLM
        response_text = self._call_llm(
            system_prompt, user_prompt
        )
        
        # 解析并标准化
        causal_graph = self._parse_and_normalize_response(response_text, variable_list)
        
        # 创建结构化图
        structured_graph: StructuredGraph | None = self._create_structured_graph(
            causal_graph, variable_list, domain_name, iteration=0
        )
        
        return structured_graph
    
    
    def _local_amendment(
        self,
        variable_list: List[str],
        domain_name: str,
        domain_context: str,
        previous_graph: StructuredGraph,
        memory: Optional[str],
        iteration: int,
        num_edge_operations: int = 3
    ) -> StructuredGraph | None:
        """
        局部修正：让模型选择对边进行操作（添加、删除、反转）
        
        Args:
            num_edge_operations: 最大操作边数（LLM可以选择少于这个数量的操作），默认为3
        """
        
        system_prompt = construct_system_prompt(domain_name)
        user_prompt = construct_local_amendment_prompt(
            variable_list, domain_name, domain_context,
            previous_graph, memory, num_edge_operations
        )
        
        # 调用LLM
        response_text = self._call_llm(
            system_prompt, user_prompt
        )
        
        # 解析操作指令
        operations = self._parse_edge_operations(response_text)
        
        # 应用操作到上一轮的图上
        updated_graph = self._apply_edge_operations(
            previous_graph, operations, variable_list
        )
        
        # 创建结构化图
        structured_graph = self._create_structured_graph(
            updated_graph, variable_list, domain_name, iteration, previous_graph
        )
        
        return structured_graph
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        调用LLM并返回响应文本
        """
        temper = self.config.get("training.temperature", 0.7)
        return self.llm_loader.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temper
        )


    def _parse_and_normalize_response(
        self, 
        response_text: str, 
        variable_list: List[str]
    ) -> Dict:
        """
        解析LLM响应并标准化为统一格式
        
        支持多种格式：
        1. 标准列表: {"nodes": [{"name": "X", "parents": [...]}, ...], "reasoning": "..."}
        2. 父->子字典: {"nodes": {"Parent": ["Child1", "Child2"]}, "reasoning": "..."}
        3. 纯文本描述（尽力解析）
        
        Returns:
            标准化的图字典: {"nodes": [{"name": "X", "parents": [...]}], "reasoning": "..."}
        """
        
        # print(f"Raw LLM response (first 500 chars):\n{response_text[:500]}\n")
        
        # 步骤1: 提取JSON
        json_obj = self._extract_json(response_text)
        
        if json_obj is None:
            print("⚠️  Failed to extract JSON.")
            exit()
            # json_obj = self._parse_text_to_graph(response_text, variable_list)
        
        # 步骤2: 标准化格式
        normalized_graph = self._normalize_graph_structure(json_obj, variable_list)

        print(f"Reasoning from LLM: {normalized_graph.get('reasoning', '')}\n")
        
        return normalized_graph
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON对象"""
        
        def remove_json_comments(json_str: str) -> str:
            """移除JSON字符串中的单行注释 (// ...)"""
            # 移除 // 开头的单行注释（但不移除字符串内的 //）
            # 简单策略：移除 // 到行尾的内容
            lines = json_str.split('\n')
            cleaned_lines = []
            for line in lines:
                # 查找 // 的位置，但要小心字符串内的 //
                # 简单处理：如果行中有 //，且不在引号内（粗略检查）
                comment_idx = line.find('//')
                if comment_idx != -1:
                    # 检查 // 前面是否有偶数个引号（说明 // 不在字符串内）
                    before_comment = line[:comment_idx]
                    quote_count = before_comment.count('"') - before_comment.count('\\"')
                    if quote_count % 2 == 0:  # 偶数个引号，说明 // 在字符串外
                        line = line[:comment_idx].rstrip()
                cleaned_lines.append(line)
            return '\n'.join(cleaned_lines)
        
        # 方法1: 直接解析整个文本
        try:
            cleaned_text = remove_json_comments(text)
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
        
        # 方法2: 提取```json ... ```代码块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                cleaned_text = remove_json_comments(json_match.group(1))
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
        
        # 方法3: 提取``` ... ```代码块（无json标记）
        code_match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            try:
                cleaned_text = remove_json_comments(code_match.group(1))
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
        
        # 方法4: 查找第一个 { 到最后一个 }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                json_candidate = text[start_idx:end_idx+1]
                cleaned_text = remove_json_comments(json_candidate)
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _parse_text_to_graph(self, text: str, variable_list: List[str]) -> Dict:
        """
        从纯文本描述中解析因果图（备用方案）
        查找类似 "X → Y" 或 "X causes Y" 的模式
        """
        
        edges = []
        
        # 模式1: X → Y 或 X -> Y
        arrow_pattern = r'(\w+)\s*[-–→>]+\s*(\w+)'
        for match in re.finditer(arrow_pattern, text):
            parent, child = match.groups()
            if parent in variable_list and child in variable_list:
                edges.append((parent, child))
        
        # 模式2: "Y has parents X, Z" 或 "Y depends on X"
        parent_pattern = r'(\w+)\s+(?:has parents?|depends? on|caused by)\s+([^.;]+)'
        for match in re.finditer(parent_pattern, text, re.IGNORECASE):
            child, parents_str = match.groups()
            if child in variable_list:
                for var in variable_list:
                    if var in parents_str and var != child:
                        edges.append((var, child))
        
        # 构建图
        node_parents = {var: [] for var in variable_list}
        for parent, child in edges:
            if parent not in node_parents[child]:
                node_parents[child].append(parent)
        
        nodes = [{'name': var, 'parents': parents} for var, parents in node_parents.items()]
        
        return {
            'nodes': nodes,
            'reasoning': 'Parsed from text description (fallback method)'
        }
    
    def _normalize_graph_structure(self, json_obj: Dict, variable_list: List[str]) -> Dict:
        """
        将各种格式的图转换为标准格式
        
        标准格式: {"nodes": [{"name": "X", "parents": [...]}, ...], "reasoning": "..."}
        """
        
        if 'nodes' not in json_obj:
            raise ValueError("JSON must contain 'nodes' field")
        
        nodes_data = json_obj['nodes']
        reasoning = json_obj.get('reasoning', '')
        
        # 情况1: nodes 已经是列表
        if isinstance(nodes_data, list):
            normalized_nodes = self._normalize_node_list(nodes_data)
        
        # 情况2: nodes 是字典
        elif isinstance(nodes_data, dict):
            normalized_nodes = self._normalize_node_dict(nodes_data, variable_list)
        
        else:
            raise ValueError(f"'nodes' must be list or dict, got {type(nodes_data)}")
        
        return {
            'nodes': normalized_nodes,
            'reasoning': reasoning
        }
    
    def _normalize_node_list(self, nodes: List) -> List[Dict]:
        """标准化节点列表格式"""
        
        normalized = []
        
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError(f"Each node must be a dict, got {type(node)}")
            
            # 提取变量名（支持 'name' 或 'variable'）
            var_name = node.get('name') or node.get('variable')
            if not var_name:
                raise ValueError(f"Node missing 'name' or 'variable': {node}")
            
            # 提取父节点
            parents = node.get('parents', [])
            if not isinstance(parents, list):
                parents = []
            
            normalized_node = {
                'name': var_name,
                'parents': parents
            }
            
            normalized.append(normalized_node)
        
        return normalized
    
    def _normalize_node_dict(self, nodes_dict: Dict, variable_list: List[str]) -> List[Dict]:
        """
        标准化节点字典格式
        
        支持两种格式：
        1. 父->子: {"Parent": ["Child1", "Child2"]}  (需要反转)
        2. 子->父: {"Child": {"parents": ["Parent1"]}}
        """
        
        # 检查是哪种格式
        first_key = next(iter(nodes_dict.keys()), None)
        if first_key is None:
            return []
        
        first_value = nodes_dict[first_key]
        
        # 格式1: 值是列表 -> 父->子格式（需要反转）
        if isinstance(first_value, list):
            return self._convert_parent_to_child_dict(nodes_dict, variable_list)
        
        # 格式2: 值是字典 -> 子->父格式
        elif isinstance(first_value, dict):
            normalized = []
            for var_name, node_info in nodes_dict.items():
                normalized_node = {
                    'name': var_name,
                    'parents': node_info.get('parents', [])
                }
                
                normalized.append(normalized_node)
            
            return normalized
        
        else:
            raise ValueError(f"Unsupported dict value type: {type(first_value)}")
    
    def _convert_parent_to_child_dict(
        self, 
        parent_to_children: Dict[str, List[str]], 
        variable_list: List[str]
    ) -> List[Dict]:
        """
        将"父->子"字典转换为标准节点列表
        
        输入: {"Parent": ["Child1", "Child2"], ...}
        输出: [{"name": "Child1", "parents": ["Parent"]}, ...]
        """
        
        # 收集所有节点
        all_nodes = set(variable_list)
        
        # 构建每个节点的父节点列表
        node_parents = {node: [] for node in all_nodes}
        
        for parent, children in parent_to_children.items():
            if parent not in all_nodes:
                print(f"⚠️  Warning: Parent '{parent}' not in variable list")
                continue
            
            for child in children:
                if child not in all_nodes:
                    print(f"⚠️  Warning: Child '{child}' not in variable list")
                    continue
                
                if parent not in node_parents[child]:
                    node_parents[child].append(parent)
        
        # 创建节点列表
        nodes = [
            {'name': var_name, 'parents': parents}
            for var_name, parents in node_parents.items()
        ]
        
        return nodes
    
    def _create_structured_graph(
        self,
        causal_graph: Dict,
        variable_list: List[str],
        domain_name: str,
        iteration: int,
        previous_graph: Optional[StructuredGraph] = None
    ) -> StructuredGraph | None:
        """创建最终的结构化图表示, None 表示无效图"""
        
        nodes = causal_graph['nodes']
        reasoning = causal_graph['reasoning']
        
        # 验证变量完整性
        graph_vars = {node['name'] for node in nodes}
        expected_vars = set(variable_list)
        
        if graph_vars != expected_vars:
            missing = expected_vars - graph_vars
            extra = graph_vars - expected_vars
            
            if missing:
                print(f"⚠️  Warning: Missing variables: {missing}")
                # 添加缺失的变量（无父节点）
                for var in missing:
                    nodes.append({'name': var, 'parents': []})
            
            if extra:
                print(f"⚠️  Warning: Extra variables (will be removed): {extra}")
                # 移除多余的变量
                nodes = [n for n in nodes if n['name'] in expected_vars]
        
        # 清理无效的父节点（不在变量列表中的父节点）
        for node in nodes:
            valid_parents = []
            invalid_parents = []
            for parent in node.get('parents', []):
                if parent in expected_vars:
                    valid_parents.append(parent)
                else:
                    invalid_parents.append(parent)
            
            if invalid_parents:
                print(f"⚠️  Warning: Node '{node['name']}' has invalid parents (not in variable list): {invalid_parents}")
                print(f"    These parents will be removed.")
            
            node['parents'] = valid_parents
        
        # 检查环
        # print("begin checking cycles")
        cycles = self._has_cycle(nodes)
        # print("end checking cycles")
        if cycles:
            # print("⚠️  Warning: Graph contains cycles! Attempting to break cycles...")
            # nodes = self._break_cycles(nodes)
            print("⚠️  Warning: Graph contains cycles! Return None")
            return None
        
        # 创建返回对象

        # 计算变化
        changes = None
        if previous_graph is not None:
            changes = self._compute_changes(previous_graph, nodes)
        change_obj = None
        if changes:
            change_obj = GraphChanges(
                added_edges=changes['added_edges'],
                removed_edges=changes['removed_edges'],
                num_added=changes['num_added'],
                num_removed=changes['num_removed']
            )
        metadata_obj = GraphMetadata(
            domain=domain_name,
            iteration=iteration,
            num_variables=len(variable_list),
            num_edges=self._count_edges(nodes),
            reasoning=reasoning,
            changes=change_obj
        )
        nodes_objs = [CausalNode(name=node['name'], parents=node['parents']) for node in nodes]
        
        # 计算邻接矩阵
        adj_matrix, _ = self._create_adjacency_matrix(nodes, variable_list)
        
        # 组装
        structured_graph = StructuredGraph(
            metadata=metadata_obj,
            nodes=nodes_objs,
            adjacency_matrix=adj_matrix
        )
        
        return structured_graph
    
    def _has_cycle(self, nodes: List[Dict]) -> bool:
        """检查是否有环（DFS算法）"""
        
        # 构建邻接表
        graph = defaultdict(list)
        all_nodes = set()
        
        for node in nodes:
            node_name = node['name']
            all_nodes.add(node_name)
            
            for parent in node.get('parents', []):
                all_nodes.add(parent)
                graph[parent].append(node_name)
        
        # DFS检测环
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in all_nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _break_cycles(self, nodes: List[Dict]) -> List[Dict]:
        """
        打破环（简单策略：移除导致环的边）
        """
        print("  Attempting to break cycles by removing edges...")
        
        # 尝试移除每条边，看是否能消除环
        for node in nodes:
            if len(node['parents']) > 0:
                original_parents = node['parents'].copy()
                
                for parent in original_parents:
                    # 临时移除这条边
                    node['parents'].remove(parent)
                    
                    # 检查是否还有环
                    if not self._has_cycle(nodes):
                        print(f"  Removed edge: {parent} → {node['name']}")
                        continue  # 成功移除，继续
                    else:
                        # 还原边
                        node['parents'].append(parent)
        
        # 最终检查
        if self._has_cycle(nodes):
            print("  ⚠️  Warning: Could not break all cycles!")
        else:
            print("  ✓ All cycles broken")
        
        return nodes
    
    def _compute_changes(self, prev_graph: StructuredGraph, curr_nodes: List[Dict]) -> Dict:
        """计算图之间的变化"""
        
        def get_edges_from_nodes(nodes):
            edges = set()
            for node in nodes:
                child = node['name']
                for parent in node.get('parents', []):
                    edges.add((parent, child))
            return edges
        
        def get_edges_from_graph(graph: StructuredGraph):
            edges = set()
            for node in graph.nodes:
                child = node.name
                for parent in node.parents:
                    edges.add((parent, child))
            return edges
        
        prev_edges = get_edges_from_graph(prev_graph)
        curr_edges = get_edges_from_nodes(curr_nodes)
        
        added = curr_edges - prev_edges
        removed = prev_edges - curr_edges
        
        return {
            "added_edges": list(added),
            "removed_edges": list(removed),
            "num_added": len(added),
            "num_removed": len(removed)
        }
    
    def _count_edges(self, nodes: List[Dict]) -> int:
        """计算边数"""
        return sum(len(node.get('parents', [])) for node in nodes)
    
    def _create_adjacency_matrix(
        self,
        nodes: List[Dict],
        variable_list: List[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """创建邻接矩阵"""
        
        n = len(variable_list)
        adjacency_matrix = np.zeros((n, n), dtype=int)
        var_to_idx = {var: idx for idx, var in enumerate(variable_list)}
        
        for node in nodes:
            child_name = node['name']
            if child_name not in var_to_idx:
                continue
            
            child_idx = var_to_idx[child_name]
            for parent_name in node.get('parents', []):
                if parent_name in var_to_idx:
                    parent_idx = var_to_idx[parent_name]
                    adjacency_matrix[parent_idx, child_idx] = 1
        
        df = pd.DataFrame(adjacency_matrix, index=variable_list, columns=variable_list)
        return adjacency_matrix, df
    
    def _parse_edge_operations(self, response_text: str) -> List[Dict]:
        """
        解析LLM返回的边操作指令
        
        Returns:
            操作列表，每个操作包含: type, parent, child, reasoning
        """
        # print(f"Raw operations response (first 500 chars):\n{response_text[:500]}\n")
        
        # 提取JSON
        json_obj = self._extract_json(response_text)
        
        if json_obj is None:
            print("⚠️  Failed to extract operations JSON. Using empty operations.")
            return []
        
        operations = json_obj.get('operations', [])
        
        if not isinstance(operations, list):
            print(f"⚠️  'operations' must be a list, got {type(operations)}")
            return []
        
        # 验证每个操作
        valid_operations = []
        for op in operations:
            if not isinstance(op, dict):
                print(f"⚠️  Skipping invalid operation (not a dict): {op}")
                continue
            
            op_type = op.get('type', '').upper()
            parent = op.get('parent')
            child = op.get('child')
            reasoning = op.get('reasoning', '')
            
            if op_type not in ['ADD', 'DELETE', 'REVERSE']:
                print(f"⚠️  Skipping operation with invalid type: {op_type}")
                continue
            
            if not parent or not child:
                print(f"⚠️  Skipping operation missing parent or child: {op}")
                continue
            
            valid_operations.append({
                'type': op_type,
                'parent': parent,
                'child': child,
                'reasoning': reasoning
            })
        
        print(f"✓ Parsed {len(valid_operations)} valid operations")
        for i, op in enumerate(valid_operations, 1):
            print(f"  {i}. {op['type']}: {op['parent']} → {op['child']}")
        
        return valid_operations
    
    def _apply_edge_operations(
        self,
        previous_graph: StructuredGraph,
        operations: List[Dict],
        variable_list: List[str]
    ) -> Dict:
        """
        将边操作应用到上一轮的图上
        
        Args:
            previous_graph: 上一轮的图结构
            operations: 操作列表
            variable_list: 变量列表
            
        Returns:
            更新后的图（nodes格式）
        """
        # 复制节点数据
        nodes = []
        for node in previous_graph.nodes:
            nodes.append({
                'name': node.name,
                'parents': node.parents.copy()
            })
        
        # 创建名称到节点的映射
        node_map = {node['name']: node for node in nodes}
        
        # 确保所有变量都在图中
        for var in variable_list:
            if var not in node_map:
                new_node = {'name': var, 'parents': []}
                nodes.append(new_node)
                node_map[var] = new_node
        
        # 应用每个操作
        for op in operations:
            op_type = op['type']
            parent = op['parent']
            child = op['child']
            
            # 验证变量存在
            if parent not in variable_list or child not in variable_list:
                print(f"⚠️  Skipping operation with invalid variables: {parent} → {child}")
                continue
            
            if parent == child:
                print(f"⚠️  Skipping self-loop: {parent} → {child}")
                continue
            
            child_node = node_map[child]
            
            if op_type == 'ADD':
                # 添加边
                if parent not in child_node['parents']:
                    child_node['parents'].append(parent)
                    print(f"  ✓ Added edge: {parent} → {child}")
                else:
                    print(f"  ⚠️  Edge already exists: {parent} → {child}")
            
            elif op_type == 'DELETE':
                # 删除边
                if parent in child_node['parents']:
                    child_node['parents'].remove(parent)
                    print(f"  ✓ Deleted edge: {parent} → {child}")
                else:
                    print(f"  ⚠️  Edge doesn't exist: {parent} → {child}")
            
            elif op_type == 'REVERSE':
                # 反转边: 删除 parent → child，添加 child → parent
                if parent in child_node['parents']:
                    child_node['parents'].remove(parent)
                    parent_node = node_map[parent]
                    if child not in parent_node['parents']:
                        parent_node['parents'].append(child)
                        print(f"  ✓ Reversed edge: {parent} → {child} to {child} → {parent}")
                    else:
                        print(f"  ⚠️  Cannot reverse: would create duplicate edge")
                else:
                    print(f"  ⚠️  Cannot reverse non-existent edge: {parent} → {child}")
        
        # 返回标准化的图格式
        return {
            'nodes': nodes,
            'reasoning': f"Applied {len(operations)} local operations"
        }
    
    def visualize_graph(
        self, 
        structured_graph: StructuredGraph,
        output_dir: str = "visualizations",
        previous_graph: Optional[StructuredGraph] = None,
        auto_open: bool = True,
        text_only: bool = False
    ):
        """
        可视化因果图（支持文本和交互式HTML两种方式）
        
        Args:
            structured_graph: 结构化图数据（StructuredGraph schema）
            output_dir: HTML输出目录
            previous_graph: 上一轮的图（用于高亮变化）
            auto_open: 是否自动在浏览器打开HTML
            text_only: 是否仅输出文本（不生成HTML）
        """
        # 文本可视化
        print("\n" + "="*60)
        print(f"CAUSAL GRAPH - {structured_graph.metadata.domain.upper()}")
        print("="*60)
        print(f"Iteration: {structured_graph.metadata.iteration}")
        print(f"Variables: {structured_graph.metadata.num_variables}")
        print(f"Edges: {structured_graph.metadata.num_edges}")
        
        # 显示变化
        if structured_graph.metadata.changes:
            changes = structured_graph.metadata.changes
            print(f"\nChanges from previous iteration:")
            print(f"  Added: {changes.num_added} edges")
            print(f"  Removed: {changes.num_removed} edges")
            
            if changes.added_edges:
                for parent, child in changes.added_edges:
                    print(f"  + {parent} → {child}")
            if changes.removed_edges:
                for parent, child in changes.removed_edges:
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
        
        for node in structured_graph.nodes:
            parents = node.parents
            if parents:
                for parent in parents:
                    edges.append(f"  {parent} → {node.name}")
            else:
                root_nodes.append(node.name)
        
        if root_nodes:
            print("\nRoot Nodes (no parents):")
            for node in root_nodes:
                print(f"  • {node}")
        
        if edges:
            print("\nCausal Edges:")
            for edge in sorted(edges):
                print(edge)
        
        print("="*60 + "\n")
        
        # 交互式HTML可视化
        if not text_only:
            try:
                visualize_causal_graph(
                    structured_graph=structured_graph,
                    output_dir=output_dir,
                    previous_graph=previous_graph,
                    auto_open=auto_open,
                    layout="hierarchical"
                )
            except Exception as e:
                print(f"⚠️  警告: 无法生成交互式可视化: {e}")
                print(f"   (你可能需要安装 pyvis: pip install pyvis networkx)")