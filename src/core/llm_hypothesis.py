import numpy as np
import pandas as pd
from collections import defaultdict

from llm_loader import LLMLoader
from utils import ConfigManager
from schemas.causal_graph import *
from schemas.causal_graph import GraphChange
from utils.llm import construct_initial_prompt, extract_json, \
    construct_system_prompt, construct_local_amendment_prompt, parse_and_normalize_response
from reflection import ReflectionManager

class LLMHypothesisGenerator:
    """
    LLM假设生成器 - 使用统一的 LLMLoader 接口
    """

    def __init__(self, llm_loader: LLMLoader):
        self.llm_loader = llm_loader
        self.config = ConfigManager()

    def generate_next_hypothesis(
        self, 
        variable_list: list[str],
        domain_name: str,
        domain_context: str = "",
        previous_graph: StructuredGraph | None = None,
        memory: str | None = None,
        iteration: int = 0,
        num_edge_operations: int = 3
    ) -> tuple[list[StructuredGraph], bool]:
        """
        生成下一步因果图修改假设（返回多个候选图）
        
        Args:
            variable_list: 变量列表
            domain_name: 领域名称
            domain_context: 领域背景知识
            previous_graph: 上一轮的因果图
            memory: 记忆(上一轮的反馈)
            iteration: 当前迭代轮次
            num_edge_operations: 允许提出的最大操作数
            
        Returns:
            (结构化的因果图列表, is_final_graph标志)
            is_final_graph=True 表示 LLM 认为 previous_graph 已经足够好，不需要再修改
        """

        if previous_graph is None:
            raise ValueError("previous_graph must not be None in local amendment")
        print(f"\n[Iteration {iteration}] Performing LOCAL amendment (n={num_edge_operations})...")
        reflection = ReflectionManager().current_reflection
        return self._local_amendment(
            variable_list, domain_name, domain_context,
            previous_graph, memory, reflection, iteration, num_edge_operations
        )

    def generate_initial_hypothesis(
        self,
        variable_list: list[str],
        domain_name: str,
        domain_context: str,
    ) -> StructuredGraph | None:
        """生成初始因果图假设"""
        system_prompt = construct_system_prompt(domain_name)
        user_prompt = construct_initial_prompt(
            variable_list, domain_name, domain_context
        )

        # 调用LLM
        response_text = self.llm_loader.generate(system_prompt, user_prompt)

        # 解析并标准化
        causal_graph = parse_and_normalize_response(response_text, variable_list)

        # 创建结构化图
        structured_graph: StructuredGraph | None = self.create_structured_graph(
            causal_graph, variable_list, domain_name, iteration=0
        )

        return structured_graph

    def _local_amendment(
        self,
        variable_list: list[str],
        domain_name: str,
        domain_context: str,
        previous_graph: StructuredGraph,
        memory: str | None,
        reflection: str | None,
        iteration: int,
        num_edge_operations: int = 3
    ) -> tuple[list[StructuredGraph], bool]:
        """
        局部修正：让模型选择对边进行操作（添加、删除、反转）
        每个操作单独应用到 previous_graph 上，生成多个候选图
        
        Args:
            num_edge_operations: 最大操作边数（LLM可以选择少于这个数量的操作），默认为3
            
        Returns:
            (StructuredGraph 列表, is_final_graph标志)
            is_final_graph=True 表示 LLM 认为 previous_graph 已经足够好
        """

        system_prompt = construct_system_prompt(domain_name)
        user_prompt = construct_local_amendment_prompt(
            variable_list, domain_name, domain_context,
            previous_graph, memory, reflection, num_edge_operations
        )

        # 调用LLM
        response_text = self.llm_loader.generate(system_prompt, user_prompt)

        # 解析操作指令和is_final_graph标志
        parse_result = self._parse_edge_operations(response_text)
        operations = parse_result['operations']
        is_final_graph = parse_result['is_final_graph']
        overall_reasoning = parse_result['overall_reasoning']
        
        # 输出总体推理
        if overall_reasoning:
            print(f"\n[Overall Reasoning]: {overall_reasoning}\n")

        # 将每个操作单独应用到 previous_graph 上，生成多个候选图
        candidate_graphs = []
        
        for op in operations:
            # 应用单个操作
            updated_graph = self._apply_single_edge_operation(
                previous_graph, op, variable_list
            )
            
            if updated_graph is None:
                continue
            
            # 创建 Change 对象
            change = GraphChange(
                type=op['type'],
                parent=op['parent'],
                child=op['child'],
                reasoning=op['reasoning']
            )
            
            # 创建结构化图
            structured_graph = self.create_structured_graph(
                updated_graph, 
                variable_list, 
                domain_name, 
                iteration, 
                previous_graph, 
                change
            )
            
            if structured_graph is not None:
                candidate_graphs.append(structured_graph)
        
        print(f"✓ Generated {len(candidate_graphs)} candidate graphs from {len(operations)} operations")
        return candidate_graphs, is_final_graph

# ==== 以下为辅助函数 ====
    def create_structured_graph(
        self,
        causal_graph: dict,
        variable_list: list[str],
        domain_name: str,
        iteration: int,
        previous_graph: StructuredGraph | None = None,
        change: GraphChange | None = None,
    ) -> StructuredGraph | None:
        """创建最终的结构化图表示, None 表示无效图"""

        nodes: list = causal_graph["nodes"]

        # 验证变量完整性
        graph_vars = {node["name"] for node in nodes}
        expected_vars = set(variable_list)

        if graph_vars != expected_vars:
            missing = expected_vars - graph_vars
            extra = graph_vars - expected_vars

            if missing:
                print(f"⚠️  Warning: Missing variables: {missing}")
                # 添加缺失的变量（无父节点）
                for var in missing:
                    nodes.append({"name": var, "parents": []})

            if extra:
                print(f"⚠️  Warning: Extra variables (will be removed): {extra}")
                # 移除多余的变量
                nodes = [n for n in nodes if n["name"] in expected_vars]

        # 清理无效的父节点（不在变量列表中的父节点）
        for node in nodes:
            valid_parents = []
            invalid_parents = []
            for parent in node.get("parents", []):
                if parent in expected_vars:
                    valid_parents.append(parent)
                else:
                    invalid_parents.append(parent)

            if invalid_parents:
                print(
                    f"⚠️  Warning: Node '{node['name']}' has invalid parents (not in variable list): {invalid_parents}"
                )
                print(f"    These parents will be removed.")

            node["parents"] = valid_parents

        # 去重节点（保留第一次出现的节点）
        # 这是关键步骤：确保图结构中每个节点只出现一次
        seen_names = set()
        unique_nodes = []
        for node in nodes:
            if node["name"] not in seen_names:
                seen_names.add(node["name"])
                unique_nodes.append(node)
            else:
                print(f"⚠️  Warning: Duplicate node detected and removed: {node['name']}")

        if len(unique_nodes) < len(nodes):
            print(f"  Removed {len(nodes) - len(unique_nodes)} duplicate node(s)")

        nodes = unique_nodes

        # 检查环
        cycles, cycle_path = self._has_cycle(nodes)
        if cycles:
            print("⚠️  Warning: Graph contains cycles! Return None")
            print(f"    Cycle path: {' -> '.join(cycle_path)}")
            return None

        # 创建返回对象
        # 构建历史变化列表
        history = previous_graph.metadata.change_history.copy() if previous_graph else []
        if change:
            history.append(change)
        # 组装 GraphMetadata 对象
        metadata_obj = GraphMetadata(
            domain=domain_name,
            iteration=iteration,
            num_variables=len(variable_list),
            num_edges=self._count_edges(nodes),
            change_history=history,
            is_final_graph=False  # 默认为 False，由搜索器标记
        )
        nodes_objs = [
            CausalNode(name=node["name"], parents=node["parents"]) for node in nodes
        ]

        # 计算邻接矩阵
        adj_matrix, _ = self._create_adjacency_matrix(nodes, variable_list)


        # 组装
        structured_graph = StructuredGraph(
            metadata=metadata_obj, 
            nodes=nodes_objs, 
            adjacency_matrix=adj_matrix,
        )
        return structured_graph
    
    def _has_cycle(self, nodes: list[dict]) -> tuple[bool, list[str]]:
        """检查是否有环（DFS算法），并记录环路信息"""

        # 构建邻接表
        graph = defaultdict(list)
        all_nodes = set()

        for node in nodes:
            node_name = node["name"]
            all_nodes.add(node_name)

            for parent in node.get("parents", []):
                all_nodes.add(parent)
                graph[parent].append(node_name)

        # DFS检测环，并记录环路
        visited = set()
        rec_stack = set()
        cycle_path = []  # 记录环路路径
        found_cycle = False

        def dfs(node, path):
            nonlocal found_cycle, cycle_path
            if found_cycle:
                return True

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # 找到环！记录从neighbor到当前节点的路径
                    cycle_start_idx = path.index(neighbor)
                    cycle_path = path[cycle_start_idx:] + [neighbor]
                    found_cycle = True
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in all_nodes:
            if node not in visited:
                if dfs(node, []):
                    return True, cycle_path

        return False, cycle_path

    def _count_edges(self, nodes: list[dict]) -> int:
        """计算边数"""
        return sum(len(node.get('parents', [])) for node in nodes)

    def _create_adjacency_matrix(
        self,
        nodes: list[dict],
        variable_list: list[str]
    ) -> tuple[np.ndarray, pd.DataFrame]:
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

    def _parse_edge_operations(self, response_text: str) -> dict:
        """
        解析LLM返回的边操作指令、overall_reasoning和is_final_graph标志
        
        Returns:
            包含 'operations', 'overall_reasoning' 和 'is_final_graph' 的字典
        """
        # print(f"Raw operations response (first 500 chars):\n{response_text[:500]}\n")

        # 提取JSON
        json_obj = extract_json(response_text)

        if json_obj is None:
            print("⚠️  Failed to extract operations JSON. Using empty operations.")
            return {'operations': [], 'overall_reasoning': '', 'is_final_graph': False}

        operations = json_obj.get('operations', [])
        is_final_graph = json_obj.get('is_final_graph', False)
        # 提取 overall_reasoning (兼容两个字段名)
        overall_reasoning = json_obj.get('overall_reasoning', json_obj.get('reasoning', ''))

        if not isinstance(operations, list):
            print(f"⚠️  'operations' must be a list, got {type(operations)}")
            return {'operations': [], 'is_final_graph': False}

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
                'reasoning': reasoning if reasoning else f"{op_type} edge: {parent} → {child}"
            })

        print(f"✓ Parsed {len(valid_operations)} valid operations")
        for i, op in enumerate(valid_operations, 1):
            print(f"  {i}. {op['type']}: {op['parent']} → {op['child']}")
        
        if is_final_graph:
            print(f"[LLMHypothesisGenerator] 🏁 LLM indicates this is a FINAL graph (no further changes needed)")

        return {
            'operations': valid_operations, 
            'overall_reasoning': overall_reasoning,
            'is_final_graph': is_final_graph
        }

    def _apply_single_edge_operation(
        self,
        previous_graph: StructuredGraph,
        operation: dict,
        variable_list: list[str]
    ) -> dict | None:
        """
        将单个边操作应用到上一轮的图上
        
        Args:
            previous_graph: 上一轮的图结构
            operation: 单个操作
            variable_list: 变量列表
            
        Returns:
            更新后的图（nodes格式），如果操作无效则返回 None
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

        op_type = operation['type']
        parent = operation['parent']
        child = operation['child']
        reasoning = operation.get('reasoning', '')

        # 验证变量存在
        if parent not in variable_list or child not in variable_list:
            print(f"⚠️  Skipping operation with invalid variables: {parent} → {child}")
            return None

        if parent == child:
            print(f"⚠️  Skipping self-loop: {parent} → {child}")
            return None

        child_node = node_map[child]

        if op_type == 'ADD':
            # 添加边
            if parent not in child_node['parents']:
                child_node['parents'].append(parent)
                print(f"  ✓ Added edge: {parent} → {child}")
            else:
                print(f"  ⚠️  Edge already exists: {parent} → {child}")
                return None  # 边已存在，返回 None

        elif op_type == 'DELETE':
            # 删除边
            if parent in child_node['parents']:
                child_node['parents'].remove(parent)
                print(f"  ✓ Deleted edge: {parent} → {child}")
            else:
                print(f"  ⚠️  Edge doesn't exist: {parent} → {child}")
                return None  # 边不存在，返回 None

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
                    return None
            else:
                print(f"  ⚠️  Cannot reverse non-existent edge: {parent} → {child}")
                return None

        # 返回标准化的图格式
        return {
            'nodes': nodes,
            'reasoning': reasoning if reasoning else f"Applied {op_type} operation: {parent} → {child}"
        }

    def _apply_edge_operations(
        self,
        previous_graph: StructuredGraph,
        operations: list[dict],
        variable_list: list[str]
    ) -> dict:
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
