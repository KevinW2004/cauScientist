from typing import Dict, List
import json
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from metrics import has_cycle

def extract_json(text: str) -> Optional[Dict]:
    """从文本中提取JSON对象，具有增强的容错性"""
    
    def remove_json_comments(json_str: str) -> str:
        """移除JSON字符串中的单行注释 (// ...)，同时处理引号状态"""
        result = []
        in_quotes = False
        escape_next = False
        i = 0
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
                
            if char == '\\':
                escape_next = True
                result.append(char)
            elif char == '"':
                in_quotes = not in_quotes
                result.append(char)
            elif not in_quotes and json_str[i:i+2] == '//':
                # 找到注释，跳到行尾
                eol = json_str.find('\n', i)
                if eol == -1:
                    break
                i = eol - 1  # 保持换行符在下一轮处理
            else:
                result.append(char)
            i += 1
        return "".join(result)
    
    def preprocess_json_text(raw_text: str) -> str:
        """预处理JSON文本，处理非法换行、末尾逗号、括号不匹配等"""
        # 1. 寻找第一个 { 和最后一个 }
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return raw_text
        
        json_candidate = raw_text[start:end+1]
        
        # 2. 移除注释
        json_candidate = remove_json_comments(json_candidate)
        
        # 3. 逐字符修复内部换行
        # 这种方法能处理复杂的转义，并为后续修复打桩
        chars = []
        in_quotes = False
        escape_next = False
        
        i = 0
        while i < len(json_candidate):
            char = json_candidate[i]
            
            if escape_next:
                chars.append(char)
                escape_next = False
                i += 1
                continue
                
            if char == '\\':
                escape_next = True
                chars.append(char)
            elif char == '"':
                in_quotes = not in_quotes
                chars.append(char)
            elif char == '\n' or char == '\r':
                if in_quotes:
                    chars.append('\\n' if char == '\n' else '\\r')
                else:
                    chars.append(char)
            else:
                chars.append(char)
            i += 1
            
        json_str = "".join(chars)

        # 4. 修复结构性错误 (LLM 常见错误)
        
        # a. 移除末尾逗号 (Trailing commas) - 处理 }, 或 ]
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        
        # b. 修复括号不匹配: "key": { ... ] -> "key": { ... }
        # 这种情况通常发生在嵌套结构的末尾
        # 我们寻找 ": {" 开启但在遇到下一个键或结束前却用了 "]" 关闭的情况
        
        # 简单的启发式修复：如果发现 "key": { 后面跟着很多内容，最后是 ], 且前面没有对应的 [
        # 这里使用一种平衡堆栈的方法来尝试修复
        
        fixed_chars = []
        stack = []
        in_quotes = False
        escape_next = False
        
        i = 0
        while i < len(json_str):
            char = json_str[i]
            if escape_next:
                fixed_chars.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                fixed_chars.append(char)
            elif char == '"':
                in_quotes = not in_quotes
                fixed_chars.append(char)
            elif not in_quotes:
                if char == '{':
                    stack.append('{')
                    fixed_chars.append(char)
                elif char == '[':
                    stack.append('[')
                    fixed_chars.append(char)
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        fixed_chars.append(char)
                    elif stack and stack[-1] == '[':
                        # 括号不匹配：应该是 ] 但写成了 }
                        # 或者 stack 为空
                        # 暂时保留，或者尝试修复
                        fixed_chars.append(']') # 尝试修复为正确的
                        stack.pop()
                    else:
                        fixed_chars.append(char)
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()
                        fixed_chars.append(char)
                    elif stack and stack[-1] == '{':
                        # 括号不匹配：应该是 } 但写成了 ]
                        fixed_chars.append('}') # 尝试修复为正确的
                        stack.pop()
                    else:
                        fixed_chars.append(char)
                else:
                    fixed_chars.append(char)
            else:
                fixed_chars.append(char)
            i += 1
            
        return "".join(fixed_chars)

    # 尝试直接解析
    processed_text = ""
    try:
        processed_text = preprocess_json_text(text)
        return json.loads(processed_text)
    except Exception as e:
        # print(f"  [Debug] Standard JSON parsing failed: {e}")
        pass
    
    # 方法2: 寻找代码块
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            processed_text = preprocess_json_text(json_match.group(1))
            return json.loads(processed_text)
        except Exception as e:
            # print(f"  [Debug] Codeblock JSON parsing failed: {e}")
            pass
    
    # 最后保底：查找文本中的 {} 结构
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1:
        try:
            json_str = text[start_idx:end_idx+1]
            processed_text = preprocess_json_text(json_str)
            return json.loads(processed_text)
        except Exception as e:
            print(f"⚠️  JSON parsing eventually failed: {e}")
            # print(f"  [Debug] Problematic processed text near error:\n{processed_text[:500]}...")
            pass
    
    return None

def convert_parent_to_child_dict(parent_to_children: Dict[str, List[str]], variable_list: List[str]) -> List[Dict]:
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

def normalize_node_dict(nodes_dict: Dict, variable_list: List[str]) -> List[Dict]:
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
        return convert_parent_to_child_dict(nodes_dict, variable_list)
    
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

def normalize_node_list(nodes: List) -> List[Dict]:
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

def normalize_graph_structure(json_obj: Dict, variable_list: List[str]) -> Dict:
    """
    将各种格式的图转换为标准格式
    
    标准格式: {"nodes": [{"name": "X", "parents": [...]}, ...], "reasoning": "...", "confirmed_edges": [], "edge_notes": {}}
    """
    
    if 'nodes' not in json_obj:
        raise ValueError("JSON must contain 'nodes' field")
    
    nodes_data = json_obj['nodes']
    reasoning = json_obj.get('reasoning', json_obj.get('overall_reasoning', ''))
    confirmed_edges = json_obj.get('confirmed_edges', [])
    edge_notes = json_obj.get('edge_notes', {})
    
    # 情况1: nodes 已经是列表
    if isinstance(nodes_data, list):
        normalized_nodes = normalize_node_list(nodes_data)
    
    # 情况2: nodes 是字典
    elif isinstance(nodes_data, dict):
        normalized_nodes = normalize_node_dict(nodes_data, variable_list)
    
    else:
        raise ValueError(f"'nodes' must be list or dict, got {type(nodes_data)}")
    
    return {
        'nodes': normalized_nodes,
        'reasoning': reasoning,
        'confirmed_edges': confirmed_edges,
        'edge_notes': edge_notes
    }

def parse_and_normalize_response(
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
    json_obj = extract_json(response_text)
    
    if json_obj is None:
        print("⚠️  Failed to extract JSON.")
        return None, {
            'success': False,
            'has_cycle': False,
            'cycle_path': None,
            'missing_variables': [],
            'extra_variables': [],
            'invalid_parents': {},
            'error_messages': ['Failed to extract JSON.'],
            'operation_errors': [],
        }

    # 步骤2: 标准化格式
    normalized_graph = normalize_graph_structure(json_obj, variable_list)
    
    return normalized_graph, {
            'success': True,
            'has_cycle': False,
            'cycle_path': None,
            'missing_variables': [],
            'extra_variables': [],
            'invalid_parents': {},
            'error_messages': [],
            'operation_errors': [],
        }

def parse_edge_operations(response_text: str, num_edge_operations: int = 1) -> Tuple[List[Dict], str, List[str], Dict]:
    """
    解析LLM返回的边操作指令，并提取因果档案信息
    
    Returns:
        (操作列表, 推理逻辑, 确认边列表, 边笔记字典)
    """
    # print(f"Raw operations response (first 500 chars):\n{response_text[:500]}\n")
    
    # 提取JSON
    json_obj = extract_json(response_text)
    
    if json_obj is None:
        print("⚠️  Failed to extract operations JSON. Using empty operations.")
        return [], "", [], {}
    
    operations = json_obj.get('operations', [])
    overall_reasoning = json_obj.get('overall_reasoning', json_obj.get('reasoning', ''))
    confirmed_edges = json_obj.get('confirmed_edges', [])
    edge_notes = json_obj.get('edge_notes', {})
    
    if not isinstance(operations, list):
        print(f"⚠️  'operations' must be a list, got {type(operations)}")
        operations = []
    
    # 验证每个操作
    valid_operations = []
    for op in operations:
        if not isinstance(op, dict):
            continue
        
        op_type = op.get('type', '').upper()
        parent = op.get('parent')
        child = op.get('child')
        reasoning = op.get('reasoning', '')
        
        if op_type not in ['ADD', 'DELETE', 'REVERSE']:
            continue
        
        if not parent or not child:
            continue
        
        valid_operations.append({
            'type': op_type,
            'parent': parent,
            'child': child,
            'reasoning': reasoning
        })
    
    if len(valid_operations) > 0:
        print(f"✓ Parsed {len(valid_operations)} valid operations")
    
    return valid_operations, overall_reasoning, confirmed_edges, edge_notes

def apply_edge_operations(
    previous_graph: Dict,
    operations: List[Dict],
    variable_list: List[str],
    valid_ops: Optional[Dict] = None
) -> Dict:
    """
    将边操作应用到上一轮的图上，并收集操作错误到验证信息
    
    Args:
        previous_graph: 上一轮的图结构
        operations: 操作列表
        variable_list: 变量列表
        
    Returns:
        更新后的图（nodes格式）
    """
    _validation_info = {
            'success': True,
            'has_cycle': False,
            'cycle_path': None,
            'missing_variables': [],
            'extra_variables': [],
            'invalid_parents': {},
            'error_messages': [],
            'operation_errors': [],
        }
    
    # 复制节点数据
    nodes = []
    for node in previous_graph['nodes']:
        nodes.append({
            'name': node['name'],
            'parents': node.get('parents', []).copy()
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
    successful_ops = 0
    
    # 特殊处理：如果操作列表为空，将其视为一种成功的“维持现状”决策
    if len(operations) == 0:
        print("✓ LLM decided no edge modifications are needed (maintaining status quo).")
        return {
            'nodes': nodes,
            'reasoning': "No modifications requested by LLM."
        }, _validation_info

    for op in operations:
        op_type = op['type']
        parent = op['parent']
        child = op['child']
        
        # 验证变量存在
        if parent not in variable_list or child not in variable_list:
            error_msg = f"Operation {op_type} has invalid variables: {parent} → {child}"
            print(f"⚠️  Skipping: {error_msg}")
            _validation_info['operation_errors'].append(error_msg)
            _validation_info['error_messages'].append(error_msg)
            continue
        
        if parent == child:
            error_msg = f"Operation {op_type} attempts to create self-loop: {parent} → {child}"
            print(f"⚠️  Skipping: {error_msg}")
            _validation_info['operation_errors'].append(error_msg)
            _validation_info['error_messages'].append(error_msg)
            continue
        
        child_node = node_map[child]
        
        if op_type == 'ADD':
            # 添加边
            if valid_ops is not None and (parent, child) not in valid_ops['can_add']:
                print(f"⚠️  Skipping operation not in valid operations: {op}")
                _validation_info['operation_errors'].append(f"ADD operation failed: edge {parent} → {child} not in valid operations")
                _validation_info['error_messages'].append(f"ADD operation failed: edge {parent} → {child} not in valid operations")
            elif parent not in child_node['parents']:
                child_node['parents'].append(parent)
                print(f"  ✓ Added edge: {parent} → {child}")
                successful_ops += 1
            else:
                error_msg = f"ADD operation failed: edge {parent} → {child} already exists"
                print(f"  ⚠️  {error_msg}")
                _validation_info['operation_errors'].append(error_msg)
                _validation_info['error_messages'].append(error_msg)
        
        elif op_type == 'DELETE':
            # 删除边
            if valid_ops is not None and (parent, child) not in valid_ops['can_delete']:
                print(f"⚠️  Skipping operation not in valid operations: {op}")
                _validation_info['operation_errors'].append(f"DELETE operation failed: edge {parent} → {child} not in valid operations")
                _validation_info['error_messages'].append(f"DELETE operation failed: edge {parent} → {child} not in valid operations")
            elif parent in child_node['parents']:
                child_node['parents'].remove(parent)
                print(f"  ✓ Deleted edge: {parent} → {child}")
                successful_ops += 1
            else:
                error_msg = f"DELETE operation failed: edge {parent} → {child} doesn't exist"
                print(f"  ⚠️  {error_msg}")
                _validation_info['operation_errors'].append(error_msg)
                _validation_info['error_messages'].append(error_msg)
        
        elif op_type == 'REVERSE':
            # 反转边: 删除 parent → child，添加 child → parent
            # 处理歧义：检查parent→child是否存在，如果不存在，尝试child→parent
            
            if valid_ops is not None and (((parent, child) not in valid_ops['can_reverse']) and ((child, parent) not in valid_ops['can_reverse'])):
                print(f"⚠️  Skipping operation not in valid operations: {op}")
                _validation_info['operation_errors'].append(f"REVERSE operation failed: edge {parent} → {child} or {child} → {parent} not in valid operations")
                _validation_info['error_messages'].append(f"REVERSE operation failed: edge {parent} → {child} or {child} → {parent} not in valid operations")
            elif parent in child_node['parents']:
                # 标准情况：parent→child存在
                child_node['parents'].remove(parent)
                parent_node = node_map[parent]
                if child not in parent_node['parents']:
                    parent_node['parents'].append(child)
                    print(f"  ✓ Reversed edge: {parent} → {child} to {child} → {parent}")
                    successful_ops += 1
                else:
                    error_msg = f"REVERSE operation failed: reversing {parent} → {child} would create duplicate edge {child} → {parent}"
                    print(f"  ⚠️  {error_msg}")
                    _validation_info['operation_errors'].append(error_msg)
                    _validation_info['error_messages'].append(error_msg)
                    # 还原删除的边
                    child_node['parents'].append(parent)
            else:
                # 检查是否是反向写法（用户想反转child→parent，但写成了parent→child）
                parent_node = node_map[parent]
                if child in parent_node['parents']:
                    # 实际存在的是child→parent，用户可能想反转它
                    print(f"  ⚠️  Note: Edge {parent} → {child} doesn't exist, but {child} → {parent} exists.")
                    print(f"      Interpreting as: reverse {child} → {parent} to {parent} → {child}")
                    
                    parent_node['parents'].remove(child)
                    if parent not in child_node['parents']:
                        child_node['parents'].append(parent)
                        print(f"  ✓ Reversed edge: {child} → {parent} to {parent} → {child}")
                        successful_ops += 1
                    else:
                        error_msg = f"REVERSE operation failed: reversing {child} → {parent} would create duplicate edge {parent} → {child}"
                        print(f"  ⚠️  {error_msg}")
                        _validation_info['operation_errors'].append(error_msg)
                        _validation_info['error_messages'].append(error_msg)
                        # 还原
                        parent_node['parents'].append(child)
                else:
                    # 两个方向的边都不存在
                    error_msg = f"REVERSE operation failed: neither {parent} → {child} nor {child} → {parent} exists"
                    print(f"  ⚠️  {error_msg}")
                    _validation_info['operation_errors'].append(error_msg)
                    _validation_info['error_messages'].append(error_msg)
    
    # 如果所有操作都失败了，标记为不成功
    if len(operations) > 0 and successful_ops == 0:
        _validation_info['success'] = False
        summary_msg = f"All {len(operations)} edge operations failed"
        _validation_info['error_messages'].append(summary_msg)
    
    # 返回标准化的图格式
    return {
        'nodes': nodes,
        'reasoning': f"Applied {successful_ops}/{len(operations)} local operations successfully"
    }, _validation_info

def compute_changes(prev_graph: Dict, curr_graph: Dict) -> Dict:
    """计算图之间的变化"""
    
    def get_edges(graph):
        edges = set()
        for node in graph.get('nodes', []):
            child = node['name']
            for parent in node.get('parents', []):
                edges.add((parent, child))
        return edges
    
    prev_edges = get_edges(prev_graph)
    curr_edges = get_edges(curr_graph)
    
    added = curr_edges - prev_edges
    removed = prev_edges - curr_edges
    
    return {
        "added_edges": list(added),
        "removed_edges": list(removed),
        "num_added": len(added),
        "num_removed": len(removed)
    }

def count_edges(nodes: List[Dict]) -> int:
    """计算边数"""
    return sum(len(node.get('parents', [])) for node in nodes)

def create_adjacency_matrix(
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

def create_structured_graph(
    causal_graph: Dict,
    variable_list: List[str],
    domain_name: str,
    iteration: int,
    previous_graph: Optional[Dict] = None,
    validation_info: Optional[Dict] = None,
    proposed_operations: Optional[List[Dict]] = None
) -> Dict:
    """创建最终的结构化图表示，并收集验证信息"""
    
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
            validation_info['missing_variables'] = list(missing)
            validation_info['error_messages'].append(
                f"Missing variables: {', '.join(missing)}"
            )
            # 添加缺失的变量（无父节点）
            for var in missing:
                nodes.append({'name': var, 'parents': []})
        
        if extra:
            print(f"⚠️  Warning: Extra variables (will be removed): {extra}")
            validation_info['extra_variables'] = list(extra)
            validation_info['error_messages'].append(
                f"Extra variables (removed): {', '.join(extra)}"
            )
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
            validation_info['invalid_parents'][node['name']] = invalid_parents
            validation_info['error_messages'].append(
                f"Node '{node['name']}' has invalid parents: {', '.join(invalid_parents)}"
            )
        
        node['parents'] = valid_parents
    
    # 去重节点（保留第一次出现的节点）
    # 这是关键步骤：确保图结构中每个节点只出现一次
    seen_names = set()
    unique_nodes = []
    for node in nodes:
        if node['name'] not in seen_names:
            seen_names.add(node['name'])
            unique_nodes.append(node)
        else:
            print(f"⚠️  Warning: Duplicate node detected and removed: {node['name']}")
            validation_info['error_messages'].append(
                f"Duplicate node: {node['name']}"
            )
    
    if len(unique_nodes) < len(nodes):
        print(f"  Removed {len(nodes) - len(unique_nodes)} duplicate node(s)")
    
    nodes = unique_nodes
    
    # 检查环 - 这是关键的验证步骤
    cycles, cycle_path = has_cycle(nodes)
    if cycles:
        validation_info['has_cycle'] = True
        validation_info['success'] = False
        validation_info['cycle_path'] = cycle_path
        
        # 构建详细的错误消息
        error_msg = "Graph contains cycles (not a valid DAG)."
        
        if cycle_path:
            # 显示环路路径
            cycle_str = " → ".join(cycle_path)
            error_msg += f"\nCycle detected: {cycle_str}"
        
            if proposed_operations:
                # 检查哪些操作涉及环路中的边
                cycle_edges = set()
                for i in range(len(cycle_path) - 1):
                    cycle_edges.add((cycle_path[i], cycle_path[i+1]))
                
                problematic_ops = []
                for op in proposed_operations:
                    if op['type'] in ['ADD', 'REVERSE']:
                        # ADD或REVERSE可能创建环
                        edge = (op['parent'], op['child'])
                        reverse_edge = (op['child'], op['parent'])
                        if edge in cycle_edges or reverse_edge in cycle_edges:
                            problematic_ops.append(op)
                
                if problematic_ops:
                    error_msg += f"\nOperations that may have caused the cycle:"
                    for op in problematic_ops:
                        error_msg += f"\n  - {op['type']}: {op['parent']} → {op['child']}"
                    error_msg += f"\nAvoid these operations or modify them to prevent cycles."
        
        print(f"⚠️  Warning: {error_msg}")
        validation_info['error_messages'].append(error_msg)
    
    # 计算变化
    changes = None
    if previous_graph is not None:
        changes = compute_changes(previous_graph, {'nodes': nodes})

    # 创建结构化图
    structured_graph = {
        "metadata": {
            "domain": domain_name,
            "iteration": iteration,
            "num_variables": len(variable_list),
            "num_edges": count_edges(nodes),
            "reasoning": reasoning,
            "confirmed_edges": causal_graph.get('confirmed_edges', []),
            "edge_notes": causal_graph.get('edge_notes', {}),
            "changes": changes,
            "proposed_operations": proposed_operations  # 保存提议的操作
        },
        "nodes": nodes
    }
    
    # 计算邻接矩阵
    adj_matrix, _ = create_adjacency_matrix(nodes, variable_list)
    structured_graph["adjacency_matrix"] = adj_matrix.tolist()
    
    return structured_graph, validation_info
