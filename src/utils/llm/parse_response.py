"""
流程图：
        parse_and_normalize_response(最终接口)
                        |
    -----------------------------------------
    |                                       |
extract_json                  _normalize_graph_structure
    |                                       |
(多种解析策略)                -------------------------------
                            |                             |
                    _normalize_node_list          _normalize_node_dict
                            |                             |
                        (标准列表格式)        -------------------------------
                                            |                             |
                                _convert_parent_to_child_dict         (子->父格式)
"""

import json
import re
from typing import List, Dict

def parse_and_normalize_response(
    response_text: str, variable_list: List[str]
) -> Dict:
    """
    解析LLM响应并标准化为统一格式

    支持多种格式：
    1. 标准列表: {"nodes": [{"name": "X", "parents": [...]}, ...], "reasoning": "..."}
    2. 父->子字典: {"nodes": {"Parent": ["Child1", "Child2"]}, "reasoning": "..."}
    3. 纯文本描述（尽力解析）

    Returns:
        标准化的图字典: {
            "nodes": [{"name": "X", "parents": [...]}], 
            "reasoning": "...",
            "is_final_graph": true/false
        }
    """

    # print(f"Raw LLM response (first 500 chars):\n{response_text[:500]}\n")

    # 步骤1: 提取JSON
    json_obj = extract_json(response_text)

    if json_obj is None:
        print("⚠️  Failed to extract JSON.")
        exit()

    # 步骤2: 标准化格式
    normalized_graph = _normalize_graph_structure(json_obj, variable_list)

    print(f"Reasoning from LLM: {normalized_graph.get('reasoning', '')}\n")

    return normalized_graph

def extract_json(text: str) -> Dict | None:
    """从文本中提取JSON对象"""

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

            if char == "\\":
                escape_next = True
                result.append(char)
            elif char == '"':
                in_quotes = not in_quotes
                result.append(char)
            elif not in_quotes and json_str[i : i + 2] == "//":
                # 找到注释，跳到行尾
                eol = json_str.find("\n", i)
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
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return raw_text

        json_candidate = raw_text[start : end + 1]

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

            if char == "\\":
                escape_next = True
                chars.append(char)
            elif char == '"':
                in_quotes = not in_quotes
                chars.append(char)
            elif char == "\n" or char == "\r":
                if in_quotes:
                    chars.append("\\n" if char == "\n" else "\\r")
                else:
                    chars.append(char)
            else:
                chars.append(char)
            i += 1

        json_str = "".join(chars)

        # 4. 修复结构性错误 (LLM 常见错误)

        # a. 移除末尾逗号 (Trailing commas) - 处理 }, 或 ]
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)

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

            if char == "\\":
                escape_next = True
                fixed_chars.append(char)
            elif char == '"':
                in_quotes = not in_quotes
                fixed_chars.append(char)
            elif not in_quotes:
                if char == "{":
                    stack.append("{")
                    fixed_chars.append(char)
                elif char == "[":
                    stack.append("[")
                    fixed_chars.append(char)
                elif char == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                        fixed_chars.append(char)
                    elif stack and stack[-1] == "[":
                        # 括号不匹配：应该是 ] 但写成了 }
                        # 或者 stack 为空
                        # 暂时保留，或者尝试修复
                        fixed_chars.append("]")  # 尝试修复为正确的
                        stack.pop()
                    else:
                        fixed_chars.append(char)
                elif char == "]":
                    if stack and stack[-1] == "[":
                        stack.pop()
                        fixed_chars.append(char)
                    elif stack and stack[-1] == "{":
                        # 括号不匹配：应该是 } 但写成了 ]
                        fixed_chars.append("}")  # 尝试修复为正确的
                        stack.pop()
                    else:
                        fixed_chars.append(char)
                else:
                    fixed_chars.append(char)
            else:
                fixed_chars.append(char)
            i += 1

        return "".join(fixed_chars)

    # 方法1: 直接解析整个文本
    processed_text = ""
    try:
        processed_text = preprocess_json_text(text)
        return json.loads(processed_text)
    except Exception as e:
        # print(f"  [Debug] Standard JSON parsing failed: {e}")
        pass

    # 方法2: 寻找代码块
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            processed_text = preprocess_json_text(json_match.group(1))
            return json.loads(processed_text)
        except Exception as e:
            # print(f"  [Debug] Codeblock JSON parsing failed: {e}")
            pass

    # 方法3: 查找第一个 { 到最后一个 }
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        try:
            json_str = text[start_idx : end_idx + 1]
            processed_text = preprocess_json_text(json_str)
            return json.loads(processed_text)
        except Exception as e:
            print(f"⚠️  JSON parsing eventually failed: {e}")
            # print(f"  [Debug] Problematic processed text near error:\n{processed_text[:500]}...")
            pass

    return None

def _normalize_graph_structure(
    json_obj: Dict, variable_list: List[str]
) -> Dict:
    """
    将各种格式的图转换为标准格式

    标准格式: {"nodes": [{"name": "X", "parents": [...]}, ...], "reasoning": "...", "is_final_graph": true/false}
    """

    if "nodes" not in json_obj:
        raise ValueError("JSON must contain 'nodes' field")

    nodes_data = json_obj["nodes"]
    reasoning = json_obj.get("reasoning", json_obj.get('overall_reasoning', ''))
    is_final_graph = json_obj.get("is_final_graph", json_obj.get("final_graph", False))

    # 情况1: nodes 已经是列表
    if isinstance(nodes_data, list):
        normalized_nodes = _normalize_node_list(nodes_data)

    # 情况2: nodes 是字典
    elif isinstance(nodes_data, dict):
        normalized_nodes = _normalize_node_dict(nodes_data, variable_list)
    else:
        raise ValueError(f"'nodes' must be list or dict, got {type(nodes_data)}")

    return {"nodes": normalized_nodes, "reasoning": reasoning, "is_final_graph": is_final_graph}

def _normalize_node_list(nodes: List) -> List[Dict]:
    """标准化节点列表格式"""

    normalized = []

    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError(f"Each node must be a dict, got {type(node)}")

        # 提取变量名（支持 'name' 或 'variable'）
        var_name = node.get("name") or node.get("variable")
        if not var_name:
            raise ValueError(f"Node missing 'name' or 'variable': {node}")

        # 提取父节点
        parents = node.get("parents", [])
        if not isinstance(parents, list):
            parents = []

        normalized_node = {"name": var_name, "parents": parents}

        normalized.append(normalized_node)

    return normalized

def _normalize_node_dict(
    nodes_dict: Dict, variable_list: List[str]
) -> List[Dict]:
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
        return _convert_parent_to_child_dict(nodes_dict, variable_list)

    # 格式2: 值是字典 -> 子->父格式
    elif isinstance(first_value, dict):
        normalized = []
        for var_name, node_info in nodes_dict.items():
            normalized_node = {
                "name": var_name,
                "parents": node_info.get("parents", []),
            }

            normalized.append(normalized_node)

        return normalized

    else:
        raise ValueError(f"Unsupported dict value type: {type(first_value)}")

def _convert_parent_to_child_dict(
    parent_to_children: Dict[str, List[str]], variable_list: List[str]
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
        {"name": var_name, "parents": parents}
        for var_name, parents in node_parents.items()
    ]

    return nodes
