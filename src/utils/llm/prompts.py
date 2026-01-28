from schemas import StructuredGraph

def construct_system_prompt(domain_name: str) -> str:
    """构建系统提示词"""
    return f"""You are an expert in causal inference and {domain_name} domain knowledge.
Your task is to generate or refine causal graph hypotheses representing causal relationships.
Always ensure the graph is a Directed Acyclic Graph (DAG) with no cycles.
Do not output the thinking process or <think></think> tags. ./no_think
"""

def construct_initial_prompt(
    variable_list: list[str],
    domain_name: str,
    domain_context: str
) -> str:
    """构建初始假设生成的提示词"""

    variables_formatted = "\n".join([f"- {var}" for var in variable_list])

    context_section = ""
    if domain_context:
        context_section = f"\n\nDomain Context:\n{domain_context}\n"

    prompt = f"""Generate an initial causal graph hypothesis for the {domain_name} domain.

Variables to analyze:
{variables_formatted}
{context_section}

Instructions:
1. **THINK FIRST**: Analyze the domain and relationships before proposing a structure
2. For each variable, determine its DIRECT CAUSES (parent variables)
3. Consider only direct causal relationships
4. Ensure the graph is a DAG (no cycles)
5. Base reasoning on domain knowledge and temporal ordering

Output Format (IMPORTANT - use this exact structure):
{{
"reasoning": "Explanation of the causal structure and your reasoning process",
"nodes": [
{{
    "name": "VariableName1",
    "parents": ["ParentVar1", "ParentVar2"]
}},
{{
    "name": "VariableName2",
    "parents": []
}}
]
}}

CRITICAL: 
- Put "reasoning" FIRST to explain your thought process
- Use "nodes" as a LIST (not a dictionary)
- Each node must have "name" and "parents" fields
- "parents" must be a list (use [] for root nodes)
- Output ONLY valid JSON"""

    return prompt

def construct_local_amendment_prompt(
    variable_list: list[str],
    domain_name: str,
    domain_context: str,
    previous_graph: StructuredGraph,
    memory: str | None,
    reflection: str | None,
    num_edge_operations: int
) -> str:
    """构建局部修正的提示词"""

    variables_formatted = "\n".join([f"- {var}" for var in variable_list])

    graph_description = _graph_description(previous_graph)

    # 格式化记忆
    memory_section = ""
    if memory:
        memory_section = f"""
Previous Feedback:
{memory}
Use this feedback to guide your edge operations.
"""

    context_section = ""
    if domain_context:
        context_section = f"\n\nDomain Context:\n{domain_context}\n"

    # 格式化反思
    reflection_section = ""
    if reflection and reflection.strip() != "":
        reflection_section = f"""HISTORICAL INSIGHTS (REFLECTION):
The following reflection contains lessons learned from previous iterations, including both successful improvements and failed attempts:
"{reflection}"
CRITICAL INSTRUCTION: 
- Learn from the SUCCESSFUL changes: replicate the causal reasoning that led to improvements
- Avoid the FAILED attempts: do not repeat operations that worsened the score
- Use these insights to guide your current amendment strategy
"""

    # 构建提示词
    prompt = f"""Perform LOCAL amendments to the causal graph for the {domain_name} domain.

Variables:
{variables_formatted}
{context_section}
{graph_description}
{reflection_section}
{memory_section}

Instructions:
You need to propose UP TO {num_edge_operations} individual edge operations to improve the graph.
The edge operations won't be combined. Instead they will be applied one by one to generate multiple new candidate graphs for evaluation. 
You can propose fewer operations if appropriate, but not more than {num_edge_operations}.
If you believe the graph is already optimal and there is no need for any further changes, you may propose ZERO operations and indicate it is final in "is_final_graph" field(boolean).

Available operations:
1. ADD: Add a new edge (Parent → Child)
2. DELETE: Remove an existing edge (Parent → Child)
3. REVERSE: Reverse an edge direction (Parent → Child becomes Child → Parent)

Constraints:
- Propose at most {num_edge_operations} operations (can be fewer)
- Ensure operations don't create cycles (resulting graph must be a DAG)
- Base decisions on domain knowledge and model fit feedback
- For DELETE operations, the edge must exist in the current graph
- For REVERSE operations, the edge must exist and reversing must not create a cycle
- The "is_final_graph" can be true only if you propose ZERO operations
- Review the 'Change History' carefully. Do NOT propose operations that undo previous changes (e.g., do not re-add a deleted edge, do not reverse an edge back to its original direction) unless there is compelling new evidence.

Output Format (IMPORTANT - use this exact JSON structure):
{{
"overall_reasoning": "A breif overall explanation of the fault of the current graph and the amendment strategy",
"operations": [
{{
    "type": "ADD",
    "parent": "VariableName1",
    "child": "VariableName2",
    "reasoning": "Why this specific edge should be added"
}},
{{
    "type": "DELETE",
    "parent": "VariableName3",
    "child": "VariableName4",
    "reasoning": "Why this specific edge should be removed"
}},
{{
    "type": "REVERSE",
    "parent": "VariableName5",
    "child": "VariableName6",
    "reasoning": "Why edge VariableName5→VariableName6 should be reversed to VariableName6→VariableName5"
}}
],
"is_final_graph": false  // Set to true if you believe the graph is final and needs no further changes
}}

CRITICAL: 
- Put "overall_reasoning" FIRST to explain your strategy
- Output ONLY valid JSON
- Include UP TO {num_edge_operations} operations (can be 0 to {num_edge_operations})
- Valid operation types: "ADD", "DELETE", "REVERSE"
- Each operation must have: reasoning, type, parent, child
- For REVERSE: always use the EXISTING edge direction (parent→child) in your JSON"""
    # 测试用：保存到文件看看提示词
    _save_prompt_to_file(prompt)
    return prompt

def construct_reflection_prompt(
    domain_name: str,
    domain_context: str,
    current_graph: StructuredGraph,
    score_diff: float,
    is_better: bool,
    current_reflection: str
) -> tuple[str, str]:
    """根据当前图的变化构建反思分析的提示词，反思会和之前的反思合并
    Args:
        domain_name (str): 领域名称
        domain_context (str): 领域背景信息
        current_graph (StructuredGraph): 当前的因果图
        score_diff (float): 当前图与上一个图的分数差异
        is_better (bool): 当前图是否优于上一个图
        current_reflection (str): 现有的全局反思内容
    Returns:
        system_prompt, user_prompt (Tuple[str, str]): 系统提示词和用户提示词
    """
    # 系统提示词
    system_prompt = f"""You are an expert in causal inference and {domain_name} domain knowledge.
Your task is to analyze the changes made to a causal graph, provide insights on their effectiveness and maintain a global, evolving memory (Reflection).
Your output must be a concise, merged reflection text that helps the agent avoid repeating mistakes.
Do NOT output future advice. Your only mission is to extract and summarize the lessons.
Do NOT output code or JSON. Output ONLY the updated reflection text.
Focus on the following aspects:
1. What changes were made and why they might be significant
2. How these changes affected model fit (validation score)
Do not output the thinking process or <think></think> tags. ./no_think
"""
    # 用户提示词
    # 使用 metadata.change_history[-1] 作为最新的变更记录，格式化变更内容
    if current_graph.metadata.change_history is None or len(current_graph.metadata.change_history) == 0:
        raise ValueError("No change history found in current_graph metadata. The reflection prompt requires at least one change record.")
    latest_change = current_graph.metadata.change_history[-1]
    latest_change_desc = f"{latest_change.type} edge {latest_change.parent} → {latest_change.child}"
    graph_desc = _graph_description(current_graph)
    score_msg = f"""The likelihood score {'IMPROVED' if is_better else 'DECLINED'} by {abs(score_diff):.4f} points due to the latest change({latest_change_desc}).
    """
    user_prompt = f"""Analyze the recent changes made to the causal graph.
{f"[Domain Context]: {domain_context}" if domain_context else ""}
[Current State]:
{graph_desc}
{score_msg}
[Previous Global Reflection]:
{current_reflection}
Instructions:
Based on the result above, UPDATE the Global Reflection.
1. If the score worsened: Explain WHY this causal assumption might be wrong (e.g., reverse causality, confounding). Merge this lesson into the reflection.
2. If the score improved: Reinforce why this relationship is likely true.
3. Merge precisely and concisely: Consolidate the new insight with the 'Previous Global Reflection' to form a single, coherent paragraph. Remove the redundant, obsolete or contradictory notes. If possible, divide the reflection into two sections: "Successful Insights" and "Mistake Analysis".
4. Concise and brief: Be brief in each specific point. The entire reflection should not exceed 500 words unless there are truly too many lessons.

Updated Reflection:"""

    return system_prompt, user_prompt

# ==== 辅助函数 ====
def _graph_description(graph: StructuredGraph) -> str:
    # 格式化当前图的边
    current_edges = []
    for node in graph.nodes:
        for parent in node.parents:
            current_edges.append(f"  {parent} → {node.name}")

    current_graph_str = "\n".join(current_edges) if current_edges else "  (no edges)"

    # 格式化修改历史, 使用 graph.metadata.change_history
    change_history_section = ""
    for change in graph.metadata.change_history:
        change_history_section += f"- {change.type} edge {change.parent} → {change.child}: {change.reasoning}\n"
    if change_history_section == "":
        change_history_section = "Null(no changes yet)."

    res = f"""Current Graph (Iteration {graph.metadata.iteration}):
{current_graph_str}

Change History (Already applied in the current graph above):
{change_history_section}

Current Likelihood Score: {graph.metadata.log_likelihood if graph.metadata.log_likelihood is not None else 'N/A'}
"""
    return res

def _save_prompt_to_file(prompt: str) -> None:
    from utils import ConfigManager
    file_path = ConfigManager().get("experiment.output.dir", ".") + "/prompt_history.txt"
    with open(file_path, "a") as f:
        f.write(prompt + "\n" + "="*35 + "\n")
