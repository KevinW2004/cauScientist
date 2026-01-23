from typing import List, Optional
from schemas import StructuredGraph

def construct_system_prompt(domain_name: str) -> str:
    """构建系统提示词"""
    return f"""You are an expert in causal inference and {domain_name} domain knowledge.
Your task is to generate or refine causal graph hypotheses representing causal relationships.
Always ensure the graph is a Directed Acyclic Graph (DAG) with no cycles.
"""

def construct_initial_prompt(
    variable_list: List[str],
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
    variable_list: List[str],
    domain_name: str,
    domain_context: str,
    previous_graph: StructuredGraph,
    memory: Optional[str],
    num_edge_operations: int
) -> str:
    """构建局部修正的提示词"""

    variables_formatted = "\n".join([f"- {var}" for var in variable_list])

    # 格式化当前图的边
    current_edges = []
    for node in previous_graph.nodes:
        for parent in node.parents:
            current_edges.append(f"  {parent} → {node.name}")

    current_graph_str = "\n".join(current_edges) if current_edges else "  (no edges)"

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

    prompt = f"""Perform LOCAL amendments to the causal graph for the {domain_name} domain.

Variables:
{variables_formatted}
{context_section}

Current Graph (Iteration {previous_graph.metadata.iteration}):
{current_graph_str}

Current BIC Score: {previous_graph.metadata.log_likelihood if previous_graph.metadata.log_likelihood is not None else 'N/A'}
{memory_section}

Instructions:
You need to propose UP TO {num_edge_operations} edge operations to improve the graph.
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

Output Format (IMPORTANT - use this exact JSON structure):
{{
"overall_reasoning": "Overall explanation of the amendment strategy",
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

    return prompt
