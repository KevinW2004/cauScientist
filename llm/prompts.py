from typing import List, Dict, Optional

def construct_system_prompt(domain_name: str) -> str:
    """构建系统提示词"""
    return f"""You are an expert in causal inference and {domain_name} domain knowledge.
Your task is to generate or refine causal graph hypotheses representing causal relationships.
Always ensure the graph is a Directed Acyclic Graph (DAG) with no cycles.
"""

def construct_initial_prompt(
    variable_list: List[str],
    domain_name: str,
    domain_context: str,
    skeleton_constraints: Optional[Dict] = None,
    failed_attempts: Optional[List[Dict]] = None,  # 失败的尝试历史
    baseline_reference: Optional[str] = None,  # 新增：基线参考信息
    interventional_evidence: Optional[str] = None  # 新增：干预实验证据
) -> str:
    """构建初始假设生成的提示词"""
    
    variables_formatted = "\n".join([f"- {var}" for var in variable_list])
    
    context_section = ""
    if domain_context:
        context_section = f"\n\nDomain Context:\n{domain_context}\n"
    
    # 添加基线参考信息
    baseline_section = ""
    if baseline_reference:
        baseline_section = f"\n{baseline_reference}\n"

    # 添加干预证据信息
    evidence_section = ""
    if interventional_evidence:
        evidence_section = f"\n🔬 INTERVENTIONAL EVIDENCE (from experiments):\n{interventional_evidence}\n"
    
    # 添加骨架约束信息
    skeleton_section = ""
    if skeleton_constraints:
        allowed_edges = skeleton_constraints.get('allowed_edges', [])
        forbidden_pairs = skeleton_constraints.get('forbidden_pairs', [])
        
        if allowed_edges:
            allowed_str = "\n".join([f"  - {p} → {c}" for p, c in allowed_edges])
            skeleton_section += f"\n\n🔬 STATISTICAL CONSTRAINTS (from MMHC algorithm):\n"
            skeleton_section += f"\nALLOWED EDGES (statistically significant relationships):\n{allowed_str}\n"
            skeleton_section += f"\n⚠️ IMPORTANT: You MUST ONLY use edges from the allowed list above."
            skeleton_section += f"\nAny edge not in this list is FORBIDDEN (statistically independent)."
        
        if forbidden_pairs:
            n_forbidden = len(forbidden_pairs)
            skeleton_section += f"\n\nFORBIDDEN PAIRS: {n_forbidden} variable pairs have been found statistically independent."
    
    # 添加失败尝试的历史信息
    failed_attempts_section = ""
    if failed_attempts:
        failed_attempts_section = "\n\n⚠️ PREVIOUS FAILED ATTEMPTS:\n"
        failed_attempts_section += "You have made the following attempts that were REJECTED:\n\n"
        
        for i, attempt in enumerate(failed_attempts, 1):
            failed_attempts_section += f"Attempt {i}:\n"
            
            # 显示错误原因
            if 'error' in attempt:
                failed_attempts_section += f"  ❌ Rejection reason: {attempt['error']}\n"
            
            # 显示环路信息（如果有）
            if 'cycle_path' in attempt and attempt['cycle_path']:
                cycle_str = " → ".join(attempt['cycle_path'])
                failed_attempts_section += f"  🔄 Cycle detected: {cycle_str}\n"
            
            # 显示提出的图结构（关键边）
            if 'graph' in attempt and attempt['graph'] is not None:
                nodes = attempt['graph'].get('nodes', [])
                if nodes:
                    edges = []
                    cycle_nodes = set(attempt.get('cycle_path', [])) if attempt.get('cycle_path') else set()
                    
                    for node in nodes:
                        for parent in node.get('parents', []):
                            edge_str = f"    {parent} → {node['name']}"
                            # 标记参与环路的边
                            if cycle_nodes and parent in cycle_nodes and node['name'] in cycle_nodes:
                                edge_str += "  ⚠️ (part of cycle)"
                            edges.append(edge_str)
                    
                    if edges:
                        # 优先显示参与环路的边
                        cycle_edges = [e for e in edges if "part of cycle" in e]
                        other_edges = [e for e in edges if "part of cycle" not in e]
                        
                        failed_attempts_section += f"  Proposed {len(edges)} edge(s):\n"
                        
                        # 显示所有环路相关的边
                        if cycle_edges:
                            failed_attempts_section += "\n".join(cycle_edges) + "\n"
                        
                        # 显示其他边（限制数量）
                        if other_edges:
                            max_other = 15  # 最多显示15条其他边
                            if len(other_edges) <= max_other:
                                failed_attempts_section += "\n".join(other_edges) + "\n"
                            else:
                                failed_attempts_section += "\n".join(other_edges[:max_other]) + "\n"
                                failed_attempts_section += f"    ... and {len(other_edges)-max_other} more edges\n"
                else:
                    failed_attempts_section += f"  (Graph structure not available - parsing may have failed)\n"
            else:
                failed_attempts_section += f"  (Graph structure not available - parsing may have failed)\n"
            
            failed_attempts_section += "\n"
        
        failed_attempts_section += "⚠️ IMPORTANT: Please generate a DIFFERENT graph structure that:\n"
        failed_attempts_section += "  - Does NOT contain any cycles (must be a valid DAG)\n"
        failed_attempts_section += "  - Avoids the problematic edge patterns from previous attempts\n"
        failed_attempts_section += "  - Uses different causal reasoning to break the cycles\n\n"
    
    prompt = f"""Generate an initial causal graph hypothesis for the {domain_name} domain.

Variables to analyze:
{variables_formatted}
{context_section}
{baseline_section}
{evidence_section}
{skeleton_section}
{failed_attempts_section}

Instructions:
1. **THINK FIRST**: Analyze the domain and relationships before proposing a structure
2. For each variable, determine its DIRECT CAUSES (parent variables)
3. {"**RESPECT STATISTICAL CONSTRAINTS**: Only use allowed edges from the skeleton" if skeleton_constraints else "Consider only direct causal relationships"}
4. Ensure the graph is a DAG (no cycles) - {"LEARN from the failed attempts above and avoid creating similar cycles" if failed_attempts else "this is critical"}
5. Base reasoning on domain knowledge and temporal ordering

Output Format (IMPORTANT - use this exact structure):
{{
"reasoning": "Step-by-step reasoning about the causal structure before proposing the graph. Explain your thought process, domain knowledge, and why certain relationships exist or don't exist.",
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
- Put "reasoning" FIRST before "nodes" to encourage thinking before outputting
- Use "nodes" as a LIST (not a dictionary)
- Each node must have "name" and "parents" fields
- "parents" must be a list (use [] for root nodes)
- Output ONLY valid JSON"""

    return prompt


def construct_amendment_prompt(
    variable_list: List[str],
    domain_name: str,
    domain_context: str,
    previous_graph: Dict,
    memory: Optional[str],
    interventional_evidence: Optional[str] = None,
    previous_reasoning: Optional[str] = None,
    confirmed_edges: List[str] = None,
    edge_notes: Dict[str, str] = None
) -> str:
    """构建全局修正的提示词"""
    
    variables_formatted = "\n".join([f"- {var}" for var in variable_list])
    
    # 格式化上一轮的图
    prev_edges = []
    for node in previous_graph['nodes']:
        for parent in node.get('parents', []):
            prev_edges.append(f"  {parent} → {node['name']}")
    
    prev_graph_str = "\n".join(prev_edges) if prev_edges else "  (no edges)"
    
    # 格式化历史推理
    prev_reasoning_section = ""
    if previous_reasoning:
        prev_reasoning_section = f"\n🧠 YOUR EXPERIMENTAL INTENT & REASONING:\n{previous_reasoning}\n"

    # 格式化因果档案 - 修正：不再在修正阶段显示 edge_notes，避免基于“缺乏证据”的盲目删除
    dossier_section = ""
    if confirmed_edges:
        dossier_section = "\n📜 CAUSAL DOSSIER (Accumulated Knowledge):\n"
        dossier_section += "Confirmed Relationships (Highly certain):\n"
        dossier_section += "\n".join([f"  - {e}" for e in confirmed_edges]) + "\n"
    
    # 格式化记忆
    memory_section = ""
    if memory:
        memory_section = f"""
################################################################################
⚠️ PREVIOUS FAILED ATTEMPTS & FEEDBACK (CRITICAL: DO NOT REPEAT):
{memory}
################################################################################
"""
    
    # 添加干预证据信息
    evidence_section = ""
    if interventional_evidence:
        evidence_section = f"\n🔬 INTERVENTIONAL EVIDENCE (from experiments):\n{interventional_evidence}\n"
    
    context_section = ""
    if domain_context:
        context_section = f"\n\nDomain Context:\n{domain_context}\n"
    
    prompt = f"""Refine the causal graph for the {domain_name} domain based on previous results.

Variables:
{variables_formatted}
{context_section}
{prev_reasoning_section}
{dossier_section}
{evidence_section}

Previous Graph (Iteration {previous_graph['metadata']['iteration']}):
{prev_graph_str}

Previous Log-Likelihood: {previous_graph['metadata'].get('log_likelihood', 'N/A')}
{memory_section}

Instructions:
1. **THINK FIRST**: Carefully analyze the 🔬 INTERVENTIONAL EVIDENCE. Only make changes (Add/Remove) that are supported by experimental data or strong domain logic.
2. **LEARN FROM MISTAKES**: Review the ⚠️ Previous Feedback/Failed Attempts section carefully. Do NOT repeat the same operations or similar patterns that were previously rejected.
3. **Review Dossier**: Respect the confirmed edges. Do NOT delete confirmed edges unless evidence strongly contradicts them.
3. **Evidence-Based Amendments**: If no experiment has been performed for a specific edge, keep it as it is in the Previous Graph. Do not delete tentative edges without evidence.
4. **Update Dossier (STRICT)**:
   - Add relationships you are now SURE about to "confirmed_edges" (format: "A -> B").
   - For any relationship you are UNCERTAIN about, do NOT delete it; instead, add it to "edge_notes" to flag it for future testing.
5. Ensure the result is still a valid DAG

Output Format (use exact structure):
{{
"reasoning": "Detailed reasoning...",
"confirmed_edges": ["VarA -> VarB"],
"edge_notes": {{"VarX -> VarY": "Reasoning for keeping/tentative status"}},
"nodes": [
{{
    "name": "VariableName",
    "parents": ["Parent1", "Parent2"]
}}
]
}}

CRITICAL: 
- Put "reasoning" FIRST
- Output ONLY valid JSON"""

    return prompt

def construct_local_amendment_prompt(
    variable_list: List[str],
    domain_name: str,
    domain_context: str,
    previous_graph: Dict,
    memory: Optional[str],
    num_edge_operations: int,
    exclude_operations: List[tuple] = None,
    skeleton_constraints: Optional[Dict] = None,
    interventional_evidence: Optional[str] = None,
    previous_reasoning: Optional[str] = None,
    confirmed_edges: List[str] = None,
    edge_notes: Dict[str, str] = None,
    candidate_operations: List[Dict] = None
) -> str:
    """构建局部修正的提示词，限制 LLM 只能从候选操作中选择"""
    
    variables_formatted = "\n".join([f"- {var}" for var in variable_list])
    
    # 格式化当前图的边
    current_edges = []
    for node in previous_graph['nodes']:
        for parent in node.get('parents', []):
            current_edges.append(f"  {parent} → {node['name']}")
    
    current_graph_str = "\n".join(current_edges) if current_edges else "  (no edges)"
    
    # 格式化历史推理
    prev_reasoning_section = ""
    if previous_reasoning:
        prev_reasoning_section = f"\n🧠 YOUR EXPERIMENTAL INTENT & REASONING:\n{previous_reasoning}\n"

    # 格式化因果档案 - 修正：不再在局部修正阶段显示 edge_notes，避免基于“缺乏证据”的盲目删除
    dossier_section = ""
    if confirmed_edges:
        dossier_section = "\n📜 CAUSAL DOSSIER (Accumulated Knowledge):\n"
        dossier_section += "Confirmed Relationships (Highly certain):\n"
        dossier_section += "\n".join([f"  - {e}" for e in confirmed_edges]) + "\n"
    
    # 格式化候选操作 (关键限制)
    candidates_section = ""
    if candidate_operations:
        candidates_section = "\n📋 CANDIDATE OPERATIONS TO EVALUATE:\n"
        for i, op in enumerate(candidate_operations, 1):
            candidates_section += f"{i}. {op['type']}: {op['parent']} → {op['child']} (Reasoning: {op['reasoning']})\n"
    elif interventional_evidence:
        candidates_section = "\n⚠️ WARNING: No candidate operations were proposed in the experiment phase. You should generally maintain the status quo unless evidence is overwhelming.\n"
    else:
        # No candidates and no evidence (Knowledge-Driven mode)
        candidates_section = ""

    # 格式化记忆
    memory_section = ""
    if memory:
        memory_section = f"""
################################################################################
⚠️ PREVIOUS FAILED ATTEMPTS & FEEDBACK (CRITICAL: DO NOT REPEAT):
{memory}
################################################################################
"""
    
    # 添加干预证据信息
    evidence_section = ""
    if interventional_evidence:
        evidence_section = f"\n🔬 INTERVENTIONAL EVIDENCE (from experiments):\n{interventional_evidence}\n"
    
    context_section = ""
    if domain_context:
        context_section = f"\n\nDomain Context:\n{domain_context}\n"
    
    # 添加骨架约束提示（如果有）
    skeleton_note = ""
    if skeleton_constraints:
        skeleton_note = f"""
⚠️ STATISTICAL SKELETON CONSTRAINTS ARE ACTIVE:
- Only edges found in the MMHC skeleton are allowed
- You should prioritize edges that are statistically significant
"""
    
    prompt = f"""Perform LOCAL amendments to the causal graph for the {domain_name} domain.

Variables:
{variables_formatted}
{context_section}
{prev_reasoning_section}
{dossier_section}
{candidates_section}
{evidence_section}

Current Graph (Iteration {previous_graph['metadata']['iteration']}):
{current_graph_str}

Current Log-Likelihood: {previous_graph['metadata'].get('log_likelihood', 'N/A')}
{skeleton_note}
{memory_section}
"""

    # 添加已排除操作的说明（如果有）
    excluded_note = ""
    if exclude_operations:
        excluded_by_type = {'ADD': [], 'DELETE': [], 'REVERSE': []}
        for op_type, p, c in exclude_operations:
            excluded_by_type[op_type].append(f"{p}→{c}")
        
        excluded_parts = []
        for op_type in ['ADD', 'DELETE', 'REVERSE']:
            if excluded_by_type[op_type]:
                excluded_parts.append(f"{op_type}: {', '.join(excluded_by_type[op_type])}")
        
        if excluded_parts:
            excluded_note = f"""
⚠️ EXCLUDED Operations (previously failed, don't try again):
{'; '.join(excluded_parts)}

"""
    
    prompt += f"""
========================================
Refinement Instructions:
========================================
"""

    if candidate_operations:
        prompt += f"""1. **Analyze Candidates & Evidence**: Carefully evaluate each operation in the 📋 CANDIDATE OPERATIONS list using the 🔬 INTERVENTIONAL EVIDENCE and your 🧠 EXPERIMENTAL INTENT.
2. **Decision Rule (STRICT EVIDENCE-FIRST POLICY)**: 
   - You must treat 🔬 INTERVENTIONAL EVIDENCE as the Ground Truth. If domain knowledge conflicts with experimental evidence, **EVIDENCE WINS**.
   - You are ONLY allowed to output operations that were listed as CANDIDATES.
   - For each candidate, follow this Decision Matrix:

     | Candidate Op | Experiment Result (do(X) -> Y) | Decision | Reason |
     |--------------|-------------------------------|----------|--------|
     | ADD X -> Y   | SIGNIFICANT effect            | CONSIDER | Evidence confirms X is an ANCESTOR. Check for intermediate variables before adding. |
     | ADD X -> Y   | NO significant effect         | DISCARD  | Evidence fails to support any causal path. |
     | DELETE X -> Y| NO significant effect         | APPLY    | Evidence confirms the link is spurious. |
     | DELETE X -> Y| SIGNIFICANT effect            | DISCARD  | **STRICT FORBIDDEN**: Cannot delete a link with significant effect. |
     | REVERSE X->Y | do(X)->Y < do(Y)->X           | APPLY    | Reversed direction is better supported. |

3. **Direct vs. Indirect Causality (CRITICAL)**:
   - A significant `do(X) -> Y` effect only proves X is an **ANCESTOR** of Y, not necessarily a **DIRECT PARENT**.
   - **DO NOT** add a direct edge X -> Y if:
     a) There is a known physiological path between X and Y involving other variables in the current variable list.
     b) The effect of X on Y can be explained by a sequence of other edges (e.g., X -> Z -> Y).
   - If you only have Marginal Evidence (no successful conditional test), you must assume the link might be indirect if intermediate variables are present.

4. **Anti-Hallucination Guardrail**: 
   - Never use "domain knowledge" to justify deleting an edge that has a SIGNIFICANT interventional effect. 
   - If an experiment shows TVD > 0.1 or p-value < 0.05, the relationship is REAL for this specific dataset.
"""
    else:
        prompt += f"""1. **Local Refinement**: Analyze the current graph and identify specific edges that could be added, removed, or reversed to better reflect the causal structure of the {domain_name} domain.
2. **Domain Knowledge**: Use your domain knowledge and the provided context to propose improvements. Focus on direct causal relationships, temporal ordering, and plausible mechanisms.
3. **Strategic Selection**: Choose operations (ADD, DELETE, REVERSE) that you believe will most likely improve the graph's log-likelihood and accurately represent the system.
"""

    prompt += f"""4. **Update Dossier**: 
   - Add relationships you are now SURE about to "confirmed_edges" (format: "A -> B").
   - For relationships you are still UNCERTAIN about, do NOT delete them. Instead, add a note to "edge_notes" explaining what you need to test next.

{excluded_note}========================================
CRITICAL CONSTRAINTS:
========================================
"""
    if candidate_operations:
        prompt += "- ONLY use operations from the CANDIDATE list.\n"

    prompt += f"""- Propose at most {num_edge_operations} operations.
- Output ONLY valid JSON.

"""
    
    prompt += f"""Instructions:
**THINK FIRST**: Before proposing operations:
1. Analyze which changes would improve the likelihood based on the current structure and evidence.
2. Choose operations that make sense based on domain knowledge and interventional data.

IMPORTANT - REVERSE operation semantics:
- If you want to reverse edge A→B to B→A, specify "parent": "A", "child": "B"
- The parent and child in your JSON should match the EXISTING edge direction
- Example: Current edge X1→X2, to reverse it, use {{"type": "REVERSE", "parent": "X1", "child": "X2"}}
- This will change X1→X2 into X2→X1

Output Format (IMPORTANT - use this exact JSON structure):
{{
"overall_reasoning": "...",
"confirmed_edges": ["VarA -> VarB"],
"edge_notes": {{"VarX -> VarY": "Reasoning for keeping/tentative status"}},
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
]
}}

CRITICAL: 
- Put "overall_reasoning" FIRST to encourage strategic thinking
- Output ONLY valid JSON
- Include UP TO {num_edge_operations} operations
- Valid operation types: "ADD", "DELETE", "REVERSE"
- Each operation must have: type, parent, child, reasoning
- For REVERSE: always use the EXISTING edge direction (parent→child) in your JSON"""

    return prompt
def construct_experiment_proposal_prompt(
    variable_list: List[str],
    domain_name: str,
    domain_context: str,
    previous_graph: Optional[Dict],
    num_experiments: int = 3,
    edge_notes: Dict[str, str] = None
) -> str:
    """构建实验建议的提示词，要求 LLM 同时提出候选操作"""
    
    variables_formatted = "\n".join([f"- {var}" for var in variable_list])
    
    prev_graph_str = "  (no initial graph yet)"
    if previous_graph:
        prev_edges = []
        for node in previous_graph['nodes']:
            for parent in node.get('parents', []):
                prev_edges.append(f"  {parent} → {node['name']}")
        prev_graph_str = "\n".join(prev_edges) if prev_edges else "  (no edges)"

    # 格式化因果档案中的笔记，用于引导实验
    notes_section = ""
    if edge_notes:
        notes_section = "\n📜 YOUR PREVIOUS TENTATIVE NOTES (Use these to guide your experiments):\n"
        for edge, note in edge_notes.items():
            notes_section += f"  - {edge}: {note}\n"

    prompt = f"""You are investigating causal relationships in the {domain_name} domain.
Before finalizing the graph, you must propose candidate changes and design experiments to verify them.

Variables:
{variables_formatted}

Current Graph:
{prev_graph_str}
{notes_section}

Goal:
1. **Identify Candidate Changes**: Based on the graph and your notes, identify up to {num_experiments} specific operations (ADD, DELETE, or REVERSE) that you suspect will improve the graph.
2. **Verify with Experiments**: For EACH candidate operation, design a targeted interventional experiment to confirm if the causal influence is direct.
   - Example: If you suspect 'A -> B' should be DELETED, propose do(A) and observe B. If the experiment later shows no effect, the deletion is justified.

Instructions:
1. Prioritize relationships mentioned in your 'Tentative Notes'.
2. For each candidate operation, provide: type, parent, child, and reasoning.
3. For each experiment, provide: treatment (do-operator), target, and conditioning_set.

Output Format (JSON):
{{
  "reasoning": "Step-by-step logic for the proposed candidates and experiments.",
  "candidate_operations": [
    {{
      "type": "ADD",
      "parent": "VarA",
      "child": "VarB",
      "reasoning": "Why you suspect this relationship exists"
    }}
  ],
  "experiments": [
    {{
      "treatment": "VarA",
      "target": "VarB",
      "conditioning_set": ["VarC"]
    }}
  ]
}}
"""
    return prompt
