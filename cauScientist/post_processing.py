"""
Post-processing Module
生成记忆(Memory),总结变化和影响
支持共享模型实例
"""

import json
from typing import Dict, Optional


class PostProcessor:
    """后处理器 - 支持共享模型实例"""
    
    def __init__(
        self, 
        model_type: str = "openai",
        base_url: str = None, 
        api_key: str = None,
        model_path: str = None,
        tokenizer=None,  # 共享tokenizer
        model=None       # 共享model
    ):
        """初始化后处理器"""
        self.model_type = model_type
        
        if model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url or "http://35.220.164.252:3888/v1/",
                api_key=api_key or "sk-x1DLgF9tW1t2IwCrUFyCfIIYFookGgO4qseCxb9vefNHQPcp"
            )
            self.tokenizer = None
            self.model = None
            
        elif model_type == "local":
            self.client = None
            
            # 使用共享的模型实例（推荐）
            if model is not None:
                print("[PostProcessor] Using shared model instance")
                self.tokenizer = tokenizer
                self.model = model
            else:
                # 独立加载
                if model_path is None:
                    raise ValueError("model_path must be provided when not using shared model")
                print("[PostProcessor] Loading independent model instance")
                self._init_local_model(model_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def _init_local_model(self, model_path: str):
        """初始化本地模型（独立实例）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading local model for PostProcessor from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print("✓ Local model loaded for PostProcessor")
    
    def generate_memory(
        self,
        current_graph: Dict,
        current_results: Dict,
        previous_graph: Optional[Dict] = None,
        previous_results: Optional[Dict] = None,
        domain_name: str = "unknown",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """生成记忆 μ_t"""
        
        if previous_graph is None or previous_results is None:
            return self._generate_initial_memory(current_graph, current_results, domain_name)
        
        return self._generate_comparative_memory(
            current_graph, current_results,
            previous_graph, previous_results,
            domain_name, model, temperature, max_tokens
        )
    
    def _generate_initial_memory(
        self,
        current_graph: Dict,
        current_results: Dict,
        domain_name: str
    ) -> str:
        """生成初始记忆(t=0)"""
        
        memory = f"""Initial Hypothesis (Iteration 0):
- Domain: {domain_name}
- Number of edges: {current_graph['metadata']['num_edges']}
- Log-likelihood: {current_results['log_likelihood']:.4f}
- Reasoning: {current_graph['metadata']['reasoning'][:200]}...

This is the baseline hypothesis. Future iterations will build upon this structure.
"""
        return memory
    
    def _generate_comparative_memory(
        self,
        current_graph: Dict,
        current_results: Dict,
        previous_graph: Dict,
        previous_results: Dict,
        domain_name: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """生成对比记忆(t≥1)"""
        
        changes = current_graph['metadata'].get('changes', {})
        
        # 使用 CV score（如果有）或普通 LL
        curr_ll = current_results.get('cv_log_likelihood', current_results.get('log_likelihood', 0))
        prev_ll = previous_results.get('cv_log_likelihood', previous_results.get('log_likelihood', 0))
        ll_change = curr_ll - prev_ll
        
        # 获取训练集 LL（如果有）
        curr_train_ll = current_results.get('train_log_likelihood', None)
        prev_train_ll = previous_results.get('train_log_likelihood', None)
        
        changes_desc = self._format_changes(changes)
        
        # 构建 LL 信息
        if curr_train_ll is not None:
            curr_ll_str = f"CV Log-likelihood: {curr_ll:.4f} (Train: {curr_train_ll:.4f})"
            prev_ll_str = f"CV Log-likelihood: {prev_ll:.4f} (Train: {prev_train_ll:.4f})" if prev_train_ll else f"Log-likelihood: {prev_ll:.4f}"
        else:
            curr_ll_str = f"Log-likelihood: {curr_ll:.4f}"
            prev_ll_str = f"Log-likelihood: {prev_ll:.4f}"
        
        prompt = f"""Analyze the changes between two iterations of causal graph discovery for the {domain_name} domain.

PREVIOUS ITERATION (t={previous_graph['metadata']['iteration']}):
- Number of edges: {previous_graph['metadata']['num_edges']}
- {prev_ll_str}

CURRENT ITERATION (t={current_graph['metadata']['iteration']}):
- Number of edges: {current_graph['metadata']['num_edges']}
- {curr_ll_str}

CHANGES:
{changes_desc}

VALIDATION SCORE CHANGE: {ll_change:+.4f} ({'improvement' if ll_change > 0 else 'decline'})

CURRENT REASONING:
{current_graph['metadata']['reasoning']}

Please provide a concise analysis (3-5 sentences) covering:
1. What changes were made and why they might be significant
2. How these changes affected model fit (validation score)
3. Recommendations for the next iteration

Focus on domain-specific insights and potential improvements."""

        system_prompt = f"You are an expert in causal inference and {domain_name}."
        
        if self.model_type == "openai":
            memory = self._call_openai(system_prompt, prompt, model, temperature)
        else:
            memory = self._call_local_model(system_prompt, prompt, temperature, max_tokens)
        
        summary = f"""
Iteration {current_graph['metadata']['iteration']} Summary:
- Edges: {previous_graph['metadata']['num_edges']} → {current_graph['metadata']['num_edges']}
- CV Score: {prev_ll:.4f} → {curr_ll:.4f} (Δ={ll_change:+.4f})
- Changes: {changes['num_added']} added, {changes['num_removed']} removed

Analysis:
{memory}
"""
        
        return summary
    
    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float
    ) -> str:
        """调用OpenAI API"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    # def _call_local_model(
    #     self,
    #     system_prompt: str,
    #     user_prompt: str,
    #     temperature: float,
    #     max_tokens: int
    # ) -> str:
    #     """调用本地模型"""
        
    #     messages = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]
        
    #     text = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
        
    #     model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
    #     generated_ids = self.model.generate(
    #         **model_inputs,
    #         max_new_tokens=max_tokens,
    #         temperature=temperature,
    #         do_sample=temperature > 0,
    #         top_p=0.95 if temperature > 0 else None,
    #         pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
    #     )
        
    #     generated_ids = [
    #         output_ids[len(input_ids):] 
    #         for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #     ]
        
    #     response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    #     return response_text.strip()

    def _call_local_model(self, system_prompt: str, user_prompt: str, 
                temperature: float, max_tokens: int ):
        """使用 vLLM 进行推理"""
        from vllm import SamplingParams
        
        # 禁用思考链输出
        # system_prompt += "\n\nIMPORTANT: Output ONLY the JSON result. Do NOT include <think> tags or reasoning before the JSON."
        
        print("begin call llm")
        # print("Input system_prompt:", system_prompt[:200], "...")
        # print("Input user_prompt:", user_prompt[:200], "...")
        
        # 构建对话（Qwen 格式）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 手动应用 chat template（因为 vLLM 需要字符串输入）
        # Qwen 的 chat template
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],  # Qwen 的停止符
            skip_special_tokens=True
        )
        
        # 生成
        outputs = self.model.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        print("\n\n\nend call llm, output:", response_text[:500], "...")
        
        return response_text.strip()
    
    def _format_changes(self, changes: Dict) -> str:
        """格式化变化描述"""
        
        if not changes:
            return "No changes recorded."
        
        lines = []
        
        if changes.get('added_edges'):
            lines.append(f"Added edges ({changes['num_added']}):")
            for parent, child in changes['added_edges']:
                lines.append(f"  + {parent} → {child}")
        
        if changes.get('removed_edges'):
            lines.append(f"Removed edges ({changes['num_removed']}):")
            for parent, child in changes['removed_edges']:
                lines.append(f"  - {parent} → {child}")
        
        if not lines:
            return "No structural changes."
        
        return "\n".join(lines)
    
    def save_memory(self, memory: str, filepath: str):
        """保存记忆到文件"""
        with open(filepath, 'w') as f:
            f.write(memory)
        print(f"[Post-processing] Memory saved to {filepath}")