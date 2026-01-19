from transformers import AutoTokenizer
from .llm_loader import LLMLoader
from utils import ConfigManager


class LocalLLMLoader(LLMLoader):
    def __init__(self):
        config = ConfigManager()
        self.model_path = config.get("llm.local.model_path")
        self.model = None
        self.tokenizer = None
        self.device = config.get("llm.local.device", "cpu")
        self.max_tokens = config.get("llm.local.max_tokens", 8192)

    def get_backend_type(self) -> str:
        return "local(vllm)"

    def load_model(self):
            """加载本地模型（使用 vLLM 加速）"""
            from vllm import LLM
            import os
            
            # 验证路径
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model path does not exist: {self.model_path}")
            
            print(f"\nLoading local model with vLLM from {self.model_path}...")
            print("This may take a few minutes...")
            
            # 使用 vLLM 加载模型
            self.model = LLM(
                model=self.model_path,
                dtype="bfloat16",  # 或 "float16"
                tensor_parallel_size=1,  # 如果有多张GPU，可以增加这个值
                gpu_memory_utilization=0.9,  # 使用90%的GPU显存
                trust_remote_code=True,  # Qwen 模型需要这个
                # max_model_len=8192,  # 根据需要调整最大长度
                # enforce_eager=True,  # 如果遇到问题，可以取消注释这行
            )
            
            # vLLM 内部已经包含 tokenizer，不需要单独加载
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)  # vLLM 内部处理
            print(f"\n✓ vLLM model ready!")
    
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        temperature: float = 0.7,
    ) -> str:
        """使用 vLLM 进行推理"""
        from vllm import SamplingParams
        
        # 禁用思考链输出
        system_prompt += "\n\nIMPORTANT: Output ONLY the JSON result. Do NOT include <think> tags or reasoning before the JSON."
        
        # print("Input system_prompt:", system_prompt[:200], "...")
        # print("Input user_prompt:", user_prompt[:200], "...")
        
        # 构建对话（Qwen 格式）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 手动应用 chat template（因为 vLLM 需要字符串输入）
        # Qwen 的 chat template
        # prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        print("begin call llm, prompt:",prompt)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=self.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],  # Qwen 的停止符
            skip_special_tokens=True
        )
        
        # 生成
        outputs = self.model.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        print("\n\n\n end call llm, output:", response_text)
        
        return response_text.strip()
