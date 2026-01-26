import httpx
from .llm_loader import LLMLoader
from utils import ConfigManager

class OpenAILLMLoader(LLMLoader):
    def __init__(self):
        config = ConfigManager()
        self.base_url = config.get("llm.openai.base_url")
        self.api_key = config.get("llm.openai.api_key")
        self.model_name = config.get("llm.openai.model_name")
        self.client = None

    def get_backend_type(self):
        return "openai(api)"

    def load_model(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(verify=False),
        )
        print(f"Loaded OpenAI model API: {self.model_name}")

    def generate(self, system_prompt, user_prompt, temperature = 0.7):
        """调用OpenAI API"""
        assert self.client is not None, "OpenAI client is not loaded. Call load_model() first."
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content \
            if response.choices[0].message.content is not None else ""
