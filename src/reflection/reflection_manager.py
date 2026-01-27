
from schemas import StructuredGraph
from llm_loader import LLMLoader
from utils.llm import construct_reflection_prompt
from utils import SingletonMeta

class ReflectionManager(metaclass=SingletonMeta):
    def __init__(self, llm_loader: LLMLoader | None = None):
        self._llm_loader = llm_loader
        self.current_reflection = "No reflections yet."

    @property
    def llm_loader(self) -> LLMLoader:
        if self._llm_loader is None:
            raise ValueError("LLM Loader has not been set for ReflectionManager.")
        return self._llm_loader

    def generate_reflection(
        self,
        domain_name: str,
        domain_context: str,
        current_graph: StructuredGraph,
        score_diff: float,
        is_better: bool
    ) -> None:
        """生成对当前图的反思"""

        system_prompt, user_prompt = construct_reflection_prompt(
            domain_name,
            domain_context,
            current_graph,
            score_diff,
            is_better,
            self.current_reflection
        )
        res = self.llm_loader.generate(system_prompt, user_prompt)
        self.current_reflection = res
        # 测试用：保存到文件看看反思历史
        self._save_to_file()
        return
    
    # ==== 辅助函数 ====
    def _save_to_file(self) -> None:
        from utils import ConfigManager
        file_path = ConfigManager().get("experiment.output.dir", ".") + "/reflection_history.txt"
        with open(file_path, "a") as f:
            f.write(self.current_reflection + "\n" + "="*35 + "\n")



