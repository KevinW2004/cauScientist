import re

from schemas import StructuredGraph
from llm_loader import LLMLoader
from utils.llm import construct_reflection_prompt, construct_review_prompt
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
        is_better: bool,
    ) -> None:
        """生成对当前图的反思"""

        system_prompt, user_prompt = construct_reflection_prompt(
            domain_name,
            domain_context,
            current_graph,
            score_diff,
            is_better,
            self.current_reflection,
        )
        res = self.llm_loader.generate(system_prompt, user_prompt)
        res = self._format_llm_text(res)
        self.current_reflection = res
        # 测试用：保存到文件看看反思历史
        # self._save_to_file()
        return

    def generate_review(
        self, domain_name: str, domain_context: str, initial_graph: StructuredGraph, final_graph: StructuredGraph
    ) -> str:
        """
        对初始图和最终最优图进行对比总结, 结合变更路径回顾，生成成功总结文本
        
        :param domain_name: 领域名称
        :type domain_name: str
        :param domain_context: 领域背景说明
        :type domain_context: str
        :param initial_graph: 初始图
        :type initial_graph: StructuredGraph
        :param final_graph: 最终最优图
        :type final_graph: StructuredGraph
        :return: 成功总结文本
        :rtype: str
        """
        system_prompt, user_prompt = construct_review_prompt(
            domain_name,
            domain_context,
            initial_graph,
            final_graph,
        )
        res = self.llm_loader.generate(system_prompt, user_prompt)
        res = self._format_llm_text(res)
        return res

    # ==== 辅助函数 ====
    def _format_llm_text(self, reflection: str) -> str:
        """格式化带思考的llm回答文本，去除开头的think、多余的引号和空白"""
        # 跳过开头的 <think> </think> 部分，直到下一个字符开始
        reflection = re.sub(r"^<think>.*?</think>\s*", "", reflection, flags=re.DOTALL)
        # 去除开头和结尾的引号和空白
        reflection = reflection.strip().strip('"').strip("'").strip()
        return reflection

    def _save_to_file(self) -> None:
        from utils import ConfigManager

        file_path = (
            ConfigManager().get("experiment.output.dir", ".")
            + "/reflection_history.txt"
        )
        with open(file_path, "a") as f:
            f.write(self.current_reflection + "\n" + "=" * 35 + "\n")
