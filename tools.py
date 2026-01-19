    def get_top_edges(
        self,
        baseline_results: Dict,
        variable_list: List[str],
        top_k: int = 20
    ) -> List[Tuple[str, str, float, str]]:
        """
        获取所有方法中最强的边（用于programmatic使用）
        
        Returns:
            List of (var1, var2, score, method_name)
        """
        all_edges = []
        
        for method_name, matrix in baseline_results.items():
            n_vars = len(variable_list)
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        all_edges.append((
                            variable_list[i],
                            variable_list[j],
                            matrix[i, j],
                            method_name
                        ))
        
        # 按分数排序
        all_edges.sort(key=lambda x: x[2], reverse=True)
        
        return all_edges[:top_k]