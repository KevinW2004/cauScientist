"""
交互式因果图可视化工具
使用 Pyvis 生成可拖拽、缩放的 HTML 交互图
"""

import os
from typing import Optional, List, Tuple
from pyvis.network import Network
import webbrowser
from schemas.causal_graph import StructuredGraph, CausalNode, GraphMetadata


class CausalGraphVisualizer:
    """因果图可视化器 - 使用 Pyvis 生成交互式 HTML"""
    
    def __init__(
        self,
        output_dir: str = "visualizations",
        auto_open: bool = True,
        height: str = "750px",
        width: str = "100%"
    ):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            auto_open: 是否自动在浏览器中打开
            height: 图的高度
            width: 图的宽度
        """
        self.output_dir = output_dir
        self.auto_open = auto_open
        self.height = height
        self.width = width
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize(
        self,
        structured_graph: StructuredGraph,
        filename: Optional[str] = None,
        previous_graph: Optional[StructuredGraph] = None,
        layout: str = "hierarchical"
    ) -> str:
        """
        生成交互式因果图
        
        Args:
            structured_graph: 结构化图数据（StructuredGraph schema）
            filename: 输出文件名（不含路径），默认自动生成
            previous_graph: 上一轮的图（用于高亮变化）
            layout: 布局方式 ("hierarchical" 或 "spring")
            
        Returns:
            生成的 HTML 文件路径
        """
        
        metadata = structured_graph.metadata
        nodes = structured_graph.nodes
        
        # 生成文件名
        if filename is None:
            domain = getattr(metadata, 'domain', 'graph')
            iteration = getattr(metadata, 'iteration', 0)
            filename = f"{domain}_iter_{iteration}.html"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 创建 Pyvis 网络
        net = Network(
            directed=True,
            height=self.height,
            width=self.width,
            notebook=False,
            bgcolor="#ffffff"
        )
        
        # 获取边的变化信息
        added_edges, removed_edges = self._get_edge_changes(
            structured_graph, previous_graph
        )
        
        # 添加节点
        self._add_nodes(net, nodes)
        
        # 添加边（当前图中的边）
        self._add_edges(net, nodes, added_edges)
        
        # 添加删除的边（如果有）
        if removed_edges:
            self._add_removed_edges(net, removed_edges)
        
        # 设置布局选项
        self._set_layout(net, layout)
        
        # 添加标题信息
        title = self._generate_title(metadata, added_edges, removed_edges)
        net.heading = title
        
        # 保存并可选打开
        net.save_graph(output_path)
        
        print(f"\n✓ 交互式因果图已保存: {output_path}")
        
        if self.auto_open:
            webbrowser.open('file://' + os.path.abspath(output_path))
            print(f"✓ 已在浏览器中打开")
        
        return output_path
    
    def _add_nodes(self, net: Network, nodes: List[CausalNode]):
        """添加节点"""
        for node in nodes:
            node_name = node.name
            parents = node.parents
            
            # 节点标签
            label = node_name
            
            # 节点悬停信息
            num_parents = len(parents)
            if num_parents > 0:
                title = f"<b>{node_name}</b><br>父节点: {', '.join(parents)}"
            else:
                title = f"<b>{node_name}</b><br>根节点（无父节点）"
            
            # 根据父节点数量设置节点大小
            size = 20 + num_parents * 5
            
            # 根节点用不同颜色
            if num_parents == 0:
                color = "#ff9999"  # 浅红色（根节点）
            else:
                color = "#97c2fc"  # 浅蓝色（普通节点）
            
            net.add_node(
                node_name,
                label=label,
                title=title,
                size=size,
                color=color,
                font={'size': 16, 'face': 'Arial'}
            )
    
    def _add_edges(
        self,
        net: Network,
        nodes: List[CausalNode],
        added_edges: set
    ):
        """添加当前图的边"""
        for node in nodes:
            child = node.name
            parents = node.parents
            
            for parent in parents:
                edge_tuple = (parent, child)
                
                # 判断是否是新增边
                if edge_tuple in added_edges:
                    # 新增边：绿色，粗线
                    color = "#00cc00"
                    width = 3
                    title = f"<b>新增:</b> {parent} → {child}"
                else:
                    # 不变边：灰色
                    color = "#666666"
                    width = 2
                    title = f"{parent} → {child}"
                
                net.add_edge(
                    parent,
                    child,
                    color=color,
                    width=width,
                    title=title,
                    arrows='to',
                    smooth={'type': 'curvedCW', 'roundness': 0.2}
                )
    
    def _add_removed_edges(self, net: Network, removed_edges: set):
        """添加删除的边（用虚线表示）"""
        for parent, child in removed_edges:
            net.add_edge(
                parent,
                child,
                color="#ff0000",
                width=2,
                title=f"<b>删除:</b> {parent} → {child}",
                dashes=True,
                arrows='to',
                smooth={'type': 'curvedCW', 'roundness': 0.2}
            )
    
    def _get_edge_changes(
        self,
        current_graph: StructuredGraph,
        previous_graph: Optional[StructuredGraph]
    ) -> Tuple[set, set]:
        """
        获取边的变化
        
        Returns:
            (added_edges, removed_edges)
        """
        if previous_graph is None:
            return set(), set()
        
        # 从 change_history 中获取最后一次变化
        change_history = current_graph.metadata.change_history
        if not change_history:
            return set(), set()
        
        # 只显示最后一次变化
        last_change = change_history[-1]
        added_edges = set()
        removed_edges = set()
        
        edge = (last_change.parent, last_change.child)
        if last_change.type == "ADD":
            added_edges.add(edge)
        elif last_change.type == "DELETE":
            removed_edges.add(edge)
        elif last_change.type == "REVERSE":
            # 反转操作：删除 parent→child，添加 child→parent
            removed_edges.add(edge)
            added_edges.add((last_change.child, last_change.parent))
        
        return added_edges, removed_edges
    
    def _set_layout(self, net: Network, layout: str = "hierarchical"):
        """设置图的布局"""
        
        if layout == "hierarchical":
            # 层次布局（自顶向下，适合 DAG）
            options = """
            {
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "direction": "UD",
                  "sortMethod": "directed",
                  "nodeSpacing": 150,
                  "levelSeparation": 200
                }
              },
              "physics": {
                "enabled": false
              },
              "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "hover": true,
                "tooltipDelay": 100
              },
              "edges": {
                "smooth": {
                  "type": "cubicBezier",
                  "forceDirection": "vertical"
                }
              }
            }
            """
        else:
            # 弹簧布局（物理模拟）
            options = """
            {
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.3,
                  "springLength": 200,
                  "springConstant": 0.04
                },
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              },
              "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "hover": true,
                "tooltipDelay": 100
              }
            }
            """
        
        net.set_options(options)
    
    def _generate_title(
        self,
        metadata: GraphMetadata,
        added_edges: set,
        removed_edges: set
    ) -> str:
        """生成图的标题"""
        
        domain = metadata.domain
        iteration = metadata.iteration
        num_vars = metadata.num_variables
        num_edges = metadata.num_edges
        
        title = f"因果图 - {domain.upper()} (Iteration {iteration})"
        title += f"<br>变量数: {num_vars} | 边数: {num_edges}"
        
        if added_edges or removed_edges:
            title += f" | 变化: "
            if added_edges:
                title += f"+{len(added_edges)} "
            if removed_edges:
                title += f"-{len(removed_edges)}"
        
        return title
    
    def visualize_comparison(
        self,
        graphs: List[StructuredGraph],
        filename: str = "comparison.html"
    ):
        """
        生成多个图的对比视图（并排显示）
        
        Args:
            graphs: 图列表
            filename: 输出文件名
        """
        # 这个功能可以后续扩展，用于对比不同迭代的图
        # 目前先实现单图可视化
        pass


def visualize_causal_graph(
    structured_graph: StructuredGraph,
    output_dir: str = "experiment_results/visualizations",
    filename: Optional[str] = None,
    previous_graph: Optional[StructuredGraph] = None,
    auto_open: bool = True,
    layout: str = "hierarchical"
) -> str:
    """
    快捷函数：生成交互式因果图
    
    Args:
        structured_graph: 结构化图数据（StructuredGraph schema）
        output_dir: 输出目录
        filename: 输出文件名
        previous_graph: 上一轮的图
        auto_open: 是否自动打开浏览器
        layout: 布局方式 ("hierarchical" 或 "spring")
        
    Returns:
        生成的 HTML 文件路径
    
    Example:
        >>> from utils.graph_visualizer import visualize_causal_graph
        >>> visualize_causal_graph(
        ...     structured_graph,
        ...     output_dir="cma_experiments/00_sachs",
        ...     filename="graph_t0.html"
        ... )
    """
    
    visualizer = CausalGraphVisualizer(
        output_dir=output_dir,
        auto_open=auto_open
    )

    return visualizer.visualize(
        structured_graph=structured_graph,
        filename=filename,
        previous_graph=previous_graph,
        layout=layout
    )


def visualize_graph(
    structured_graph: StructuredGraph,
    filename: str | None = None,
    output_dir: str = "experiment_results/visualizations",
    previous_graph: Optional[StructuredGraph] = None,
    auto_open: bool = False,
    text_only: bool = True
):
    """
    可视化因果图（支持文本和交互式HTML两种方式）

    Args:
        structured_graph: 结构化图数据（StructuredGraph schema）
        output_dir: HTML输出目录
        previous_graph: 上一轮的图（用于高亮变化）
        auto_open: 是否自动在浏览器打开HTML
        text_only: 是否仅输出文本（不生成HTML）
    """
    # 文本可视化
    print("\n" + "=" * 60)
    print(f"CAUSAL GRAPH - {structured_graph.metadata.domain.upper()}")
    print("=" * 60)
    print(f"Iteration: {structured_graph.metadata.iteration}")
    print(f"Variables: {structured_graph.metadata.num_variables}")
    print(f"Edges: {structured_graph.metadata.num_edges}")

    # 显示变化
    change_history = structured_graph.metadata.change_history
    if change_history:
        last_change = change_history[-1]
        print(f"\nLast change:")
        print(f"  Type: {last_change.type}")
        if last_change.type == "REVERSE":
            print(f"  Reversed: {last_change.parent} → {last_change.child}")
            print(f"  Now: {last_change.child} → {last_change.parent}")
        else:
            print(f"  Edge: {last_change.parent} → {last_change.child}")
        if last_change.reasoning:
            reasoning = last_change.reasoning[:100]
            print(f"  Reasoning: {reasoning}..." if len(last_change.reasoning) > 100 else f"  Reasoning: {reasoning}")

    # print("\nReasoning:")
    # reasoning = structured_graph['metadata']['reasoning']
    # print(reasoning[:300] + "..." if len(reasoning) > 300 else reasoning)

    print("\n" + "-" * 60)
    print("CAUSAL RELATIONSHIPS:")
    print("-" * 60)

    # 显示边
    edges = []
    root_nodes = []

    for node in structured_graph.nodes:
        parents = node.parents
        if parents:
            for parent in parents:
                edges.append(f"  {parent} → {node.name}")
        else:
            root_nodes.append(node.name)

    if root_nodes:
        print("\nRoot Nodes (no parents):")
        for node in root_nodes:
            print(f"  • {node}")

    if edges:
        print("\nCausal Edges:")
        for edge in sorted(edges):
            print(edge)

    print("=" * 60 + "\n")

    # 交互式HTML可视化
    if not text_only:
        try:
            visualize_causal_graph(
                structured_graph=structured_graph,
                output_dir=output_dir,
                filename=filename,
                previous_graph=previous_graph,
                auto_open=auto_open,
                layout="hierarchical",
            )
        except Exception as e:
            print(f"⚠️  警告: 无法生成交互式可视化: {e}")
            print(f"   (你可能需要安装 pyvis: pip install pyvis networkx)")
