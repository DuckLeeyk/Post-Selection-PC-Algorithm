import networkx as nx
import matplotlib.pyplot as plt


def visualize_cpdag(adj_matrix, ordering=None):
    """
    可视化由邻接矩阵表示的 CPDAG。

    参数：
      adj_matrix: 二维 numpy 数组，表示图的邻接矩阵。
                  如果 A[i, j] == 1 表示存在从 i 到 j 的边。
      ordering: 一维数组，每个元素为非负整数，表示节点所在的层次，
                数字越小的节点放在左侧，数字相同的节点依据垂直方向均匀分布，
                仅允许ordering较小的节点指向ordering较大的节点。
    """
    n = adj_matrix.shape[0]

    # 用于有向边的图
    G_dir = nx.DiGraph()
    # 用于无向边的图（仅考虑同层双向边）
    G_und = nx.Graph()

    # 添加所有节点（标签为节点编号）
    nodes = list(range(n))
    G_dir.add_nodes_from(nodes)
    G_und.add_nodes_from(nodes)

    if ordering is not None:

        # 边的处理：只保留ordering较小节点指向ordering较大的边，
        # 对于处于同一层且存在相互指向的边，视为无向边（仅添加一次）
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # 忽略自环
                if adj_matrix[i, j] == 1:
                    if ordering[i] < ordering[j]:
                        # 允许小层次节点指向大层次节点的有向边
                        G_dir.add_edge(i, j)
                    elif ordering[i] == ordering[j]:
                        # 在同一层次中，如果存在双向联系，则添加无向边
                        if adj_matrix[j, i] == 1 and i < j:
                            G_und.add_edge(i, j)
                    # 若 ordering[i] > ordering[j]，则不添加该边，因为违背层次顺序

        # 构造基于 ordering 的自定义布局
        # 同一层次的节点在x方向上位置一致，层次越小的节点位于左侧
        # 同一层次内的节点沿 y 方向均匀分布
        layers = {}  # key: 层次值, value: list of节点
        for idx, level in enumerate(ordering):
            layers.setdefault(level, []).append(idx)

        pos = {}
        x_gap = 2.0  # 水平间隔
        y_gap = 1.0  # 垂直间隔
        for level in sorted(layers.keys()):
            nodes_in_layer = layers[level]
            count = len(nodes_in_layer)
            # 为使得布局垂直居中，计算起始 y 坐标（使得总高度居中于0)
            y_start = (count - 1) / 2.0
            for i, node in enumerate(sorted(nodes_in_layer)):
                x = level * x_gap
                y = y_start - i * y_gap
                pos[node] = (x, y)

        plt.figure(figsize=(8, 6))
        plt.title("CPDAG with Custom Layered Layout")

    else:

        # 遍历邻接矩阵，判断边的类型
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # 忽略自环（如果有的话）
                if adj_matrix[i, j] == 1:
                    # 两个节点间若同时存在正反馈，则视为无向边，只添加一次
                    if adj_matrix[j, i] == 1:
                        if i < j:  # 防止重复添加无向边
                            G_und.add_edge(i, j)
                    else:
                        # 有向边，从 i 到 j
                        G_dir.add_edge(i, j)

        # 计算布局（这里选用 shell 布局，也可以根据需要修改其他布局算法）
        pos = nx.shell_layout(nodes)

        plt.figure(figsize=(8, 6))
        plt.title("CPDAG Visualization")


    # 绘制无向边（黑色实线）
    nx.draw_networkx_edges(
        G_und, pos,
        width=2,
        edge_color='black'
    )

    # 绘制有向边（红色箭头），使用特定箭头样式和尺寸
    nx.draw_networkx_edges(
        G_dir, pos,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        edge_color='red'
    )

    # 绘制节点（浅蓝色圆形）
    nx.draw_networkx_nodes(
        G_dir, pos,
        node_size=700,
        node_color='lightblue'
    )

    # 绘制节点标签
    nx.draw_networkx_labels(
        G_dir, pos,
        font_size=12,
        font_color='black'
    )

    plt.axis('off')
    plt.show()
