import numpy as np
from itertools import combinations
from math import log, sqrt
from copy import deepcopy


class PCAlgorithm:
    """
    使用 PC 算法推断因果CPDAG的示例性实现。

    参数:
    ----------
    X : np.ndarray
        数据矩阵, shape = [n_samples, d_vars].
    alpha : float
        显著性水平, 用于条件独立检验 (Fisher's Z 检验).
    ordering : list or np.ndarray, optional
        每个变量的时序信息 O(V)。若所有变量在同一层, 可置为 None 或全为同一值.
        ordering[i] = k 表示第i个变量处于第 k 层/时间点.
    """

    def __init__(self, X, alpha=0.05, ordering=None):
        self.X = X
        self.n, self.d = X.shape
        self.alpha = alpha
        # 如果没有传递 ordering，就认为所有变量处于相同层
        if ordering is None:
            self.ordering = np.zeros(self.d, dtype=int)
        else:
            self.ordering = np.array(ordering, dtype=int)

        # 初始化分离集 sepset[i][j] 用于存储在算法中分离 i-j 的条件节点
        self.sepset = [[set() for _ in range(self.d)] for __ in range(self.d)]

        # 初始图：完全无向图，用邻接矩阵(或邻接列表)表示都可以
        # 这里用邻接矩阵: 1表示有边, 0表示无边
        self.G = np.ones((self.d, self.d), dtype=int)
        np.fill_diagonal(self.G, 0)  # 自环为0

    def fit(self):
        """
        执行PC算法, 返回CPDAG的(邻接矩阵)表示。
        """
        # 第一步: 邻接搜索 (去边)
        self._remove_edges()

        # 第二步: 方向定向
        # (a) 识别 v-structures
        self._orient_colliders()
        # (b) 应用 Meek 规则 (R1-R4) 反复更新，直至稳定
        self._apply_meek_rules()

        # 返回最终CPDAG
        return self.G

    def _remove_edges(self):
        """
        通过逐渐增大条件集 S 的大小进行条件独立检验，
        删除独立边，并更新 sepset。
        """
        # s: 条件集合大小，逐步增加
        s = 0
        cont = True  # 用于循环控制

        # 为了实现PC-stable, 每轮要收集所有要删除的边
        while cont:
            cont = False
            # 复制当前邻接情况, 以防止本轮循环中动态修改影响顺序
            current_adjacency = [set(np.where(self.G[i] == 1)[0]) for i in range(self.d)]

            for i in range(self.d):
                # 先记录下来本轮开始时 i 的邻居集合
                Adj_i = deepcopy(current_adjacency[i])

                for j in list(Adj_i):
                    if j <= i:
                        # 无向图中, 为避免重复, 只需要在 i < j 时处理即可
                        continue

                    # 决定 i-j 是否要进行检验：只有当可供选择的邻居集合大小 >= s 时才可能找到大小为s的子集
                    # 同时需要排除在 ordering 上“比 i,j 都晚”的节点
                    O_bar_ij = self._get_future_nodes(i, j)
                    # 对 i 来说，可用于做条件的邻居
                    possible_parents_i = current_adjacency[i] - {j} - O_bar_ij

                    if len(possible_parents_i) >= s:
                        # 遍历 possible_parents_i 的所有大小为 s 的子集
                        for S in combinations(possible_parents_i, s):
                            S = set(S)
                            # 进行条件独立检验
                            independent = self._conditional_indep_test(i, j, S)
                            if independent:
                                # 删除边 i-j
                                self.G[i, j] = 0
                                self.G[j, i] = 0
                                # 记录分离集
                                self.sepset[i][j] = S
                                self.sepset[j][i] = S
                                cont = True
                                break  # 跳出子集循环
            s += 1

    def _get_future_nodes(self, i, j):
        """
        返回在时序上同时晚于 i 和 j 的所有节点集合 Ō_ij。
        """
        oi = self.ordering[i]
        oj = self.ordering[j]
        # 只有当 ordering[k] > oi 且 ordering[k] > oj 才算是同时晚于 i 和 j
        # 这里也可以写成: ordering[k] > max(oi, oj)
        future_nodes = set(idx for idx, ok in enumerate(self.ordering)
                           if (ok > oi and ok > oj))
        return future_nodes

    def _conditional_indep_test(self, i, j, S):
        """
        Fisher's Z 检验评估 X_i 与 X_j 在给定 X_S 条件下是否独立。
        返回 True 表示判断为独立（p-value > alpha），否则 False。

        这里给出了一个示例实现：
        1. 对 Xi, Xj 分别回归在 XS 之上，取残差 ri, rj
        2. 计算残差 ri, rj 的相关系数 r
        3. Fisher's Z 转换并对应检验
        """
        # 如果 S 为空
        if len(S) == 0:
            corr = np.corrcoef(self.X[:, i], self.X[:, j])[0, 1]
        else:
            # 对 Xi, Xj 各自回归 XS，取残差
            # 简单方法：用线性代数投影(也可用 sklearn 线性回归)
            XS = self.X[:, list(S)]
            Xi = self.X[:, i]
            Xj = self.X[:, j]

            # 为简化，假设 XS 非常数列，且行数>列数
            # 计算投影矩阵 P_S = X_S (X_S^T X_S)^{-1} X_S^T
            # 残差 = X - P_S * X
            # 注意要考虑 XS 可能只有一列或多列
            # 增加一个截距列
            ones = np.ones((self.n, 1))
            XS_ = np.hstack([ones, XS])

            # (X_S^T X_S)^{-1}
            inv_ = np.linalg.inv(XS_.T @ XS_)

            # 投影矩阵 P_S
            P_S = XS_ @ inv_ @ XS_.T

            # 残差
            ri = Xi - P_S @ Xi
            rj = Xj - P_S @ Xj
            # 计算相关系数
            corr = np.corrcoef(ri, rj)[0, 1]

        # 若相关系数的绝对值非常小(数值问题), 直接判为独立
        if abs(corr) < 1e-12:
            return True

        # Fisher's Z 检验
        z = 0.5 * log((1 + corr) / (1 - corr))
        # 自由度近似为 n - |S| - 3
        df = self.n - len(S) - 3
        # Z-score
        Z = sqrt(df) * z

        # 双侧检验
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(Z)))

        return p_value > self.alpha

    def _orient_colliders(self):
        """
        (a) v-structure 定向：
        对所有三元组 (i, k, j)，当 i-k, k-j 有边且 i-j 无边，
        并且 k 不在 sepset[i][j] 里，则定向 i→k←j。
        在定向时统筹时序信息：若 ordering 指定 i < k，则可直接 i→k；若 k < i 则 k→i。
        """
        for k in range(self.d):
            # 找到与 k 相连的节点对
            neighbors = np.where(self.G[k] == 1)[0]
            if len(neighbors) < 2:
                continue
            # 对节点对做组合
            for i, j in combinations(neighbors, 2):
                # 如果 i 和 j 不相连
                if self.G[i, j] == 0 and self.G[j, i] == 0:
                    # 若 k 不在 sepset[i][j] 中则是一个 collider
                    if k not in self.sepset[i][j]:
                        # 定向 i -> k <- j
                        self._set_oriented_edge(i, k)
                        self._set_oriented_edge(j, k)

    def _apply_meek_rules(self):
        """
        (b) Meek 规则推断更多方向，直到没有更新为止。
        主要利用 acyclic 假设和对已知方向的传递性。
        """
        updated = True
        while updated:
            updated = False
            # 这里简单实现 R1-R4 的思路:
            # R1 (若 p->q, p-q', q-q' 形成三元组, 则定 q->q')
            # ...
            # 为了简化，这里用一个粗略的循环检索所有未定向边，看能否通过“唯一定向可能”来处理。
            # 在实际中可根据原文的 Meek (R1)-(R4) 做更细化判断。

            # 仅举一个示例：对于每条未定向边 x - y，如果在 x 或 y 的某些邻居中能唯一推出定向，则更新。
            # 因为涉及到较多细节，这里只给出一个示例形式，以说明处理思路。
            for x in range(self.d):
                for y in range(self.d):
                    if x < y and self.G[x, y] == 1 and self.G[y, x] == 1:
                        # 检查是否因为时序而强制定向
                        if self.ordering[x] < self.ordering[y]:
                            self._set_oriented_edge(x, y)
                            updated = True
                        elif self.ordering[y] < self.ordering[x]:
                            self._set_oriented_edge(y, x)
                            updated = True
                        # 省略更多Meek规则的详细实现，可在此处补充

    def _set_oriented_edge(self, i, j):
        """
        将边定向为 i->j，即 G[i,j] = 1, G[j,i] = 0。
        """
        self.G[i, j] = 1
        self.G[j, i] = 0

