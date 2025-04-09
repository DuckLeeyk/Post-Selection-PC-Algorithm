"""
PC Algorithm, and Resampled PC Algorithm Implementation
"""


import numpy as np
from copy import deepcopy
from math import log, sqrt
from scipy.stats import norm
from itertools import combinations


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


class ResampledPC(PCAlgorithm):
    """
    在基础 PC 算法上，实现文档中提出的 Step 1 (重采样和筛选) 与 Step 2 (聚合)。
    """

    def __init__(self, X, alpha=0.05, ordering=None, M=100, c_star=0.01, gamma=0.05):
        """
        X: 数据矩阵
        alpha: 原始显著性水平 (如0.05)
        ordering: 时序信息
        M: 重采样运行 PC 算法的次数
        gamma: 用于最后(1-gamma)%置信区间
        """
        super().__init__(X, alpha, ordering)
        self.M = M
        self.c_star = c_star
        self.gamma = gamma

        # 存储多次运行的结果(可能含无效CPDAG)
        self.cpdags = []
        # 最终保留下来的有效CPDAG的索引集合
        self.valid_indices = []

    def fit_resampling(self):
        """
        实现 Step 1: 做 M 次(重采样) PC 算法，并筛除无效CPDAG。
        返回所有保留的有效 CPDAG 列表 self.cpdags。
        """
        self.cpdags = []  # 清空
        for m in range(self.M):
            # 每次都重新创建一个 PC 对象，但在其中对统计量施加重采样
            pc_obj = PCAlgorithmWithResampledTests(
                X=self.X, alpha=self.alpha, ordering=self.ordering, M=self.M, c_star=self.c_star
            )
            # 拟合(执行PC)
            G_m = pc_obj.fit()
            # 判断结果是否为“有效CPDAG”
            if self._is_valid_cpdag(G_m):
                self.cpdags.append(G_m)

        # 最后保留的索引集合
        self.valid_indices = list(range(len(self.cpdags)))
        return self.cpdags

    def aggregate_confidence_intervals(self, exposure, outcome):
        """
        实现 Step 2: 聚合多个有效 CPDAG 的置信区间 (union of intervals).

        参数:
        -------
        exposure: int, 暴露(自变量)的索引 i
        outcome: int,  结果(因变量)的索引 j

        假设我们要得到对真因果效应 beta_i,j(G) 的一个(1 - gamma)% 的区间，
        就需要对每个保留的 CPDAG, 解析它所代表的所有DAG, 再用某种方式对 beta 做 back-door 调整估计。

        这里仅示例一个非常简化的处理思路:
          - 假设 CPDAG ~ DAG (不做完整等价类列举)
          - 用线性回归(包括所有“可被视作父节点”的变量) 做回归，拿到该DAG下的beta估计
          - 再给出置信区间
          - 最终做并集

        **实际中可用更多工具(如 networkx)辨别CPDAG->所有DAG，以获取最小调整集或父集等。**
        """

        # 转换为0-index
        exposure = exposure - 1
        outcome = outcome - 1

        if len(self.cpdags) == 0:
            raise ValueError("尚未发现任何有效CPDAG，请先调用 fit_resampling().")

        # 准备一个列表，存放所有区间
        intervals = []

        # 对每个保留CPDAG:
        for idx in self.valid_indices:
            G_m = self.cpdags[idx]

            # 这里简单假设 CPDAG 已经是(唯一)有向无环图 (实际上可能不是唯一)
            # 找到 exposure 的“父节点”集合(含 exposure 自身可被看作调节后)
            # 仅做演示: 这里我们取 Pai(G_m, j) 并包含 i (exposure)
            # 真实中需分析 DAG 的结构
            parents_of_outcome = list(np.where(G_m[:, outcome] == 1)[0])
            # 若 不在时序未来层，则也可能视为父节点(具体视情况).
            # 这里只是个简化示例

            # 组装回归设计矩阵, X_i 及 parents_of_outcome
            # 确保不重复
            regressors = set(parents_of_outcome + [exposure])
            X_reg = self.X[:, list(regressors)]
            y_reg = self.X[:, outcome]
            # 用线性回归估计
            beta_hat, se_hat = self._estimate_linear_effect(X_reg, y_reg, exposure_idx=list(regressors).index(exposure))

            # 在正态假设下，(1-gamma)% 置信区间 ~ beta_hat +- z_{1-gamma/2} * se_hat
            z_val = norm.ppf(1 - self.gamma / 2)
            interval = (beta_hat - z_val * se_hat, beta_hat + z_val * se_hat)
            intervals.append(interval)

        # 取并集(最小左端, 最大右端)
        left_bounds = [itv[0] for itv in intervals]
        right_bounds = [itv[1] for itv in intervals]
        final_interval = (min(left_bounds), max(right_bounds))  # union

        return final_interval

    # ---------- 辅助函数 ----------

    def _is_valid_cpdag(self, G):
        """
        检查一个邻接矩阵 G 是否“有效CPDAG”，即无向边只能是 -，有向边无环且无双箭头。
        此处仅做一个简化判断：不含双向箭头 & 检查是否有向环(若有环则非有效).

        真正严谨的CPDAG判断，需要更多完整检查，这里只是示例。
        """
        d = G.shape[0]

        # 无双向箭头
        for i in range(d):
            for j in range(i + 1, d):
                if G[i, j] == 1 and G[j, i] == 1:
                    return False  # 存在双向
        # 无环
        # 可用 DFS 或拓扑排序检查是否有向环
        if self._has_cycle(G):
            return False

        return True

    def _has_cycle(self, G):
        """
        检查有向图 G 是否存在环(经典 DFS 拓扑排序检测)。
        """
        d = G.shape[0]
        visited = [0] * d  # 0=未访问, 1=访问中, 2=访问完

        def dfs(node):
            visited[node] = 1
            for nxt in range(d):
                if G[node, nxt] == 1:  # node->nxt
                    if visited[nxt] == 1:
                        return True  # 找到回环
                    if visited[nxt] == 0 and dfs(nxt):
                        return True
            visited[node] = 2
            return False

        for v in range(d):
            if visited[v] == 0:
                if dfs(v):
                    return True
        return False

    def _estimate_linear_effect(self, X_reg, y_reg, exposure_idx):
        """
        假设 y_reg = X_reg * Beta + eps, 其中 Beta中 exposure对应一列的系数为目标效应.
        返回该系数的估计值和标准误(简单 OLS).

        exposure_idx: 指示 X_reg中哪一列是 exposure.
        """
        # X_reg形状 [n, p], y_reg形状 [n, ]
        n, p = X_reg.shape
        # OLS估计 = (X'X)^-1 X'y
        # 估计协方差矩阵 Var(Beta_hat) ~ sigma^2 (X'X)^-1
        # sigma^2 = RSS/(n-p)

        # 增加截距
        ones = np.ones((n, 1))
        X_ = np.hstack([ones, X_reg])  # [n, p+1]
        # Beta_hat (p+1, )
        inv_ = np.linalg.inv(X_.T @ X_)
        Beta_hat = inv_ @ (X_.T @ y_reg)

        # 残差
        y_pred = X_ @ Beta_hat
        resid = y_reg - y_pred
        RSS = np.sum(resid ** 2)
        sigma2 = RSS / (n - p - 1)  # p+1含截距

        cov_Beta_hat = sigma2 * inv_

        # 曝光变量所在的系数是 Beta_hat[1 + exposure_idx]
        bhat = Beta_hat[1 + exposure_idx]
        se_bhat = np.sqrt(cov_Beta_hat[1 + exposure_idx, 1 + exposure_idx])

        return bhat, se_bhat


class PCAlgorithmWithResampledTests(PCAlgorithm):
    """
    重写父类中的 _conditional_indep_test，使用"一次高斯扰动"对 Z 统计量做重采样，
    并且结合文档(1)式中的“阈值收缩”。
    """

    def __init__(self, X, alpha=0.05, ordering=None, M=100, c_star=0.01):
        super().__init__(X, alpha, ordering)
        self.M = M
        self.c_star = c_star

    def _conditional_indep_test(self, i, j, S):
        """
        在计算出 Fisher's Z 后，随机扰动一次:
          Z(sampled) = Z(obs) + N(0,1)
        再比较  |Z(sampled)| > tau(M)*z_{alpha/2} 的阈值
        """
        # 先用与父类同样的方法得到observed Z值(不直接做p-value对比)
        if len(S) == 0:
            corr = np.corrcoef(self.X[:, i], self.X[:, j])[0, 1]
        else:
            XS = self.X[:, list(S)]
            Xi = self.X[:, i]
            Xj = self.X[:, j]

            ones = np.ones((self.n, 1))
            XS_ = np.hstack([ones, XS])
            inv_ = np.linalg.inv(XS_.T @ XS_)
            P_S = XS_ @ inv_ @ XS_.T

            ri = Xi - P_S @ Xi
            rj = Xj - P_S @ Xj
            corr = np.corrcoef(ri, rj)[0, 1]

        # 防止数值问题
        if abs(corr) > 0.999999:
            corr = 0.999999 * np.sign(corr)

        z_obs = 0.5 * log((1 + corr) / (1 - corr)) if abs(corr) < 1 else np.inf
        df = self.n - len(S) - 3
        # 典型公式Z = sqrt(df)*z_obs，但这里为了简便，我们直接把 sqrt(df) 融到 z_obs 上
        Z_obs = sqrt(df) * z_obs if df > 0 else z_obs

        # 重采样
        Z_sampled = Z_obs + np.random.normal(0, 1, 1)[0]

        # 计算shrink tau(M)
        L = (self.d*self.d-1) * (self.max_adjacent_edges()+1) / 2
        shrink_tau = self.c_star * (np.log(self.n) / self.M)**(1/L)

        # shrink tau(M)* z_{alpha/2}, 其中 z_{alpha/2} = norm.ppf(1-alpha/2)
        z_thresh = shrink_tau * norm.ppf(1 - self.alpha / 2)

        # 判断独立: 若 |Z_sampled| <= z_thresh 则认为独立
        return (abs(Z_sampled) <= z_thresh)

    def max_adjacent_edges(self):
        """
        计算有向无环图的邻接矩阵中邻接边数最多的节点的邻接边数

        返回:
        int: 邻接边数最多的节点的邻接边数（即出边数最多的节点的出边数）。
        """
        # 计算每个节点的出边数（即每一行的和）
        out_edge_counts = np.sum(self.G, axis=1)
        # 返回出边数的最大值
        return int(np.max(out_edge_counts))
