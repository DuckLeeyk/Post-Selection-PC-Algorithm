import numpy as np
from copy import deepcopy
from math import log, sqrt
from scipy.stats import norm
from itertools import combinations

class PCAlgorithm:
    """
    使用 PC 算法推断因果CPDAG的示例性实现。
    此版本使用两个邻接矩阵:
      - G_undir: 无向边 (i-j) 的矩阵 (对称)
      - G_dir:   有向边 (i->j) 的矩阵 (非对称)
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

        # 初始化分离集 sepset[i][j]
        self.sepset = [[set() for _ in range(self.d)] for __ in range(self.d)]

        # 初始图：完全无向 + 没有有向边
        self.G_undir = np.ones((self.d, self.d), dtype=int)
        np.fill_diagonal(self.G_undir, 0)  # 对角线为0
        self.G_dir = np.zeros((self.d, self.d), dtype=int)

    def fit(self):
        """
        执行 PC 算法, 返回 CPDAG (由 self.G_undir, self.G_dir 组合表示).
        """
        # 第一步: 邻接搜索 (去边)
        self._remove_edges()

        # 第二步: 方向定向
        # (a) 识别 v-structures
        self._orient_colliders()
        # (b) 应用 Meek 规则 (R1-R4) 反复更新，直至稳定
        self._apply_meek_rules()

        # 返回最终的 (G_undir, G_dir)
        # 用户可根据需要将二者保存并使用
        return self.G_undir, self.G_dir

    def _remove_edges(self):
        """
        通过逐渐增大条件集 S 的大小进行条件独立检验，
        删除无向边，并更新分离集 sepset。
        仅在 G_undir 中存在的无向边上进行检验。
        """
        s = 0
        cont = True

        while cont:
            cont = False
            # 当前（本轮开始时）的无向邻接列表
            current_adjacency = [
                set(np.where(self.G_undir[i] == 1)[0]) for i in range(self.d)
            ]

            for i in range(self.d):
                Adj_i = deepcopy(current_adjacency[i])

                for j in list(Adj_i):
                    if j <= i:
                        continue  # 只考虑 i<j 避免重复
                    # 确认目前仍是无向边才检验
                    if self.G_undir[i, j] == 0:
                        continue

                    # 决定是否要进行检验
                    O_bar_ij = self._get_future_nodes(i, j)
                    possible_parents_i = current_adjacency[i] - {j} - O_bar_ij

                    if len(possible_parents_i) >= s:
                        for S in combinations(possible_parents_i, s):
                            S = set(S)
                            independent = self._conditional_indep_test(i, j, S)
                            if independent:
                                # 从无向图中去掉 i-j
                                self._remove_undirected_edge(i, j)
                                # 记录分离集
                                self.sepset[i][j] = S
                                self.sepset[j][i] = S
                                cont = True
                                break
            s += 1

    def _get_future_nodes(self, i, j):
        """
        返回在时序上同时晚于 i 和 j 的所有节点集合 Ō_ij。
        """
        oi = self.ordering[i]
        oj = self.ordering[j]
        future_nodes = set(idx for idx, ok in enumerate(self.ordering)
                           if (ok > oi and ok > oj))
        return future_nodes

    def _conditional_indep_test(self, i, j, S):
        """
        Fisher's Z 检验评估 X_i 与 X_j 在给定 X_S 条件下是否独立。
        返回 True 表示独立 (p-value > alpha)，否则 False。
        """
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

        if abs(corr) < 1e-12:
            return True

        z = log((1 + corr) / (1 - corr))
        df = self.n - len(S) - 3
        Z = sqrt(df) * z
        p_value = 2 * (1 - norm.cdf(abs(Z)))
        return p_value > self.alpha

    def _orient_colliders(self):
        """
        (a) 识别 v-structure i->k<-j:
          对所有三元组 (i, k, j)，若 i-k, k-j 是无向边 且 i-j 无边，
          而且 k不在 sepset[i][j]中，则定向为 i->k<-j。
          并结合 ordering 来保证时序合理性。
        """
        for k in range(self.d):
            neighbors = np.where(self.G_undir[k] == 1)[0]
            if len(neighbors) < 2:
                continue
            for i, j in combinations(neighbors, 2):
                # i, j 不连通 -> i,j 没有无向边也没有有向边
                if (self.G_undir[i, j] == 0 and self.G_undir[j, i] == 0 and
                    self.G_dir[i, j] == 0 and self.G_dir[j, i] == 0):
                    # 若 k 不在 sepset[i][j] 中，说明是 collider
                    if k not in self.sepset[i][j]:
                        # 定向 i->k<-j
                        self._set_oriented_edge(i, k)
                        self._set_oriented_edge(j, k)

    def _apply_meek_rules(self):
        """
        (b) Meek 规则推断更多方向，直到没有更新为止。
        这里只示例一个简化版本: 若在无向边 i-j 上，根据 ordering
        可直接唯一确定方向，就定向。
        """
        updated = True
        while updated:
            updated = False
            # 遍历所有无向边尝试定向
            for x in range(self.d):
                for y in range(x+1, self.d):
                    if self.G_undir[x, y] == 1:  # 尚未定向的无向边
                        # 根据时序来决定方向
                        if self.ordering[x] < self.ordering[y]:
                            self._set_oriented_edge(x, y)
                            updated = True
                        elif self.ordering[y] < self.ordering[x]:
                            self._set_oriented_edge(y, x)
                            updated = True

    # ---------- 工具函数 ----------

    def _remove_undirected_edge(self, i, j):
        """
        从 G_undir 中去除无向边 i-j。
        """
        self.G_undir[i, j] = 0
        self.G_undir[j, i] = 0

    def _set_oriented_edge(self, i, j):
        """
        将无向边 i-j 定向为 i->j：
          - G_undir[i,j] 及 G_undir[j,i] 置 0
          - G_dir[i,j] 置 1, G_dir[j,i] 置 0
        """
        self.G_undir[i, j] = 0
        self.G_undir[j, i] = 0
        self.G_dir[i, j] = 1
        self.G_dir[j, i] = 0


class ResampledPC(PCAlgorithm):
    """
    在基础 PC 算法之上，实现文档中提出的 Step 1 (重采样和筛选) 与 Step 2 (聚合)。
    同样使用 (G_undir, G_dir) 两个矩阵来表示。
    """

    def __init__(self, X, alpha=0.05, ordering=None, M=100, c_star=0.01, gamma=0.05, nu=0.01):
        super().__init__(X, alpha, ordering)
        self.M = M
        self.c_star = c_star
        self.gamma = gamma
        self.nu = nu

        self.cpdags = []
        self.valid_indices = []

    def fit_resampling(self):
        """
        Step 1: 做 M 次重采样 PC 算法，并筛除无效 CPDAG。
        返回所有保留的有效 CPDAG 列表 self.cpdags。
        """
        self.cpdags = []
        for m in range(self.M):
            pc_obj = PCAlgorithmWithResampledTests(
                X=self.X,
                alpha=self.alpha,
                ordering=self.ordering,
                M=self.M,
                c_star=self.c_star,
                nu=self.nu
            )
            G_undir_m, G_dir_m = pc_obj.fit()
            if self._is_valid_cpdag(G_undir_m, G_dir_m):
                self.cpdags.append((G_undir_m, G_dir_m))

        self.valid_indices = list(range(len(self.cpdags)))
        return self.cpdags

    def aggregate_confidence_intervals(self, exposure, outcome):
        """
        Step 2: 聚合多个有效 CPDAG 的置信区间 (union of intervals).
        简化示例, 同时假设 CPDAG ~ DAG。
        """
        exposure -= 1
        outcome -= 1
        if len(self.cpdags) == 0:
            print("尚未发现任何有效CPDAG，请先调用 fit_resampling().")
            return None

        intervals = []
        for idx in self.valid_indices:
            G_undir_m, G_dir_m = self.cpdags[idx]

            # 简化假设: G_dir_m 真正是 DAG
            # 找出该 outcome 的父节点集
            parents_of_outcome = list(np.where(G_dir_m[:, outcome] == 1)[0])

            # 这里再将 exposure 也加入回归
            regressors = set(parents_of_outcome + [exposure])
            X_reg = self.X[:, list(regressors)]
            y_reg = self.X[:, outcome]

            beta_hat, se_hat = self._estimate_linear_effect(
                X_reg, y_reg, exposure_idx=list(regressors).index(exposure)
            )

            z_val = norm.ppf(1 - (self.gamma - self.nu) / 2)
            interval = (beta_hat - z_val * se_hat, beta_hat + z_val * se_hat)
            intervals.append(interval)

        lefts = [itv[0] for itv in intervals]
        rights = [itv[1] for itv in intervals]
        final_interval = (min(lefts), max(rights))
        return final_interval

    # ---------- 辅助函数 ----------

    def _is_valid_cpdag(self, G_undir, G_dir):
        """
        检查 (G_undir, G_dir) 是否是“有效CPDAG”。
        这里只做简化: 不应含双向有向边, 且有向部分无环。
        """
        d = G_undir.shape[0]
        # 不允许任何 i->j 和 j->i 同时存在
        for i in range(d):
            for j in range(d):
                if i != j and G_dir[i, j] == 1 and G_dir[j, i] == 1:
                    return False

        # 有向图部分无环
        if self._has_cycle(G_dir):
            return False

        return True

    def _has_cycle(self, G_dir):
        """
        检查 G_dir 是否有向环 (DFS 或拓扑排序)。
        """
        d = G_dir.shape[0]
        visited = [0]*d  # 0=未访问,1=访问中,2=访问完

        def dfs(node):
            visited[node] = 1
            for nxt in range(d):
                if G_dir[node, nxt] == 1:
                    if visited[nxt] == 1:
                        return True
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
        n, p = X_reg.shape
        ones = np.ones((n, 1))
        X_ = np.hstack([ones, X_reg])
        inv_ = np.linalg.inv(X_.T @ X_)
        Beta_hat = inv_ @ (X_.T @ y_reg)

        y_pred = X_ @ Beta_hat
        resid = y_reg - y_pred
        RSS = np.sum(resid ** 2)
        sigma2 = RSS / (n - p - 1)
        cov_Beta_hat = sigma2 * inv_

        bhat = Beta_hat[1 + exposure_idx]
        se_bhat = np.sqrt(cov_Beta_hat[1 + exposure_idx, 1 + exposure_idx])
        return bhat, se_bhat


class PCAlgorithmWithResampledTests(PCAlgorithm):
    """
    重写父类中的 _conditional_indep_test，使用“高斯扰动 + shrink_tau”做检验。
    同样使用 (G_undir, G_dir) 表示图结构。
    """

    def __init__(self, X, alpha=0.05, ordering=None, M=100, c_star=0.01, nu=0.01):
        super().__init__(X, alpha, ordering)
        self.M = M
        self.c_star = c_star
        self.nu = nu
        self.max_adjacent_edges = 7

    def _conditional_indep_test(self, i, j, S):
        """
        类似原文描述：在得到 Z_obs 后，Z_sampled = Z_obs + N(0,1)；对比 shrink_tau(M)*z_{阈值} 判断独立。
        这里只做示例。
        """
        # 先按父类方法获取 Z_obs
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

        if abs(corr) > 0.999999:
            corr = 0.999999 * np.sign(corr)

        z_obs = 0.0
        if abs(corr) < 1:
            z_obs = log((1 + corr) / (1 - corr))
        df = self.n - len(S) - 3
        Z_obs = sqrt(df)*z_obs if df > 0 else z_obs

        # 重采样
        Z_sampled = Z_obs + np.random.normal(0, 1)

        # shrink tau(M)
        # 此处仅示例给出一个 L
        L = (self.d*self.d-1) * (self.max_adjacent_edges+1) / 2
        shrink_tau = self.c_star * (np.log(self.n) / self.M)**(1 / L)

        z_thresh = shrink_tau * norm.ppf(1 - self.nu/2)
        return abs(Z_sampled) <= z_thresh
