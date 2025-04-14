import numpy as np
import numpy.linalg as la
from model_v3_2 import PCAlgorithm


def naive_ci_using_pc(X, alpha, ordering, exposure, outcome):
    """
    演示: 调用已有PCAlgorithm, 拿到一次CPDAG后,
          当作已知真图计算 naive CI.
    """
    # 1) 单次 PC
    pc = PCAlgorithm(X, alpha=alpha, ordering=ordering)
    G_undir, G_dir = pc.fit()

    # 2) 判断cpdag是否有效(无双箭头, 无环等)
    if not is_valid_cpdag(G_undir, G_dir):
        print("提示: 得到的CPDAG无效, 无法计算naive区间.")
        return None

    else:
        all_dags = enumerate_all_dags_from_cpdag(G_undir, G_dir)
        intervals = []
        # 2) 对每个 DAG，用 back-door adjustment 估计 (exposure -> outcome)
        for DAG in all_dags:
            # 3) 回归
            e = exposure - 1
            o = outcome - 1
            parents_of_o = list(np.where(DAG[:, o] == 1)[0])
            regressors = set(parents_of_o + [e])
            X_reg = X[:, list(regressors)]
            y_reg = X[:, o]
            e_index_in_reg = list(regressors).index(e)

            bhat, se = ols_effect(X_reg, y_reg, e_index_in_reg)
            ci_left = bhat - 1.96*se
            ci_right = bhat + 1.96*se

            intervals.append((ci_left, ci_right))

            # 3) 最终聚合 => 取所有区间的并集
            if len(intervals) == 0:
                return None
            else:
                lefts = [itv[0] for itv in intervals]
                rights = [itv[1] for itv in intervals]
                return (min(lefts), max(rights))


def is_valid_cpdag(G_undir, G_dir):

    d = G_undir.shape[0]
    # 不允许任何 i->j 和 j->i 同时存在
    for i in range(d):
        for j in range(d):
            if i != j and G_dir[i, j] == 1 and G_dir[j, i] == 1:
                return False

    # 有向图部分无环
    if has_cycle(G_dir):
        return False

    return True


def has_cycle(G_dir):
    """
    检查 G_dir 是否存在有向环(DFS 或拓扑排序)，返回 True/False。
    """
    d = G_dir.shape[0]
    visited = [0] * d  # 0=未访问,1=访问中,2=访问完

    def dfs(u):
        visited[u] = 1
        for v in range(d):
            if G_dir[u, v] == 1:
                if visited[v] == 1:
                    return True  # 找到环
                if visited[v] == 0 and dfs(v):
                    return True
        visited[u] = 2
        return False

    for node in range(d):
        if visited[node] == 0:
            if dfs(node):
                return True
    return False


def enumerate_all_dags_from_cpdag(G_undir, G_dir):

    d = G_undir.shape[0]

    # 找到还存在的无向边列表
    edges_undir = []
    for i in range(d):
        for j in range(i + 1, d):
            if G_undir[i, j] == 1:  # 说明还有无向边 i-j
                edges_undir.append((i, j))

    # 如果已经没有无向边了 => (G_undir, G_dir) 就是一个"确定”的有向图(可能含部分孤立顶点)
    # 复制 G_dir 作为一个完整 DAG 返回
    if len(edges_undir) == 0:
        return [G_dir.copy()]

    # 否则，取出一条无向边 i-j 来尝试两种定向
    i0, j0 = edges_undir[0]

    # 为了后续递归，先做一次拷贝
    G_undir_copy = G_undir.copy()
    G_dir_copy = G_dir.copy()

    # 从 G_undir 中去掉 i0-j0
    G_undir_copy[i0, j0] = 0
    G_undir_copy[j0, i0] = 0

    all_dags = []

    # 尝试  i0->j0
    G_dir_try = G_dir_copy.copy()
    G_dir_try[i0, j0] = 1
    G_dir_try[j0, i0] = 0
    # 若无环，则继续递归
    if not has_cycle(G_dir_try):
        all_dags.extend(enumerate_all_dags_from_cpdag(G_undir_copy, G_dir_try))

    # 再尝试 j0->i0
    G_dir_try = G_dir_copy.copy()
    G_dir_try[j0, i0] = 1
    G_dir_try[i0, j0] = 0
    if not has_cycle(G_dir_try):
        all_dags.extend(enumerate_all_dags_from_cpdag(G_undir_copy, G_dir_try))

    return all_dags


def ols_effect(X_reg, y_reg, exposure_idx):
    n, p = X_reg.shape
    inv_ = la.inv(X_reg.T @ X_reg)
    Beta_hat = inv_ @ (X_reg.T @ y_reg)
    y_pred = X_reg @ Beta_hat
    resid = y_reg - y_pred
    RSS = np.sum(resid**2)
    sigma2 = RSS/(n - p)
    cov_Beta = sigma2*inv_
    bhat = Beta_hat[exposure_idx]
    se_bhat = np.sqrt(cov_Beta[exposure_idx, exposure_idx])
    return bhat, se_bhat
