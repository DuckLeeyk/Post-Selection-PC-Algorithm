import numpy as np
import numpy.linalg as la
from model_v2_0 import PCAlgorithm


def naive_ci_using_pc(X, alpha, ordering, exposure, outcome):
    """
    演示: 调用已有PCAlgorithm, 拿到一次CPDAG后,
          当作已知真图计算 naive CI.
    """
    # 1) 单次 PC
    pc = PCAlgorithm(X, alpha=alpha, ordering=ordering)
    cpdag = pc.fit()

    # 2) 判断cpdag是否有效(无双箭头, 无环等)
    if not is_valid_cpdag(cpdag):
        print("提示: 得到的CPDAG无效, 无法计算naive区间.")
        return None

    # 3) 回归
    e = exposure - 1
    o = outcome - 1
    parents_of_o = list(np.where(cpdag[:, o] == 1)[0])
    regressors = set(parents_of_o + [e])
    X_reg = X[:, list(regressors)]
    y_reg = X[:, o]
    e_index_in_reg = list(regressors).index(e)

    bhat, se = ols_effect(X_reg, y_reg, e_index_in_reg)
    ci_left = bhat - 1.96*se
    ci_right = bhat + 1.96*se
    return (ci_left, ci_right)

def is_valid_cpdag(G):
    # 与上面类似
    d = G.shape[0]
    for i in range(d):
        for j in range(i+1,d):
            if G[i,j]==1 and G[j,i]==1:
                return False
    if has_cycle(G):
        return False
    return True

def has_cycle(G):
    d = G.shape[0]
    visited = [0]*d
    def dfs(u):
        visited[u]=1
        for v in range(d):
            if G[u,v]==1:
                if visited[v]==1:
                    return True
                if visited[v]==0 and dfs(v):
                    return True
        visited[u]=2
        return False
    for node in range(d):
        if visited[node]==0:
            if dfs(node):
                return True
    return False

def ols_effect(X_reg, y_reg, exposure_idx):
    n, p = X_reg.shape
    inv_ = la.inv(X_reg.T @ X_reg)
    Beta_hat = inv_ @ (X_reg.T @ y_reg)
    y_pred = X_reg @ Beta_hat
    resid = y_reg - y_pred
    RSS = np.sum(resid**2)
    sigma2 = RSS/(n - p - 1)
    cov_Beta = sigma2*inv_
    bhat = Beta_hat[exposure_idx]
    se_bhat = np.sqrt(cov_Beta[exposure_idx, exposure_idx])
    return bhat, se_bhat
