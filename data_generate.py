import os
import numpy as np


def generate_dense_dag_data(n=1000, seed=42):
    """
    生成图文中描述的 10 节点 DAG (平均每节点7条边），
    并模拟 n 条观测数据。

    返回:
    -------
    X : np.ndarray, shape = [n, 10]
        生成的观测数据 (每行一个样本).
    W : np.ndarray, shape = [10, 10]
        最终的邻接权重矩阵，W[i,j] != 0 表示 i->j 的边权.
    """
    np.random.seed(seed)
    d = 10

    # 1) 构造一个稠密的DAG: 固定节点拓扑顺序为 0,1,...,9
    #    允许 i->j 仅当 i<j; 在所有可能的 i<j 中选出 35 条边.
    all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    # 一共 45 对 (i<j)；打散顺序
    np.random.shuffle(all_pairs)
    # 选前 35 对边
    chosen_edges = all_pairs[:35]

    # 邻接矩阵(0/1表示有无边), 下标: i->j
    A = np.zeros((d, d), dtype=int)
    for (i, j) in chosen_edges:
        A[i, j] = 1

    # 2) 为每条有向边 i->j 生成“原始”权重 wtilde[i,j]
    #    从 (-1, -0.5) ∪ (0.5, 1) 中均匀抽样
    #    注意区间(-0.5, 0.5)排除在外。
    w_tilde = np.zeros((d, d), dtype=float)
    for (i, j) in chosen_edges:
        # 先决定正负
        sign = 1 if np.random.rand() < 0.5 else -1
        # 再在 [0.5, 1) 区间采样绝对值
        mag = 0.5 + 0.5 * np.random.rand()  # uniform in [0.5,1.0)
        w_tilde[i, j] = sign * mag

    # 3) 按照公式 w_{ij} = wtilde[i,j] / sqrt( sum_{k in pa_j} wtilde[k,j]^2 + 1 )
    #    对 j 的全部父节点一次性进行缩放，使 “若X_k~N(0,1)则X_j也具有单位方差”
    W = np.zeros((d, d), dtype=float)
    for j in range(d):
        # 找到 j 的所有父节点
        parents_j = np.where(A[:, j] == 1)[0]
        if len(parents_j) == 0:
            continue
        # 计算 \sum_{k in pa_j} wtilde[k,j]^2
        sum_squares = np.sum(w_tilde[parents_j, j] ** 2)
        denom = np.sqrt(sum_squares + 1.0)
        # 对 j 的所有父节点做同样的缩放
        for k in parents_j:
            W[k, j] = w_tilde[k, j] / denom

    # 4) 根据最终权重 W, 按拓扑顺序从 X_0,...,X_9 依次生成 n 条样本
    X = np.zeros((n, d), dtype=float)
    for j in range(d):
        # 对每个节点 j, 其值 = sum_{k in pa_j} W[k,j]*X_k + e_j, e_j ~ N(0,1)
        # 在这里可以一次性生成 e_j
        eps_j = np.random.normal(loc=0.0, scale=1.0, size=n)
        parents_j = np.where(W[:, j] != 0)[0]
        if len(parents_j) == 0:
            X[:, j] = eps_j
        else:
            # X[:, j] = Σ_k W[k,j]*X[:,k] + eps_j
            X[:, j] = X[:, parents_j] @ W[parents_j, j] + eps_j

    return X, W


if __name__ == "__main__":

    for seed in range(1, 501):

        X_data, W_adj = generate_dense_dag_data(n=500, seed=seed)

        # 保存生成的数据和邻接矩阵
        os.makedirs(os.path.join("simulation/data1", str(seed)), exist_ok=True)
        np.save(f"simulation/data1/{seed}/X_data_seed_{seed}.npy", X_data)
        np.save(f"simulation/data1/{seed}/W_adj_seed_{seed}.npy", W_adj)
