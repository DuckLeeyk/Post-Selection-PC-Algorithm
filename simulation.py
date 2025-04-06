import numpy as np
from model import PCAlgorithm

if __name__ == "__main__":
    # 这里给出一个简单的随机数据示例
    np.random.seed(42)
    n_samples = 200
    X1 = np.random.normal(0, 1, n_samples)
    X2 = 2 * X1 + np.random.normal(0, 1, n_samples)
    X3 = 0.5 * X1 + 0.3 * X2 + np.random.normal(0, 1, n_samples)
    X4 = np.random.normal(0, 1, n_samples)
    X_data = np.column_stack([X1, X2, X3, X4])

    # 假设时序为：X1, X2 同一时间点(层级0)，X3 在层级1，X4 在层级2
    ordering = [0, 0, 1, 2]

    pc = PCAlgorithm(X_data, alpha=0.01, ordering=ordering)
    cpdag = pc.fit()

    print("最终的 CPDAG 邻接矩阵为 (1表示有方向 i->j 或 j->i，需看行列关系)：\n", cpdag)
