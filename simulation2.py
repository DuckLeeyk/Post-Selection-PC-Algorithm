import numpy as np
from model import PCAlgorithm


def generate_complex_data(n=1000, random_state=42):
    """
    生成一个包含 8 个变量的数据集，存在层级化的因果依赖关系:
      层 0: X1, X2, X3  (互相独立)
      层 1: X4, X5      (依赖上一层相关变量)
      层 2: X6, X7
      层 3: X8
    返回: X, shape = [n, 8]
    """
    np.random.seed(random_state)
    # 层 0: X1, X2, X3 ~ N(0, 1)
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)

    # 层 1: X4, X5
    # X4 = 0.8X1 + 0.5X2 + eps4
    # X5 = 1.2X2 + 0.7X3 + eps5
    eps4 = np.random.normal(0, 1, n)
    eps5 = np.random.normal(0, 1, n)
    X4 = 0.8 * X1 + 0.5 * X2 + eps4
    X5 = 1.2 * X2 + 0.7 * X3 + eps5

    # 层 2: X6, X7
    # X6 = 0.5X4 + 0.4X5 + eps6
    # X7 = 0.6X4 + eps7
    eps6 = np.random.normal(0, 1, n)
    eps7 = np.random.normal(0, 1, n)
    X6 = 0.5 * X4 + 0.4 * X5 + eps6
    X7 = 0.6 * X4 + eps7

    # 层 3: X8
    # X8 = 0.4X5 + 0.8X6 + 0.3X7 + eps8
    eps8 = np.random.normal(0, 1, n)
    X8 = 0.4 * X5 + 0.8 * X6 + 0.3 * X7 + eps8

    # 拼接为 (n, 8)
    X_data = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8])
    return X_data


# 生成更复杂的数据
X_data = generate_complex_data(n=2000, random_state=12345)

# 设定时序信息: (X1,X2,X3)在0层, (X4,X5)在1层, (X6,X7)在2层, (X8)在3层
ordering = [0, 0, 0, 1, 1, 2, 2, 3]

# 传入 PC 算法
pc = PCAlgorithm(X_data, alpha=0.01, ordering=ordering)
cpdag = pc.fit()

print("最终的 CPDAG 邻接矩阵:\n", cpdag)