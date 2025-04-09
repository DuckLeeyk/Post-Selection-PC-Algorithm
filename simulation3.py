"""
A 10-variable simulation example for resampled PC algorithm.
"""


import numpy as np
from model_v2_0 import ResampledPC


def generate_data_10vars(n=2000, random_state=12345):
    """
    生成10个变量的数据, ordering = (0,0,0,1,1,1,1,1,2,2)
    对应的简单因果依赖可能设计如下:

    层0: X1, X2, X3
         - 互相独立
    层1: X4 ~ X1, X2
         X5 ~ X2, X3
         X6 ~ X1
         X7 ~ X2
         X8 ~ X3
    层2: X9 ~ X7, X8
         X10 ~ X4, X5, X6

    其中, 每个变量 = 线性组合 + 独立N(0,1)的噪声
    """
    np.random.seed(random_state)
    # 层0
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)

    # 层1
    eps4 = np.random.normal(0, 1, n)
    eps5 = np.random.normal(0, 1, n)
    eps6 = np.random.normal(0, 1, n)
    eps7 = np.random.normal(0, 1, n)
    eps8 = np.random.normal(0, 1, n)

    X4 = 0.9 * X1 + 0.4 * X2 + eps4
    X5 = 0.5 * X2 + 0.8 * X3 + eps5
    X6 = 0.7 * X1 + eps6
    X7 = 1.1 * X2 + eps7
    X8 = 1.0 * X3 + eps8

    # 层2
    eps9 = np.random.normal(0, 1, n)
    eps10 = np.random.normal(0, 1, n)

    # X9 ~ X7, X8
    X9 = 0.6 * X7 + 0.7 * X8 + eps10

    # X10 ~ X4, X5, X6
    X10 = 0.4 * X4 + 0.3 * X5 + 0.5 * X6 + eps9

    # 汇总
    data = np.column_stack([X1, X2, X3, X4, X5, X6, X7, X8, X9, X10])
    return data

data_10 = generate_data_10vars(n=1000, random_state=20230406)

# ordering = (0,0,0,1,1,1,1,1,2,2)
ordering_10 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

# 准备一个带重采样的PC对象
rpc = ResampledPC(
    X=data_10,
    alpha=0.02,  # 可调
    ordering=ordering_10,
    M=200,  # 重采样次数
    c_star=0.01,
    gamma=0.05  # 用于置信区间的置信水平
)

# Step B: 多次重采样 + PC 发现
valid_cpdags = rpc.fit_resampling()
print(f"共有 {len(valid_cpdags)} 个有效CPDAG 保留下来.")

# Step C: 如果存在有效CPDAG, 则尝试对 (6->10) 聚合区间
final_ci = rpc.aggregate_confidence_intervals(exposure=6, outcome=10)
if final_ci is not None:
    print(f"(exposure=6 -> outcome=10) 的聚合区间: \n {final_ci}")
else:
    print("未能获取聚合区间，因为没有有效CPDAG。请考虑增大 alpha 或增大数据量等。")
