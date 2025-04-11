import numpy as np
import pandas as pd
from model_v2_0 import ResampledPC
from model_oracle import oracle_pc
from model_naive import naive_ci_using_pc
from data_generate import generate_dense_dag_data

ordering_10 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

for seed in range(5):

    X_data, W_adj = generate_dense_dag_data(n=500, seed=seed)

    print("-" * 20)
    print("Oracle method:")
    CI_oracle = oracle_pc(X_data, W_adj, exposure=6, outcome=10)
    print(CI_oracle)

    print("-" * 20)
    print("Naive method(alpha=0.05):")
    CI_naive5 = naive_ci_using_pc(X_data, alpha=0.05, ordering=ordering_10, exposure=6, outcome=10)
    print(CI_naive5)

    print("-" * 20)
    print("Naive method(alpha=0.01):")
    CI_naive1 = naive_ci_using_pc(X_data, alpha=0.01, ordering=ordering_10, exposure=6, outcome=10)
    print(CI_naive1)

    print("-" * 20)
    print("Resample method:")

    # 准备一个带重采样的PC对象
    rpc = ResampledPC(
        X=X_data,
        alpha=0.02,  # 实际未使用该参数
        ordering=ordering_10,
        M=200,  # 重采样次数
        c_star=0.05,
        gamma=0.05,  # 用于置信区间的置信水平
        nu = 0.025
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


