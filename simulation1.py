import numpy as np
import pandas as pd
from model_v2_0 import ResampledPC
from model_oracle import oracle_pc
from model_naive import naive_ci_using_pc

ordering_10 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

results = []
for seed in range(500):

    X_data = np.load(f"simulation/data1/{seed}/X_data_seed_{seed}.npy")
    W_adj = np.load(f"simulation/data1/{seed}/W_adj_seed_{seed}.npy")

    w_true = W_adj[5, 9]

    CI_oracle = oracle_pc(X_data, W_adj, exposure=6, outcome=10)
    CI_naive5 = naive_ci_using_pc(X_data, alpha=0.05, ordering=ordering_10, exposure=6, outcome=10)
    CI_naive1 = naive_ci_using_pc(X_data, alpha=0.01, ordering=ordering_10, exposure=6, outcome=10)

    rpc = ResampledPC(
        X=X_data,
        alpha=0.02,  # 实际未使用该参数
        ordering=ordering_10,
        M=200,  # 重采样次数
        c_star=0.05,
        gamma=0.05,  # 用于置信区间的置信水平
        nu = 0.025
    )
    valid_cpdags = rpc.fit_resampling()
    CI_resample = rpc.aggregate_confidence_intervals(exposure=6, outcome=10)

    # 如果方法返回None则对应的上下界记为NaN
    def unpack_interval(ci):
        if ci is None:
            return (np.nan, np.nan)
        return ci
    oracle_lower, oracle_upper = unpack_interval(CI_oracle)
    naive5_lower, naive5_upper = unpack_interval(CI_naive5)
    naive1_lower, naive1_upper = unpack_interval(CI_naive1)
    resample_lower, resample_upper = unpack_interval(CI_resample)
    results.append({
        "seed": seed,
        "w_true": w_true,
        "oracle_lower": oracle_lower,
        "oracle_upper": oracle_upper,
        "naive5_lower": naive5_lower,
        "naive5_upper": naive5_upper,
        "naive1_lower": naive1_lower,
        "naive1_upper": naive1_upper,
        "resample_lower": resample_lower,
        "resample_upper": resample_upper
    })

# 汇总至DataFrame
df = pd.DataFrame(results)
print(df)
