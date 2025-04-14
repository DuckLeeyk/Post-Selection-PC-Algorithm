import numpy as np
import pandas as pd
from model_v3_2 import ResampledPC


ordering_10 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

results = []
for seed in range(1, 501):

    X_data = np.load(f"simulation/data1/{seed}/X_data_seed_{seed}.npy")
    W_adj = np.load(f"simulation/data1/{seed}/W_adj_seed_{seed}.npy")

    w_true = W_adj[5, 9]

    rpc = ResampledPC(
        X=X_data,
        alpha=100,  # 实际未使用该参数
        ordering=ordering_10,
        M=100,  # 重采样次数
        c_star=0.04,
        gamma=0.05,  # 用于置信区间的置信水平
        nu = 0.025
    )
    valid_cpdags = rpc.fit_resampling()
    valid_counts = len(valid_cpdags)
    print(f"finished seed{seed}, valid counts: ", valid_counts)
    CI_resample = rpc.aggregate_confidence_intervals(exposure=6, outcome=10)

    # 如果方法返回None则对应的上下界记为NaN
    def unpack_interval(ci):
        if ci is None:
            return (np.nan, np.nan)
        return ci

    resample_lower, resample_upper = unpack_interval(CI_resample)
    results.append({
        "seed": seed,
        "w_true": w_true,
        "resample_lower": resample_lower,
        "resample_upper": resample_upper,
        "valid_counts": valid_counts
    })

# 汇总至DataFrame
df = pd.DataFrame(results)
df.to_csv("simulation/data1/summary_resample004.csv", index=False)
