import numpy as np
import pandas as pd
from model_v2_0 import ResampledPC
from model_oracle import oracle_pc
from model_naive import naive_ci_using_pc

ordering_10 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

results = []
for seed in range(1, 501):

    X_data = np.load(f"simulation/data1/{seed}/X_data_seed_{seed}.npy")
    W_adj = np.load(f"simulation/data1/{seed}/W_adj_seed_{seed}.npy")

    w_true = W_adj[5, 9]

    CI_oracle = oracle_pc(X_data, W_adj, exposure=6, outcome=10)
    print("finished seed", seed)

    # 如果方法返回None则对应的上下界记为NaN
    def unpack_interval(ci):
        if ci is None:
            return (np.nan, np.nan)
        return ci
    oracle_lower, oracle_upper = unpack_interval(CI_oracle)
    results.append({
        "seed": seed,
        "w_true": w_true,
        "oracle_lower": oracle_lower,
        "oracle_upper": oracle_upper,
    })

# 汇总至DataFrame
df = pd.DataFrame(results)
