import numpy as np
import pandas as pd
from model_naive import naive_ci_using_pc

ordering_10 = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

results = []
for seed in range(1, 501):

    X_data = np.load(f"simulation/data1/{seed}/X_data_seed_{seed}.npy")
    W_adj = np.load(f"simulation/data1/{seed}/W_adj_seed_{seed}.npy")

    w_true = W_adj[5, 9]

    CI_naive = naive_ci_using_pc(X_data, alpha=0.01, ordering=ordering_10, exposure=6, outcome=10)
    print("finished seed", seed)

    # 如果方法返回None则对应的上下界记为NaN
    def unpack_interval(ci):
        if ci is None:
            return (np.nan, np.nan)
        return ci
    naive_lower, naive_upper = unpack_interval(CI_naive)
    results.append({
        "seed": seed,
        "w_true": w_true,
        "naive_lower": naive_lower,
        "naive_upper": naive_upper,
    })

# 汇总至DataFrame
df = pd.DataFrame(results)


# 方法及对应的区间下界和上界列名
methods = {
    "naive": ("naive_lower", "naive_upper"),
}

results = {}
alpha = 0.05

# 针对每种方法，计算覆盖率、区间长度以及 Interval Score
for method, (low_col, up_col) in methods.items():
    # 筛选有效的区间记录
    valid = ~df[low_col].isna() & ~df[up_col].isna()

    # 覆盖率：真实值 w_true 落在 [L, U] 内
    coverage = df.loc[valid].apply(
        lambda row: row[low_col] <= row["w_true"] <= row[up_col], axis=1
    )
    avg_coverage = coverage.mean()

    # 区间长度
    interval_length = df.loc[valid, up_col] - df.loc[valid, low_col]
    avg_length = interval_length.mean()


    # Interval Score 指标
    # 公式: (U - L) + (2/alpha) * (L - y) * I{y < L} + (2/alpha) * (y - U) * I{y > U}
    def interval_score(row):
        L = row[low_col]
        U = row[up_col]
        y = row["w_true"]
        penalty_lower = (L - y) if y < L else 0
        penalty_upper = (y - U) if y > U else 0
        return (U - L) + (2 / alpha) * (penalty_lower + penalty_upper)


    interval_scores = df.loc[valid].apply(interval_score, axis=1)
    avg_score = interval_scores.mean()

    results[method] = {"avg_coverage": avg_coverage, "avg_length": avg_length, "avg_interval_score": avg_score}

# 输出结果
summary_df = pd.DataFrame(results).T
print(summary_df)
