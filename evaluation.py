# python
import pandas as pd
import numpy as np

# 从 CSV 文件中加载数据
df = pd.read_csv('simulation/data1/summary.csv')

# 方法及对应的区间下界和上界列名
methods = {
    "oracle": ("oracle_lower", "oracle_upper"),
    "naive5": ("naive5_lower", "naive5_upper"),
    "naive1": ("naive1_lower", "naive1_upper"),
    "resample": ("resample_lower", "resample_upper")
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
