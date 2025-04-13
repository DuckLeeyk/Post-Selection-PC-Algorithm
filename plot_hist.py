import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


load = "simulation/data1/"
file_info = [
    (load + "summary_resample0.05.csv", "Valid CPDAG Counts (c*=0.05)"),
    (load + "summary_resample0.1.csv", "Valid CPDAG Counts (c*=0.1)"),
    (load + "summary_resample0.2.csv", "Valid CPDAG Counts (c*=0.2)"),
    (load + "summary_resample0.5.csv", "Valid CPDAG Counts (c*=0.5)")
]

# 预先收集所有 valid_counts 数据以统一横轴刻度
all_counts = []
for filename, _ in file_info:
    df = pd.read_csv(filename)
    counts = df["valid_counts"].dropna().values
    all_counts.extend(counts)

global_min = np.min(all_counts)
global_max = np.max(all_counts)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, (filename, title) in zip(axs.flatten(), file_info):
    df = pd.read_csv(filename)
    valid_counts = df["valid_counts"].values
    # 使用每个数据点的权重转换为频率（count/总数）
    weights = np.ones_like(valid_counts) / len(valid_counts)
    ax.hist(valid_counts, bins=20, edgecolor="k", weights=weights)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("valid counts", fontsize=12)
    ax.set_ylabel("frequency", fontsize=16)
    ax.set_xlim(global_min, global_max)
    # 关闭右侧和上方边框
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # 设置刻度字号
    ax.tick_params(axis="both", which="major", labelsize=16)

plt.tight_layout()
plt.show()