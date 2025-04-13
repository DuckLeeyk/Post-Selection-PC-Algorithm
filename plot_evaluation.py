import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 从 CSV 文件加载数据
data = pd.read_csv("simulation/data1/evaluation.csv")

# 定义横坐标 c\*
c_values = [0.05, 0.10, 0.20, 0.50]

# 定义常量方法（值不随 c\* 改变），其中 oracle10 为原始oracle，oracle6 显示为 oracle(Pa_6(G))
constant_methods = ['oracle6', 'oracle10', 'naive0.05', 'naive0.01']

# 颜色映射
color_mapping = {
    'oracle10': '#999999',
    'oracle6': '#999999',
    'naive0.05': '#019e73',
    'naive0.01': '#0072b2',
    'resample': '#d55e00'
}

# 线型映射：oracle6使用虚线，其余均为实线
line_styles = {
    'oracle10': 'solid',
    'oracle6': 'dashed',
    'naive0.05': 'solid',
    'naive0.01': 'solid'
}

# 提取常量方法的指标
constant_coverage = {}
constant_length = {}
constant_score = {}
for m in constant_methods:
    row = data[data["methods"] == m].iloc[0]
    constant_coverage[m] = row["coverage"]
    constant_length[m] = row["length"]
    constant_score[m] = row["score"]

# 对 resample 方法，根据方法中 c\* 数值绘制
resample_mask = data["methods"].str.startswith("resample")
resample_data = data[resample_mask].copy()
resample_data["c_star"] = resample_data["methods"].str.replace("resample", "").astype(float)
resample_data.sort_values(by="c_star", inplace=True)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 图 1: Coverage Rate
ax = axs[0, 0]
for m in constant_methods:
    y = [constant_coverage[m]] * len(c_values)
    ax.plot(c_values, y, marker="o", linestyle=line_styles[m], color=color_mapping[m])
ax.plot(resample_data["c_star"], resample_data["coverage"],
        marker="o", linestyle="solid", color=color_mapping["resample"])
ax.set_title(r"Coverage Rate", fontsize=18)
ax.set_xlabel(r"$c^*$", fontsize=16)
ax.set_ylabel("Coverage", fontsize=16)
ax.set_xticks(c_values)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# 图 2: CI Length
ax = axs[0, 1]
for m in constant_methods:
    y = [constant_length[m]] * len(c_values)
    ax.plot(c_values, y, marker="o", linestyle=line_styles[m], color=color_mapping[m])
ax.plot(resample_data["c_star"], resample_data["length"],
        marker="o", linestyle="solid", color=color_mapping["resample"])
ax.set_title(r"CI Length", fontsize=18)
ax.set_xlabel(r"$c^*$", fontsize=16)
ax.set_ylabel("Length", fontsize=16)
ax.set_xticks(c_values)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# 图 3: Interval Score
ax = axs[1, 0]
for m in constant_methods:
    y = [constant_score[m]] * len(c_values)
    ax.plot(c_values, y, marker="o", linestyle=line_styles[m], color=color_mapping[m])
ax.plot(resample_data["c_star"], resample_data["score"],
        marker="o", linestyle="solid", color=color_mapping["resample"])
ax.set_title("Interval Score", fontsize=18)
ax.set_xlabel(r"$c^*$", fontsize=16)
ax.set_ylabel("Score", fontsize=16)
ax.set_xticks(c_values)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# 图 4: Valid CPDAG Counts % (仅 resample，counts 除以 100)
ax = axs[1, 1]
ax.plot(resample_data["c_star"], resample_data["valid"],
        marker="o", linestyle="solid", color=color_mapping["resample"])
ax.set_title("Valid CPDAG Counts %", fontsize=18)
ax.set_xlabel(r"$c^*$", fontsize=16)
ax.set_ylabel("valid counts %", fontsize=16)
ax.set_xticks(c_values)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# 全局图例放置在图外右侧
legend_elements = [
    Line2D([0], [0], color=color_mapping["oracle10"], marker="o", linestyle="solid", label=r'oracle ($Pa_{10}(G)$)'),
    Line2D([0], [0], color=color_mapping["oracle6"], marker="o", linestyle="dashed", label=r'oracle ($Pa_6(G)$)'),
    Line2D([0], [0], color=color_mapping["naive0.05"], marker="o", linestyle="solid", label="naive0.05"),
    Line2D([0], [0], color=color_mapping["naive0.01"], marker="o", linestyle="solid", label="naive0.01"),
    Line2D([0], [0], color=color_mapping["resample"], marker="o", linestyle="solid", label="resample")
]

fig.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1, 0.8),
           fontsize=12, frameon=False)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()