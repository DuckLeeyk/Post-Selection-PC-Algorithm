import numpy as np
import statsmodels.api as sm

def oracle_pc(sample_matrix, dag, exposure, outcome):
    """
    对于给定的样本矩阵和变量间的DAG邻接矩阵，
    以 dependent_idx 对应的变量作为因变量，
    以 target_idx 及其父节点对应的变量作为自变量进行线性回归，
    返回回归后 target_idx 对应变量的回归系数和标准误。

    参数：
        sample_matrix: numpy 数组，形状为 (n_samples, n_variables)
        dag: numpy 数组，形状为 (n_variables, n_variables)，
             若 dag[i, j]==1 表示变量 i->j(即 i 是 j 的父节点)
        dependent_idx: int，因变量的列索引
        target_idx: int，自变量列表中必须包括的目标变量列索引（另外还需把它的父节点加入进来）

    返回：
        coeff: 目标变量对应的回归系数（float）
        std_err: 目标变量对应的标准误（float）
    """

    dependent_idx = outcome - 1
    target_idx = exposure - 1

    # # 找到 target_idx 变量的父节点：即 dag[:, target_idx] 为 1 的那些变量
    # parent_idxs = list(np.where(dag[:, target_idx] != 0)[0])
    parent_idxs = list(np.where(dag[:, dependent_idx] != 0)[0])

    # parent_dependent_idxs = list(np.where(dag[:, dependent_idx] != 0)[0])
    # parent_target_idxs = list(np.where(dag[:, target_idx] != 0)[0])
    # parent_idxs = list(set(parent_dependent_idxs) & set(parent_target_idxs))

    # 构造自变量索引集合：目标变量及其父节点
    # 注意：保证目标变量出现在自变量的第一位，便于后续提取其回归系数和标准误
    # 如果 target_idx 不在父节点列表中，则置于第一位；如果已在父节点列表中，移到第一位。
    if target_idx in parent_idxs:
        parent_idxs.remove(target_idx)
    predictors = [target_idx] + parent_idxs

    # 提取变量对应的数据：X 为自变量矩阵，y 为因变量向量
    X = sample_matrix[:, predictors]
    y = sample_matrix[:, dependent_idx]

    # 拟合线性回归模型
    model = sm.OLS(y, X).fit()

    # 因为在 X 中第一个自变量是 target_idx 对应变量
    target_coefficient = model.params[0]
    target_std_err = model.bse[0]

    return (target_coefficient - 1.96*target_std_err, target_coefficient + 1.96*target_std_err)
