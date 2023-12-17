import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, precision_score,confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import spearman
import seaborn as sns

# 选择特征指标
def select_feature():
    # 读取CSV文件(总的特征文件)
    filename = 'data_feature1.csv'
    df = pd.read_csv(filename)

    # 计算第二列到倒数第二列与第一列的Spearman相关系数
    spearman_correlations = {}
    columns_to_compare = df.columns[2:-2]
    for col in columns_to_compare:
        spearman_correlation, _ = spearmanr(df['gv'], df[col])
        spearman_correlations[col] = abs(spearman_correlation)

    # 获取绝对值排名前200的列名
    top_200_columns = sorted(spearman_correlations, key=lambda x: spearman_correlations[x], reverse=True)[:200]

    # 进行二次筛选，保留两两相关性系数小于等于0.8的列,剔除较小相关性系数的列
    selected_columns = []
    dele_colums = []
    for i, col1 in enumerate(top_200_columns):
        if len(selected_columns) == 20:  # 保留前20列
            break
        if col1 in dele_colums:
            continue
        selected_columns.append(col1)
        for col2 in top_200_columns[i + 1:]:
            spearman_correlation, _ = spearmanr(df[col1], df[col2])
            if abs(spearman_correlation) > 0.8:
                dele_colums.append(col2)

    # 输出结果
    print("排名在前，但和现有筛选过的指标相关性大于0.8的指标：")
    print(dele_colums)
    print("经过二次筛选后的前20列名：")
    print(selected_columns)

    # 保存表格的时候，在表格最后两列加上类别和id
    selected_columns.append("high_gv")
    selected_columns.append("id")
    selected_columns.append("datetime")
    selected_columns_df = df[selected_columns]

    output_filename = 'selected_data.csv'
    selected_columns_df.to_csv(output_filename, index=False)

    # 读取刚刚保存的CSV文件
    filename = 'selected_data.csv'
    df = pd.read_csv(filename)[:-3]

    # 计算Spearman相关系数矩阵
    spearman_corr = df.corr(method='spearman')

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Spearman Correlation Heatmap")
    plt.show()
