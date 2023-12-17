import pandas as pd
import random
import os
seeds=[100,333,1000,9999]
seed=seeds[1]
# 设置输入和输出文件夹
input_folder = 'your_input_folder'
output_folder = 'split_folder_'+str(seed)


# 创建一个空的DataFrame来存储所有文件的10%数据
merged_data_10 = pd.DataFrame()
merged_data_90_all= pd.DataFrame()
# 循环处理每个CSV文件
for filename in os.listdir():
    if filename.endswith('.csv'):
        # 读取原始CSV文件
        input_path = os.path.join(filename)
        df = pd.read_csv(input_path)

        # 计算10%的数据量
        sample_size_10 = int(0.1 * len(df))
        sample_size_90 = len(df) - sample_size_10
        # 随机选择10%的数据
        sampled_data_10 = df.sample(n=sample_size_10, random_state=seed)  # 设置随机种子以确保可重复性

        # 将选中的10%数据追加到合并的DataFrame中
        merged_data_10 = pd.concat([merged_data_10, sampled_data_10], ignore_index=True)

        print(f'{filename} 的10%数据处理完成。')
        sampled_data_90 = df.drop(sampled_data_10.index)
        merged_data_90_all=pd.concat([merged_data_90_all, sampled_data_90], ignore_index=True)
        # 将剩余的90%数据保存为新的CSV文件
        output_filename_90 = f'train_{filename}'
        output_path_90 = os.path.join(output_folder, output_filename_90)
        sampled_data_90.to_csv(output_path_90, index=False)
# 将合并的DataFrame保存为新的CSV文件
output_filename_10 = 'test_data.csv'
output_path_10 = os.path.join(output_folder, output_filename_10)
merged_data_10.to_csv(output_path_10, index=False)
output_path_all = os.path.join(output_folder, 'trian_user_all.csv')
merged_data_90_all.to_csv(output_path_all,index=False)
print(f'合并并保存为 {output_filename_10} 完成。')

