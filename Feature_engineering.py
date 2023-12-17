import pandas as pd
import numpy as np
import os
import datetime
import scipy.signal
import flirt.reader.empatica
import flirt.reader.garmin
import flirt


# 定义函数
# 频域分析 功率谱
def get_fft_power_spectrum(y: pd.Series):
    """
    y:穿戴设备获得的生理数据
    """
    N = len(y)
    # 频率真实幅值
    fft_values_ = np.abs(np.fft.fft(y))
    fft_values = fft_values_[0:N // 2] / (N / 2)
    # 功率谱
    ps_values = fft_values ** 2 / N

    # 重心频率
    S = []
    for i in range(N // 2):
        p = ps_values[i]
        f = fft_values[i]
        S.append(p * f)
    F1 = np.sum(S) / np.sum(ps_values)
    # 平均频率
    F2 = np.mean(ps_values)
    # 频率标准差
    S = []
    for i in range(N // 2):
        p = ps_values[i]
        f = fft_values[i]
        S.append(p * ((f - F1) ** 2))
    F3 = np.sqrt(np.sum(S) / np.sum(ps_values))
    # 均方根频率
    S = []
    for i in range(N // 2):
        p = ps_values[i]
        f = fft_values[i]
        S.append(p * (f ** 2))
    F4 = np.sqrt(np.sum(S) / np.sum(ps_values))

    return F1, F2, F3, F4


# 计算flirt特征数据
def get_flirt(bio: pd.DataFrame, df_dex: pd.DataFrame, bio_type: str, window_length: int = 5):
    """
    bio：生理数据
    df_dex：血糖数据
    bio_type：生理数据类型
    window_length:时间窗大小，默认为5分钟
    """
    bio_df = bio[(bio['datetime'] >= df_dex['datetime'].iloc[0] - datetime.timedelta(minutes=5)) &
                 (bio['datetime'] <= df_dex['datetime'].iloc[-1])]
    bio_df.set_index('datetime', inplace=True)

    if bio_type == 'acc':
        bio_fea = flirt.get_acc_features(bio_df,
                                         window_length=window_length * 60,
                                         window_step_size=300,
                                         data_frequency=32)
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=bio_fea.columns)])
        for i in df_dex.index:
            df = bio_fea[(bio_fea.index > df_dex.iloc[i, 0] - datetime.timedelta(minutes=5)) & (
                    bio_fea.index <= df_dex.iloc[i, 0])]
            if len(df) > 1:
                df_dex.loc[i, bio_fea.columns] = df.iloc[-1]
            elif len(df) == 1:
                df_dex.loc[i, bio_fea.columns] = df.iloc[0]
            else:
                continue
    elif bio_type == 'bvp':
        bio_fea = flirt.get_acc_features(bio_df,
                                         window_length=window_length * 60,
                                         window_step_size=300,
                                         data_frequency=64)
        fea = []
        for i in bio_fea.columns:
            if 'bvp' in i:
                fea.append(i)
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=fea)])
        for i in df_dex.index:
            df = bio_fea[(bio_fea.index > df_dex.iloc[i, 0] - datetime.timedelta(minutes=5)) & (
                    bio_fea.index <= df_dex.iloc[i, 0])]
            if len(df) > 1:
                df_dex.loc[i, fea] = df.iloc[-1, :len(fea)]
            elif len(df) == 1:
                df_dex.loc[i, fea] = df.iloc[0, :len(fea)]
            else:
                continue
    elif bio_type == 'eda':
        bio_fea = flirt.get_eda_features(bio_df[' eda'],
                                         window_length=window_length * 60,
                                         window_step_size=300,
                                         data_frequency=4)
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=bio_fea.columns)])
        for i in df_dex.index:
            df = bio_fea[(bio_fea.index > df_dex.iloc[i, 0] - datetime.timedelta(minutes=5)) & (
                    bio_fea.index <= df_dex.iloc[i, 0])]
            if len(df) > 1:
                df_dex.loc[i, bio_fea.columns] = df.iloc[-1]
            elif len(df) == 1:
                df_dex.loc[i, bio_fea.columns] = df.iloc[0]
            else:
                continue
    elif bio_type == 'ibi':
        bio_fea = flirt.get_hrv_features(bio_df[' ibi'],
                                         window_length=window_length * 60,
                                         window_step_size=300,
                                         domains=['td', 'fd', 'stat', 'nl'],
                                         threshold=0,
                                         clean_data=True)
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=bio_fea.columns)])
        for i in df_dex.index:
            df = bio_fea[(bio_fea.index > df_dex.iloc[i, 0] - datetime.timedelta(minutes=5)) & (
                    bio_fea.index <= df_dex.iloc[i, 0])]
            if len(df) > 1:
                df_dex.loc[i, bio_fea.columns] = df.iloc[-1]
            elif len(df) == 1:
                df_dex.loc[i, bio_fea.columns] = df.iloc[0]
            else:
                continue
    elif bio_type == 'temp':
        bio_fea = flirt.get_acc_features(bio_df,
                                         window_length=window_length * 60,
                                         window_step_size=300,
                                         data_frequency=4)
        fea = []
        for i in bio_fea.columns:
            if 'temp' in i:
                fea.append(i)
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=fea)])
        for i in df_dex.index:
            df = bio_fea[(bio_fea.index > df_dex.iloc[i, 0] - datetime.timedelta(minutes=5)) & (
                    bio_fea.index <= df_dex.iloc[i, 0])]
            if len(df) > 1:
                df_dex.loc[i, fea] = df.iloc[-1, :len(fea)]
            elif len(df) == 1:
                df_dex.loc[i, fea] = df.iloc[0, :len(fea)]
            else:
                continue
    else:
        raise ValueError("错误的数据类型")

    return df_dex


# 计算生理数据特征
def get_fea(df_dex: pd.DataFrame, df: pd.DataFrame, pos: int, col: str, window_size: int = 5):
    """
    df_dex:血糖数据
    df:筛选后的生理数据
    pos:当前时间血糖数据的索引
    col:对应生理数据中的数据列
    window_size:时间窗大小，默认为5分钟
    """
    if window_size == 5:
        fea = ['ms', 'xr', 'med', 'q1', 'q3', 'cov',
               'cf', 'pf', 'yudu', 'ff',
               'f1', 'f2', 'f3', 'f4',
               'peak_mean', 'peak_med', 'peak_std', 'peak_cov']
        for j in range(len(fea)):
            fea[j] = col + '_' + fea[j]
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=fea)])
        mean = df[col].mean()  # 平均数
        rms = np.sqrt(np.mean(df[col] ** 2))  # RMS均方根
        std = df[col].std()  # 标准差

        df_dex.loc[pos, fea[0]] = np.mean(df[col] ** 2)  # 均方值
        df_dex.loc[pos, fea[1]] = (np.mean(np.abs(df[col]))) ** 2  # 方根幅值
        df_dex.loc[pos, fea[2]] = df[col].median()  # 中位数
        df_dex.loc[pos, fea[3]] = df[col].quantile(0.25)  # 上四分位数
        df_dex.loc[pos, fea[4]] = df[col].quantile(0.75)  # 下四分位数
        df_dex.loc[pos, fea[5]] = std / mean  # 变异系数
        df_dex.loc[pos, fea[6]] = (np.max(np.abs(df[col]))) / rms  # 峰值因子
        df_dex.loc[pos, fea[7]] = (np.max(np.abs(df[col]))) / (np.mean(np.abs(df[col])))  # 脉冲因子
        df_dex.loc[pos, fea[8]] = (np.max(np.abs(df[col]))) / df_dex.loc[pos, fea[1]]  # 裕度因子
        df_dex.loc[pos, fea[9]] = rms / (np.mean(np.abs(df[col])))  # 波形因子
        f1, f2, f3, f4 = get_fft_power_spectrum(df[col])
        df_dex.loc[pos, fea[10]] = f1  # 重心频率
        df_dex.loc[pos, fea[11]] = f2  # 平均频率
        df_dex.loc[pos, fea[12]] = f3  # 频率标准差
        df_dex.loc[pos, fea[13]] = f4  # 均方根频率
        df = df.reset_index(drop=True)
        peaks, _ = scipy.signal.find_peaks(df[col])
        df_dex.loc[pos, fea[14]] = df[col][peaks].mean()  # 峰值平均
        df_dex.loc[pos, fea[15]] = df[col][peaks].median()  # 峰值中位数
        df_dex.loc[pos, fea[16]] = df[col][peaks].std()  # 峰值标准差
        df_dex.loc[pos, fea[17]] = df_dex.loc[pos, fea[16]] / df_dex.loc[pos, fea[14]]  # 峰值变异系数
    elif window_size in [15, 30, 60, 120]:
        fea = ['mean', 'med', 'q1', 'q3', 'std', 'cov',
               'peak_mean', 'peak_med', 'peak_std', 'peak_cov']
        for j in range(len(fea)):
            fea[j] = col + str(window_size) + '_' + fea[j]
        df_dex = pd.concat([df_dex, pd.DataFrame(columns=fea)])
        df_dex.loc[pos, fea[0]] = df[col].mean()  # 平均数
        df_dex.loc[pos, fea[1]] = df[col].median()  # 中位数
        df_dex.loc[pos, fea[2]] = df[col].quantile(0.25)  # 上四分位数
        df_dex.loc[pos, fea[3]] = df[col].quantile(0.75)  # 下四分位数
        df_dex.loc[pos, fea[4]] = df[col].std()  # 标准差
        df_dex.loc[pos, fea[5]] = df_dex.loc[pos, fea[4]] / df_dex.loc[pos, fea[0]]  # 变异系数
        df = df.reset_index(drop=True)
        peaks, _ = scipy.signal.find_peaks(df[col])
        df_dex.loc[pos, fea[6]] = df[col][peaks].mean()  # 峰值平均
        df_dex.loc[pos, fea[7]] = df[col][peaks].median()  # 峰值中位数
        df_dex.loc[pos, fea[8]] = df[col][peaks].std()  # 峰值标准差
        df_dex.loc[pos, fea[9]] = df_dex.loc[pos, fea[8]] / df_dex.loc[pos, fea[6]]  # 峰值变异系数
    else:
        raise ValueError("错误的数据类型")

    return df_dex


# 筛选时间窗口内的数据
def get_data(bio: pd.DataFrame, pos: int, window_size: int = 5):
    '''
    bio：生理数据
    pos：当前时间点在血糖数据中的索引
    window_size：时间窗大小，默认5分钟
    '''
    df = bio[(bio['datetime'] <= df_dex.iloc[pos, 0]) & (
            bio['datetime'] >= df_dex.iloc[pos, 0] - datetime.timedelta(minutes=window_size))]

    return df


# 读取、处理血糖数据，并计算昼夜节律特征
def read_dexdata(file, threshold: float = 7.0):
    '''
    threshold:区分血糖正常或偏高的阈值，默认为7.0
    '''
    df_dex = pd.read_csv(file)
    df_dex = df_dex.drop(df_dex.head(12).index)
    df_dex = df_dex.iloc[:, [1, 7]]
    df_dex = df_dex.reset_index(drop=True)
    df_dex.columns = ['datetime', 'gv']
    df_dex.datetime = pd.to_datetime(df_dex.datetime, infer_datetime_format=True)
    df_dex['gv'] = round(df_dex['gv'] / 18, 2)
    # 判断是否高血糖
    for i in df_dex.index:
        if df_dex.iloc[i, 1] >= threshold:
            df_dex.loc[i, 'high_gv'] = 1
        else:
            df_dex.loc[i, 'high_gv'] = 0
        df_dex.loc[i, 'min_mid'] = df_dex.iloc[i, 0].hour * 60 + df_dex.iloc[i, 0].minute
        df_dex.loc[i, 'hour_mid'] = df_dex.iloc[i, 0].hour
    return df_dex


# 读取、处理生理数据
def read_biodata(file):
    df_bio = pd.read_csv(file)
    df_bio.datetime = pd.to_datetime(df_bio.datetime, infer_datetime_format=True)

    return df_bio


# 定位数据文件文件夹位置
path1 = 'D:/研究生/人工智能竞赛/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0'
files1 = os.listdir(path1)

df_all = pd.DataFrame()
for Id in files1[0:16]:
    start = datetime.datetime.now()
    print(start)
    print(Id)
    # 定位数据位置
    path2 = path1 + '/' + Id
    files2 = os.listdir(path2)
    print(files2)
    # 读取血糖数据
    df_dex = read_dexdata(path2 + '/' + files2[2],threshold=7.0)
    print(datetime.datetime.now())
    # acc
    df_acc = read_biodata(path2 + '/' + files2[0])
    df_dex = get_flirt(df_acc, df_dex, 'acc', window_length=5)

    df_acc['acc'] = (df_acc[' acc_x'] ** 2 + df_acc[' acc_y'] ** 2 + df_acc[' acc_z'] ** 2) ** 0.5
    for i in df_dex.index:
        for j in [5, 15, 30, 60, 120]:
            df = get_data(df_acc, pos=i, window_size=j)
            if len(df) > 0:
                df_dex = get_fea(df_dex, df, pos=i, col=df.columns[-1], window_size=j)
    print(datetime.datetime.now())
    # bvp
    df_bvp = read_biodata(path2 + '/' + files2[1])
    df_dex = get_flirt(df_bvp, df_dex, 'bvp', window_length=5)

    for i in df_dex.index:
        for j in [5, 15, 30, 60, 120]:
            df = get_data(df_bvp, pos=i, window_size=j)
            if len(df) > 0:
                df_dex = get_fea(df_dex, df, pos=i, col=df.columns[-1], window_size=j)
    print(datetime.datetime.now())
    # eda
    df_eda = read_biodata(path2 + '/' + files2[3])
    df_dex = get_flirt(df_eda, df_dex, 'eda', window_length=5)

    for i in df_dex.index:
        for j in [5, 15, 30, 60, 120]:
            df = get_data(df_eda, pos=i, window_size=j)
            if len(df) > 0:
                df_dex = get_fea(df_dex, df, pos=i, col=df.columns[-1], window_size=j)
    print(datetime.datetime.now())
    # hr
    df_hr = read_biodata(path2 + '/' + files2[5])
    if Id == '001':
        df_hr['datetime'] = df_hr['datetime'].apply(lambda x: x - datetime.timedelta(days=150))
    for i in df_dex.index:
        for j in [5, 15, 30, 60, 120]:
            df = get_data(df_hr, pos=i, window_size=j)
            if len(df) > 0:
                df_dex = get_fea(df_dex, df, pos=i, col=df.columns[-1], window_size=j)
    print(datetime.datetime.now())
    # ibi
    df_ibi = read_biodata(path2 + '/' + files2[6])
    df_ibi[' ibi'] *= 1000
    df_dex = get_flirt(df_ibi, df_dex, 'ibi', window_length=5)

    for i in df_dex.index:
        for j in [5, 15, 30, 60, 120]:
            df = get_data(df_ibi, pos=i, window_size=j)
            if len(df) > 0:
                df_dex = get_fea(df_dex, df, pos=i, col=df.columns[-1], window_size=j)
    print(datetime.datetime.now())
    # temp
    df_temp = read_biodata(path2 + '/' + files2[7])
    df_dex = get_flirt(df_temp, df_dex, 'temp', window_length=5)

    for i in df_dex.index:
        for j in [5, 15, 30, 60, 120]:
            df = get_data(df_temp, pos=i, window_size=j)
            if len(df) > 0:
                df_dex = get_fea(df_dex, df, pos=i, col=df.columns[-1], window_size=j)

    # 保存（含缺失值）
    df_dex.to_csv('./final_test/withnan/{}.csv'.format(Id), encoding='utf-8_sig',
                  index=None)
    df_all = pd.concat([df_all, df_dex])
    end = datetime.datetime.now()
    print(end)
    print(end - start)

#处理缺失值
for i in df_all.columns[1:]:
    df_all[i]=df_all[i].astype('float64')
    df_all[i][np.isinf(df_all[i])]=np.nan
col_null=df_all.isnull().sum(axis=0)
drop_list=[]
for i in range(len(col_null)):
    if col_null[i]>13000:
        drop_list.append(i)
df_all=df_all.drop(columns=df_all.columns[drop_list])
df_all=df_all.dropna()
df_all.to_csv('./final_test/final_data.csv', encoding='utf-8_sig', index=None)
