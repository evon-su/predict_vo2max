import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random


def get_info(path=None, win=3, hrr_range=(0,1), speed_range=(0, 30), visual=True):
    if not path:
        path = "../../vo2max_data_for_interview/"

    info = pd.read_csv(path+"user_info.csv", index_col='user')
    actdf = pd.read_csv(path+"vo2max_GT.csv", index_col='user')
    info['real_vo2max_avg'] = None
    actdf['ratio_med'] = None

    for user in info.index:
        avg = actdf.loc[user].real_vo2max.mean()
        info.loc[user, 'real_vo2max_avg'] = avg
    info['bmi'] = info.apply(lambda row: row.weight / (row.height * row.height / 10000), axis=1)
    info['hr_diff'] = info.apply(lambda row: row.hr_max - row.hr_rest, axis=1)

    # GET USER FEATURES
    cols = []
    for col in info.columns:
        if col != 'real_vo2max_avg':
            actdf[col] = None
            cols.append(col)

    for user, row in actdf.iterrows():
        workout_number = row['workout_number']
        for col in cols:
            actdf.loc[actdf.workout_number == workout_number, col] = info.loc[user, col]

        # COMPUTE RATIO
        workout = get_workout_df(user, workout_number)
        workout = workout[(workout['SPEED'] < speed_range[1]) & (workout['SPEED'] > speed_range[0])]  # FILTER SPEED
        if len(workout) > win * 2 and len(workout) > 60:
            win2 = win
        else:
            win2 = 3
        workout['SPEED'] = workout['SPEED'].rolling(window=win2).mean()
        workout = workout.iloc[win2-1:]
        workout['HEART_RATE'] = workout['HEART_RATE'].rolling(window=3).mean()
        workout['hrr'] = workout['HEART_RATE'].apply(lambda hr: hr2hrr(hr,
                                                                       info.loc[user, 'hr_max'],
                                                                       info.loc[user, 'hr_rest']))
        # FILTER HEART RATE
        workout = workout[(workout['hrr'] >= hrr_range[0]) & (workout['hrr'] <= hrr_range[1])]

        # GET RATIO
        workout['ratio'] = workout.apply(lambda row: row['SPEED'] / row['hrr'], axis=1)

        if len(workout) > 10:
            ratio_med = float(workout['ratio'].median())
        else:
            ratio_med = None
        actdf.loc[actdf['workout_number']==workout_number, 'ratio_med'] = ratio_med
        actdf.loc[actdf['workout_number']==workout_number, 'c'] = 'b' if info.loc[user]['gender'] == 1 else 'r'

    actdf.dropna(axis=0, how='any', subset=['ratio_med'], inplace=True)

    # VISUALIZATION
    if visual:
        corr = np.corrcoef(actdf['ratio_med'].to_list(), actdf['real_vo2max'].to_list())[0][1]
        coef = np.polyfit(actdf['ratio_med'].to_list(), actdf['real_vo2max'].to_list(), deg=1)
        print("COEF : ", coef)
        x = np.linspace(5, 22, 100)
        y = 0
        for i in range(len(coef)):
            y += coef[i] * x ** (len(coef) - i - 1)
        plt.scatter(actdf['ratio_med'], actdf['real_vo2max'], alpha=0.3, c=actdf['c'])
        plt.plot(x, y, alpha=0.4, c='gray', linestyle='--')
        plt.xlabel('RATIO_median')
        plt.ylabel('VO2max')
        plt.title(f"corr = {corr: .3f}, dataLen={len(actdf)}")
        plt.show()

    return info.to_dict(orient='index'), actdf


def get_workout_df(user, workout_number, path=None):
    if not path:
        path = "../../vo2max_data_for_interview/raw_data/"
    return pd.read_csv(f"{path}{user}_{workout_number}.csv")

def hr2hrr(hr, hr_max, hr_rest):
    return (hr - hr_rest) / (hr_max - hr_rest)

def tmp_filter_speed(velocity):
    if velocity > 30 or velocity < 0:
        return False
    return True

def get_train_test_ls(total_len, train_ratio, rand=0):
    random.seed(rand)
    train_num = int(total_len * train_ratio)

    train_ls = random.sample(range(total_len), train_num)
    test_ls = [i for i in range(total_len) if i not in train_ls]
    train_ls.sort()
    test_ls.sort()

    return train_ls, test_ls


