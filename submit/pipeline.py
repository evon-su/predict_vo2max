import preprocess
import evaluate
import modeling
from predict import predictVO2max

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


def get_info(path=None, win=3, hrr_range=(0,1), speed_range=(0, 30), visual=True):
    if not path:
        path = "vo2max_data_for_interview/"

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
        workout['hrr'] = workout['HEART_RATE'].apply(lambda hr: preprocess.hr2hrr(hr,
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
        path = "vo2max_data_for_interview/raw_data/"
    return pd.read_csv(f"{path}{user}_{workout_number}.csv")

def fit_polyfit2(ratio_ls, vo2max_ls, degree=1):
    coef = np.polyfit(ratio_ls, vo2max_ls, degree)

    return coef


def pipeline(model,
             degree=1,
             speed_win=3,
             hrr_range=(0.7, 0.92),
             speed_range=(3, 30),
             train_ratio=0.8,
             random_state=0,
             plot=True):
    ml_cols = ['ratio_med', 'gender', 'hr_rest']

    # GET DATA
    info, actdf = get_info(win=speed_win, hrr_range=hrr_range, speed_range=speed_range, visual=plot)

    # SPLIT TRAIN & TEST BY USER
    user_ls = [user for user in info]
    train_user_ls, test_user_ls = preprocess.get_train_test_ls(len(info), train_ratio, rand=random_state)
    train_user_ls = [user_ls[i] for i in train_user_ls]
    test_user_ls = [user_ls[i] for i in test_user_ls]
    train = actdf.loc[actdf.index.isin(train_user_ls)]
    test = actdf.loc[actdf.index.isin(test_user_ls)]

    coef = fit_polyfit2(train.ratio_med.to_list(), train.real_vo2max.to_list(), degree=degree)
    print("REGRESSION COEF : ", coef)

    # Visualization
    x = np.linspace(5, 22, 100)
    y_train = 0
    for i in range(len(coef)):
        y_train += coef[i] * x**(len(coef)-i-1)
    if plot:
        corr = np.corrcoef(train.ratio_med.to_list(), train.real_vo2max.to_list())[0][1]
        print("CORR = ", corr)
        plt.scatter(train.ratio_med, train.real_vo2max, alpha=0.4, color=train.c)
        plt.plot(x, y_train, linestyle='--', alpha=0.4, c='gray')
        plt.title(f"(Training) corr={corr:.3f}")
        plt.xlabel('ratio of speed / hrr')
        plt.ylabel('vo2max')
        plt.show()

    # MACHINE LEARNING
    X_train = train[ml_cols].values
    y_train = train['real_vo2max'].values

    rf = LinearRegression()
    rf.fit(X_train, y_train)
    # joblib.dump(rf, 'ml_model.pkl')
    y_pred = rf.predict(X_train)
    rmse = evaluate.get_mse(y_train, y_pred) ** 0.5
    r2 = evaluate.get_rsq(y_train, y_pred)
    print(f"Training Result: rmse={rmse: .1f}, r2={r2:.2f}", )

    # PREDICT
    output = []
    for workout_number in test.workout_number:
        user = test[test.workout_number==workout_number].index[0]
        print("user, workout : ", user, workout_number, end=', ')
        workout = get_workout_df(user, workout_number)

        # PREDICTING
        vo2max_predict = predictVO2max(info[user], workout.SPEED, workout.HEART_RATE, model())

        vo2max_real = test[test['workout_number'] == workout_number]['real_vo2max'].iloc[0]
        output.append([user, workout_number, -1, vo2max_predict, vo2max_real])
        print("vo2max : ", round(vo2max_predict, 1), f"({vo2max_real})")

    # EVALUATE
    output = pd.DataFrame(output, columns=['user', 'workout_number', 'ratio_med', 'vo2max_pred', 'vo2max_real'])
    rmse_1 = evaluate.get_mse(output.vo2max_real, output.vo2max_pred) ** 0.5
    r2_1 = evaluate.get_rsq(output.vo2max_real, output.vo2max_pred)
    print(f"RMSE={rmse_1:.1f}, R2={r2_1:.3f}")

    output['diff'] = abs(output.vo2max_real - output.vo2max_pred)
    output['filename'] = output.apply(lambda row: f"{row.user}_{row.workout_number}", axis=1)
    output.to_csv("out.csv")

    print(f"ML: RMSE={rmse_1:.1f}, R2={r2_1:.3f}")

    if plot:
        corr = np.corrcoef(output.ratio_med, output.vo2max_pred)[0][1]
        x = np.linspace(5, 22, 100)
        y_train = 0
        for i in range(len(coef)):
            y_train += coef[i] * x ** (len(coef) - i - 1)
        plt.scatter(output.vo2max_real, output.vo2max_pred, alpha=0.4)
        plt.plot(output.vo2max_real, output.vo2max_real, alpha=0.3, c='gray', linestyle='--')
        plt.title(f"(Testing by Func) rmse={rmse_1:.2f}, r2={r2_1:.2f}")
        plt.xlabel('vo2max_real')
        plt.ylabel('vo2max_predict')
        plt.legend()
        plt.show()
    return rmse_1, r2_1


if __name__ == '__main__':
    pipeline(modeling.model2,
             degree=1,
             speed_win=3,
             hrr_range=(0.7, 0.92),
             speed_range=(3, 30),
             train_ratio=0.8,
             random_state=24,
             plot=False)

    # rmse_ls = []
    # r2_ls = []
    #
    # for i in range(100):
    #     rmse, r2 = pipeline(modeling.model2,
    #                         degree=1,
    #                         speed_win=3,
    #                         hrr_range=(0.7, 0.92),
    #                         speed_range=(3, 30),
    #                         train_ratio=0.8,
    #                         random_state=i,
    #                         plot=False)
    #     rmse_ls.append(rmse)
    #     r2_ls.append(r2)
    #
    # df = pd.DataFrame()
    # df['rmse'] = rmse_ls
    # df['r2'] = r2_ls
    # df.to_csv("statistic.csv")
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].hist(rmse_ls, alpha=0.3, bins=50)
    # ax[1].hist(r2_ls, alpha=0.3, bins=50)
    # ax[0].set_xlabel("rmse", fontsize=14)
    # ax[1].set_xlabel("r2", fontsize=14)
    # ax[0].set_ylabel("count", fontsize=14)
    # plt.show()
    #
