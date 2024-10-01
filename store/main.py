import preprocess
import evaluate
import modeling

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import joblib


def predictVO2max(user_info, speed_ls, hr_ls, model=modeling.model2()):
    """
    :param user_info: contains "gender", "hr_max", "hr_rest" of the user
    :param model: the api of MODEL
    :return: predict vo2max
    """

    vo2max = model(user_info, speed_ls, hr_ls)

    return vo2max

def from_batch_data(model):
    import pandas as pd

    info, actdf = preprocess.get_info()

    output = []
    for user in info:
        print(" USER : ", user, f"gender={info[user]['gender']}, bmi={info[user]['bmi']:.1f}")
        for workout_number in actdf.loc[user]['workout_number']:
            print(" -- workout : ", workout_number, end=' : ')
            workout = preprocess.get_workout_df(user, workout_number)

            # MAIN
            vo2max_predict = predictVO2max(info[user], workout.SPEED, workout.HEART_RATE, model)

            vo2max_real = actdf[actdf['workout_number']==workout_number]['real_vo2max'].iloc[0]
            output.append([user, workout_number, vo2max_predict, vo2max_real])
            print("vo2max : ", round(vo2max_predict, 1), f"({vo2max_real})")

            # # PLOT
            # import matplotlib.pyplot as plt
            # plt.plot(range(len(workout.HEART_RATE)), workout.SPEED, alpha=0.3)
            # ax2 = plt.twinx()
            # ax2.plot(range(len(workout.HEART_RATE)), workout.HEART_RATE, c='r', alpha=0.3)
            # plt.show()

        print("_________________________________")
    output = pd.DataFrame(output, columns=['user', 'workout_number', 'vo2max_pred', 'vo2max_real'])
    rmse = evaluate.get_mse(output.vo2max_real, output.vo2max_pred) ** 0.5
    r2 = evaluate.get_rsq(output.vo2max_real, output.vo2max_pred)

    return {
        "rmse": rmse,
        "r2": r2
    }


def pipeline(model,
             degree=1,
             speed_win=3,
             hrr_range=(0.7, 0.92),
             speed_range=(3, 30),
             train_ratio=0.8,
             random_state=0,
             plot=True):
    ml_cols = ['age', 'gender', 'hr_max', 'hr_rest', 'bmi', 'ratio_med']
    ml_cols = ['ratio_med', 'gender', 'hr_rest']

    # GET DATA
    info, actdf = preprocess.get_info(win=speed_win, hrr_range=hrr_range, speed_range=speed_range)

    # SPLIT TRAIN & TEST BY USER
    user_ls = [user for user in info]
    train_user_ls, test_user_ls = preprocess.get_train_test_ls(len(info), train_ratio, rand=random_state)
    train_user_ls = [user_ls[i] for i in train_user_ls]
    test_user_ls = [user_ls[i] for i in test_user_ls]
    train = actdf.loc[actdf.index.isin(train_user_ls)]
    test = actdf.loc[actdf.index.isin(test_user_ls)]

    # train_ls, test_ls = preprocess.get_train_test_ls(len(actdf), train_ratio, rand=random_state)
    # train = actdf.iloc[train_ls]
    # test = actdf.iloc[test_ls]

    coef = modeling.fit_polyfit2(train.ratio_med.to_list(), train.real_vo2max.to_list(), degree=degree)
    print("COEF : ", coef)

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
    print(train.columns)

    X_train = train[ml_cols].values
    y_train = train['real_vo2max'].values

    rf = LinearRegression()
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'random_forest_model.pkl')
    y_pred = rf.predict(X_train)
    rmse = evaluate.get_mse(y_train, y_pred) ** 0.5
    r2 = evaluate.get_rsq(y_train, y_pred)
    print(f"Training Result: rmse={rmse: .1f}, r2={r2:.2f}", )

    # PREDICT
    output = []
    for workout_number in test.workout_number:
        user = test[test.workout_number==workout_number].index[0]
        print("user, workout : ", user, workout_number, end=', ')
        workout = preprocess.get_workout_df(user, workout_number)

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

    # PREDICT BY MACHINE LEARNING
    X_test = test[ml_cols].values
    y_test = test['real_vo2max'].values
    rf2 = joblib.load('random_forest_model.pkl')
    y_test_pred = rf2.predict(X_test)
    rmse_2 = evaluate.get_mse(y_test, y_test_pred) ** 0.5
    r2_2 = evaluate.get_rsq(y_test, y_test_pred)

    # df = test
    # df['real'] = y_test
    # df['predict'] = y_test_pred
    # df.sort_index(inplace=True)
    # df.to_csv("out.csv")
    output['diff'] = abs(output.vo2max_real - output.vo2max_pred)
    output['filename'] = output.apply(lambda row: f"{row.user}_{row.workout_number}", axis=1)
    output.to_csv("out.csv")

    print(f"ML: RMSE={rmse_2:.1f}, R2={r2_2:.3f}")

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

        plt.scatter(y_test, y_test_pred, alpha=0.4)
        plt.plot(y_test, y_test, alpha=0.3, c='gray', linestyle='--')
        plt.title(f"(Testing ML) rmse={rmse_2:.2f}, r2={r2_2:.2f}")
        plt.xlabel('vo2max_real')
        plt.ylabel('vo2max_predict')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    # output = from_batch_data(model=modeling.MODEL2)
    # print(f"RMSE={output['rmse']:.1f}, R2={output['r2']:.3f}")

    pipeline(modeling.model2,
             degree=1,
             speed_win=3,
             hrr_range=(0.7, 0.92),
             speed_range=(3, 30),
             train_ratio=0.8,
             random_state=0,
             plot=True)

