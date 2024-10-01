import modeling


def predictVO2max(user_info, speed_ls, hr_ls, model=modeling.model2()):
    """
    :param user_info: contains "gender", "hr_max", "hr_rest" of the user
    :param model: the api of MODEL
    :return: predict vo2max
    """

    vo2max = model(user_info, speed_ls, hr_ls)

    return vo2max


if __name__ == '__main__':
    import pandas as pd

    data_path = "vo2max_data_for_interview/"
    user_info = pd.read_csv(data_path + "user_info.csv", index_col='user').to_dict(orient='index')

    actdf = pd.read_csv(data_path + "vo2max_GT.csv", index_col='user')

    cnt, mse = 0, 0
    ls = []
    for user, row in actdf.iterrows():
        cnt += 1
        workout_number = row['workout_number']
        workout = pd.read_csv(data_path + f'raw_data/{user}_{workout_number}.csv')
        vo2max_real = actdf.loc[actdf['workout_number']==workout_number, 'real_vo2max'].values[0]

        # (MAIN) PREDICTING
        vo2max_pred = predictVO2max(user_info[user], workout['SPEED'], workout['HEART_RATE'], modeling.model2())

        mse += (vo2max_pred - vo2max_real)**2
        print(f"{user}_{workout_number}: predict: {vo2max_pred:.1f}, real: {vo2max_real}, diff={abs(vo2max_pred - vo2max_real):.1f}")
        ls.append(abs(vo2max_pred - vo2max_real))
    print("RMSE=", (mse/cnt)**0.5)

    import matplotlib.pyplot as plt
    plt.hist(ls, bins=30, alpha=0.3)
    plt.show()
