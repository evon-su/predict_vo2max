import preprocess
import evaluate
import modeling


def predictVO2max(user_info, speed_ls, hr_ls, model):
    """
    :param user_info: contains "hr_max" & "hr_rest" of the user
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
    rmse = evaluate.get_mse(output.vo2max_pred, output.vo2max_real) ** 0.5
    r2 = evaluate.get_rsq(output.vo2max_real, output.vo2max_pred)

    return {
        "rmse": rmse,
        "r2": r2
    }


if __name__ == '__main__':

    output = from_batch_data(model=modeling.MODEL2)
    print(f"RMSE={output['rmse']:.1f}, R2={output['r2']:.3f}")


