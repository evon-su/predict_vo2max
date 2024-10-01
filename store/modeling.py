import preprocess

import numpy as np
import joblib


def MODEL0(user_info, speed_ls, hr_ls):
    ratio_ls = []
    for speed, hr in zip(speed_ls, hr_ls):
        if preprocess.tmp_filter_speed(speed):
            hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])

            # Skip hrr <= 0
            if hrr > 0:
                ratio_ls.append(speed / hrr)
    ratio_med = ratio_ls[len(ratio_ls)//2]

    return 2.5 * ratio_med + 20

def MODEL1(user_info, speed_ls, hr_ls, coef):
    ratio_ls = []
    for speed, hr in zip(speed_ls, hr_ls):
        if preprocess.tmp_filter_speed(speed):
            hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])

            # Skip hrr <= 0
            if hrr > 0:
                ratio_ls.append(speed / hrr)
    ratio_ls.sort()
    ratio_med = ratio_ls[len(ratio_ls)//2]

    vo2max = 0
    for i in range(len(coef)):
        vo2max += coef[i] * ratio_med**(len(coef)-1-i)

    return vo2max

def model1(coef, hrr_range=(0.7, 0.92), speed_range=(3, 30)):
    def fn(user_info, speed_ls, hr_ls):
        nonlocal coef, hrr_range, speed_range
        ratio_ls = []
        for speed, hr in zip(speed_ls, hr_ls):
            if speed_range[0] < speed < speed_range[1]:
                hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])

                # SKIP HRR OUT OF RANGE
                if hrr_range[0] <= hrr <= hrr_range[1]:
                    ratio_ls.append(speed / hrr)
        # HANDLING INSUFFICIENT DATA
        if len(ratio_ls) < 30:
            ratio_ls = []
            for speed, hr in zip(speed_ls, hr_ls):
                if 1 < speed < 30:
                    hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])
                    ratio_ls.append(speed / hrr)
        ratio_ls.sort()
        ratio_med = ratio_ls[len(ratio_ls) // 2]

        vo2max = 0
        for i in range(len(coef)):
            vo2max += coef[i] * ratio_med ** (len(coef) - 1 - i)

        return vo2max

    return fn

def model2(hrr_range=(0.7, 0.92), speed_range=(3, 30)):
    predictor = joblib.load('random_forest_model.pkl')

    def fn(user_info, speed_ls, hr_ls):
        nonlocal predictor, hrr_range, speed_range
        ratio_ls = []
        for speed, hr in zip(speed_ls, hr_ls):
            if speed_range[0] < speed < speed_range[1]:
                hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])

                # SKIP HRR OUT OF RANGE
                if hrr_range[0] <= hrr <= hrr_range[1]:
                    ratio_ls.append(speed / hrr)
        # HANDLING INSUFFICIENT DATA
        if len(ratio_ls) < 30:
            ratio_ls = []
            for speed, hr in zip(speed_ls, hr_ls):
                if speed_range[0] < speed < speed_range[1]:
                    hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])
                    ratio_ls.append(speed / hrr)
        ratio_ls.sort()
        ratio_med = ratio_ls[len(ratio_ls) // 2]
        vo2max_pred = predictor.predict([[ratio_med, user_info['gender'], user_info['hr_rest']]])[0]

        return vo2max_pred

    return fn


def fit_polyfit(user_info, actdf, degree=1):
    import matplotlib.pyplot as plt
    ratio_med_ls = []
    vo2max_ls = []
    color_ls = []
    for user in actdf.index.unique():
        for workout_number in actdf.loc[[user]]['workout_number']:
            print(" -- workout_number : ", user, workout_number)
            workout = preprocess.get_workout_df(user, workout_number)
            rls = []
            for speed, hr in zip(workout.SPEED, workout.HEART_RATE):
                if preprocess.tmp_filter_speed(speed):
                    hrr = preprocess.hr2hrr(hr, user_info[user]['hr_max'], user_info[user]['hr_rest'])
                    if hrr > 0:
                        rls.append(speed / hrr)

            ratio_med = rls[len(rls) // 2]
            c = 'b' if user_info[user]['gender'] == 1 else 'r'
            ratio_med_ls.append(ratio_med)
            vo2max_ls.append(actdf[actdf.workout_number==workout_number]['real_vo2max'].iloc[0])
            color_ls.append(c)

    coef = np.polyfit(ratio_med_ls, vo2max_ls, degree)
    plt.scatter(ratio_med_ls, vo2max_ls, alpha=0.3, color=color_ls)
    x = np.linspace(5, 21, 100)
    y = 0
    for i in range(len(coef)):
        y += coef[i] * x**(len(coef)-1-i)
    # print("COEF : ", coef)
    # plt.plot(x, y, c='gray', alpha=0.4)
    # plt.grid()
    # plt.title(f"corr = {np.corrcoef(ratio_med_ls, vo2max_ls)[0][1]:.2f}")
    # plt.show()

    return coef

def fit_polyfit2(ratio_ls, vo2max_ls, degree=1):
    coef = np.polyfit(ratio_ls, vo2max_ls, degree)

    return coef


if __name__ == '__main__':

    info, actdf = preprocess.get_info()
    fit_polyfit(info, actdf)


