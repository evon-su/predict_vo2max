import preprocess
import numpy as np


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

def MODEL1(user_info, speed_ls, hr_ls):
    ratio_ls = []
    for speed, hr in zip(speed_ls, hr_ls):
        if preprocess.tmp_filter_speed(speed):
            hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])

            # Skip hrr <= 0
            if hrr > 0:
                ratio_ls.append(speed / hrr)
    ratio_med = ratio_ls[len(ratio_ls)//2]

    return 1.5850014408338886 * ratio_med + 27.915252481456296

def MODEL2(user_info, speed_ls, hr_ls):
    ratio_ls = []
    for speed, hr in zip(speed_ls, hr_ls):
        if preprocess.tmp_filter_speed(speed):
            hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])

            # Skip hrr <= 0
            if hrr > 0:
                ratio_ls.append(speed / hrr)
    ratio_med = ratio_ls[len(ratio_ls)//2]

    return 0.10564508 * ratio_med**2 - 1.22491908 * ratio_med + 45.16544165

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
    print("COEF : ", coef)
    plt.plot(x, y, c='gray', alpha=0.4)
    plt.grid()
    plt.title(f"corr = {np.corrcoef(ratio_med_ls, vo2max_ls)[0][1]:.2f}")
    plt.show()


if __name__ == '__main__':

    info, actdf = preprocess.get_info()
    fit_polyfit(info, actdf)


