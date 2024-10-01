import joblib

import preprocess


def model1(coef=(2.25368142, 18.48279631), hrr_range=(0.7, 0.92), speed_range=(3, 30)):
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
    predictor = joblib.load('ml_model.pkl')

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
