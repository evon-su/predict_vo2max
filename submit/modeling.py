import json

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
    with open('ml_model.json', 'r', encoding='utf-8') as file:
        ml_params = json.load(file)

    coef = ml_params['coef_']
    intercept = ml_params['intercept_']

    def fn(user_info, speed_ls, hr_ls):
        nonlocal coef, intercept, hrr_range, speed_range
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

        # PREDICTING BY ML MODEL
        params = [ratio_med, user_info['gender'], user_info['hr_rest']]
        vo2max_pred = intercept
        for i, col in enumerate(params):
            vo2max_pred += coef[i] * col

        return vo2max_pred

    return fn
