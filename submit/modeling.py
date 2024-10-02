import preprocess


def model1(coef=None,
           hrr_range=(0.7, 0.92),
           speed_range=(3, 30)):
    """
    Simple Linear Regression by 'RATIO_median'
    :param coef: linear regression coefficient
    :param hrr_range: filter by hrr range
    :param speed_range: filter by speed range
    :return:
    """
    def fn(user_info, speed_ls, hr_ls):
        nonlocal coef, hrr_range, speed_range
        if coef is None:
            if user_info['gender'] == 1:
                coef = [ 1.87931922, 25.20964299]
            else:
                coef = [ 1.99660646, 19.09162092]
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
                    if 0 < hrr <= 1:
                        ratio_ls.append(speed / hrr)
        ratio_ls.sort()
        ratio_med = ratio_ls[len(ratio_ls) // 2]

        vo2max = 0
        for i in range(len(coef)):
            vo2max += coef[i] * ratio_med ** (len(coef) - 1 - i)

        return vo2max

    return fn

def model2(coef=None,
           hrr_range=(0.7, 0.92),
           speed_range=(3, 30)):
    """
    Linear Regression by [ RATIO_median, gender, hr_rest ]
    """
    if coef is None:
        coef = (34.5303885547102, 1.568010111027074, 5.006365018949635, -0.14893660037111436)

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
                if speed_range[0] < speed < speed_range[1]:
                    hrr = preprocess.hr2hrr(hr, user_info['hr_max'], user_info['hr_rest'])
                    ratio_ls.append(speed / hrr)
        ratio_ls.sort()
        ratio_med = ratio_ls[len(ratio_ls) // 2]

        # PREDICTING BY ML MODEL
        params = [1, ratio_med, user_info['gender'], user_info['hr_rest']]
        vo2max_pred = 0
        for i, col in enumerate(params):
            vo2max_pred += coef[i] * col

        return vo2max_pred

    return fn
