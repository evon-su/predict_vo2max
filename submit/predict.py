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
    import sample_data

    user_info = sample_data.user_info
    speed_ls = sample_data.speed_ls
    hr_ls = sample_data.hr_ls

    vo2max_pred = predictVO2max(user_info, speed_ls, hr_ls, modeling.model2())

    print("PREDICT: ", vo2max_pred)