import pandas as pd


def get_info(path=None):
    if not path:
        path = "../vo2max_data_for_interview/"

    info = pd.read_csv(path+"user_info.csv", index_col='user')
    actdf = pd.read_csv(path+"vo2max_GT.csv", index_col='user')
    info['vo2max'] = None

    for user in info.index:
        avg = actdf.loc[user].real_vo2max.mean()
        info.loc[user, 'vo2max'] = avg
    info['bmi'] = info.apply(lambda row: row.weight / (row.height * row.height / 10000), axis=1)
    info = info.to_dict(orient='index')

    return info, actdf

def get_workout_df(user, workout_number, path=None):
    if not path:
        path = "../vo2max_data_for_interview/raw_data/"
    return pd.read_csv(f"{path}{user}_{workout_number}.csv")

def hr2hrr(hr, hr_max, hr_rest):
    return (hr - hr_rest) / (hr_max - hr_rest)

def tmp_filter_speed(velocity):
    if velocity > 30 or velocity == -1:
        return False
    return True


