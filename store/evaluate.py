from sklearn.metrics import r2_score


def get_mse(real_ls, predict_ls):
    if len(predict_ls) != len(real_ls):
        raise ValueError("data length not equal!")
    s = 0
    for v1, v2 in zip(predict_ls, real_ls):
        s += (v1 - v2)**2

    return s / len(predict_ls)

def get_rsq(real_ls, pred_ls):

    return r2_score(real_ls, pred_ls)

