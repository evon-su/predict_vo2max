import random


def hr2hrr(hr, hr_max, hr_rest):
    return (hr - hr_rest) / (hr_max - hr_rest)

def get_train_test_ls(total_len, train_ratio, rand=0):
    random.seed(rand)
    train_num = int(total_len * train_ratio)

    train_ls = random.sample(range(total_len), train_num)
    test_ls = [i for i in range(total_len) if i not in train_ls]
    train_ls.sort()
    test_ls.sort()

    return train_ls, test_ls

def get_std(ls):
    summation = sum(ls)
    ls_len = len(ls)
    if summation > 0:
        avg = summation / ls_len
        variance = 0
        for v in ls:
            variance += (v - avg) * (v - avg)

        return (variance / ls_len) ** 0.5

    else:
        return 0



