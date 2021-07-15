import numpy as np
from matplotlib import pyplot as plt
# from sklearn.metrics import adjusted_mutual_info_score as mis
# from scipy.stats import pearsonr as r
# from sklearn.metrics import mean_squared_error as mse

from utils import read_in


def raw_data_exp(data_fn):
    data = read_in(data_fn)
    col_count = np.sum(data != '', axis=0)
    non_empty = col_count > 0
    data = data[:, non_empty]
    col_count = col_count[non_empty]
    print(col_count)
    plt.hist(col_count, bins=100)
    plt.show()


def clean_data_exp(data_fn):
    data = read_in(data_fn)
    X = data['X']
    y = data['y']
    header = data['header']

    feature_set = [0, 2, 3, 5, 7, 8, 9, 10, 11, 12]

    # # testing dummy regressor
    # y_bar = np.mean(y)
    # print('mean', y_bar)
    # print('rmse', np.sqrt(mse(y, [y_bar] * len(y))))

    # testing numerical features
    for i in feature_set:
        title = header[i]
        x = X[:, i]

        # print(title, mis(y, x))
        # print(title, r(y, x)[0])
        # plt.plot(x, y, '.')

        counter = {}
        for xj, yj in zip(x, y):
            if xj not in counter:
                counter[xj] = [yj]
            else:
                counter[xj].append(yj)

        counter = sorted(list(counter.items()))
        keys, vals = zip(*counter)
        # mean_vals = [np.mean(v) for v in vals]
        # plt.plot(keys, mean_vals, '.')
        plt.boxplot(vals, labels=keys)
        plt.title(title)
        plt.show()


    # # testing categorical features
    # for i in col_cat:
    #     title = header[i]
    #     x = X[:, i]
    #
    #     counter = {}
    #     for xj, yj in zip(x, y):
    #         if xj not in counter:
    #             counter[xj] = [yj]
    #         else:
    #             counter[xj].append(yj)
    #     plt.boxplot(counter.values(), labels=counter.keys())
    #
    #     plt.title(title)
    #     plt.show()


if __name__ == '__main__':
    # raw_data_exp('MasterFile_2016.csv')
    clean_data_exp('Data2017.pkl')
