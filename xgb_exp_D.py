import math
import numpy as np
from itertools import product
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot as plt

from utils import read_in


feature_set = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
# feature_set = range(13)


def xgb_sfs_exp(data_fn):
    data = read_in(data_fn)
    header = np.array(data['header'])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size=0.2)

    # set params
    params = {'objective': 'reg:squarederror'}

    # SFS
    min_rmse = np.inf
    selected = []
    candidate = range(len(X_train[0]))
    while len(candidate):
        unchanged = True
        to_select = None
        for fea in candidate:
            if fea in selected:
                continue
            selected.append(fea)

            # screening
            X_train_sel = X_train[:, selected]
            X_test_sel = X_test[:, selected]

            # train
            train_dmat = xgb.DMatrix(X_train_sel, y_train)
            model = xgb.train(params, train_dmat)

            # test
            test_dmat = xgb.DMatrix(X_test_sel)
            y_pred = model.predict(test_dmat)

            # eval
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            if min_rmse > rmse:
                unchanged = False
                min_rmse = rmse
                to_select = fea
            selected.pop()
        if unchanged:
            break
        selected.append(to_select)

    print(data_fn[:-4], ', '.join(str(int(e)) for e in selected), 'rmse:', min_rmse)

    pass


def xgb_skl_exp(data_fn, on_train=False):
    data = read_in(data_fn)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size=0.2)

    for val in [(data_fn, on_train)]:
        # train
        model = xgb.XGBRegressor(max_depth=8, n_estimators=150, objective='reg:squarederror')
        model.fit(X_train, y_train)


        if on_train:
            # test
            y_pred = model.predict(X_train)

            # eval
            rmse = math.sqrt(mean_squared_error(y_train, y_pred))
        else:
            # test
            y_pred = model.predict(X_test)

            # eval
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))

        # print(model.feature_importances_)
        print(val, ':', rmse)



if __name__ == '__main__':
    # for i in range(10):
    #     print('test split', i)
    #     xgb_sfs_exp('Data2016.pkl')
    #     xgb_sfs_exp('Data2017.pkl')

    xgb_skl_exp('Data2016.pkl')
    xgb_skl_exp('Data2016.pkl', on_train=True)
    xgb_skl_exp('Data2017.pkl')
    xgb_skl_exp('Data2017.pkl', on_train=True)
