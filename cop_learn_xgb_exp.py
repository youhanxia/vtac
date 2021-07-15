import os
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection.rfe import RFECV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# import pickle


data_dir = 'data'
res_dir = 'result'
dt_fn = 'Column_types.csv'

target = 'Changed'

not_used = [
    'Year_of_application',
    'Date_of_Birth',
    'Residential_Postcode',
    'Residential_Country',
    'SA1_for_residential_Australian_address',
    'Country_of_Citizenship_Code',
    'Country_Born_Code',
    'Non_English_Language_Code_Spoken_At_Home',

]

cats = [
    'Category',
    'Gender',
    'Residential_State',
    'Citizenship_Status',
    'GET_English_language_assessment',
    'ATAR_Type',
    'Aboriginal_Australian_and_or_Torres_Strait_Islander_descent'
]

to_hide = [
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_1_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_1_Field_of_Study',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_2_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_2_Field_of_Study',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_3_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_3_Field_of_Study',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_1_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_1_Field_of_Study',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_2_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_2_Field_of_Study',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_3_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_3_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_1_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_1_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_2_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_2_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_3_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_3_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_4_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_4_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_5_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_5_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_6_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_6_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_7_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_7_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_8_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_8_Field_of_Study',
]


def data_prep(data):
    y = data[target]
    X = data.drop(target, axis=1)

    # one hot encoding
    X.drop(not_used, axis=1, inplace=True)

    for i in range(1, 9):
        cats.append(
            'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_' + str(i) + '_Institution_Code')
        cats.append(
            'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_' + str(i) + '_Field_of_Study')
        cats.append(
            'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_' + str(i) + '_Institution_Code')
        cats.append(
            'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_' + str(i) + '_Field_of_Study')
        # cats.append(
        #     'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_' + str(i) + '_Institution_Code')
        # cats.append(
        #     'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_' + str(i) + '_Field_of_Study')

    for fea in cats:
        dummies = pd.get_dummies(X[fea], prefix=fea)
        X = pd.concat([X, dummies], axis=1)
    X.drop(cats, axis=1, inplace=True)

    # hide last round
    X.drop(to_hide, axis=1, inplace=True)

    return X, y


def learn(X_train, X_test, y_train, y_test):
    # train
    model = xgb.XGBClassifier(max_depth=3, n_estimators=100, scale_pos_weight=8)
    model.fit(X_train, y_train)

    # test
    y_pred = model.predict(X_test)

    # eval
    return {
        'acc': accuracy_score(y_test.values, y_pred),
        'f1_score': f1_score(y_test.values, y_pred),
        'conf_mat': confusion_matrix(y_test.values, y_pred),
        'model': model
    }


def clf_xgb_exp(data_fn, test_fn=None):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    if test_fn is not None:
        test = pd.read_csv(os.path.join(data_dir, test_fn), dtype=dict(dt['dtype']))

        data = pd.concat([data, test], ignore_index=True)

        X, y = data_prep(data)

        res = learn(X[:43319], X[43319:], y[:43319], y[43319:])

        print('acc:', res['acc'])
        print('f1_sc', res['f1_score'])

        # fea_impt = res['model'].feature_importances_
        # print('feature_importance:')
        # print('\n'.join(X.keys()))
        # print('\n'.join(fea_impt.astype(str)))

        return res
    else:
        X, y = data_prep(data)

        # n repeats
        n = 30
        ress = []
        for i in range(n):
            # print('test', i, end='\r')
            print('test', i)
            # CV separate
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            res = learn(X_train, X_test, y_train, y_test)
            ress.append(res)

            print('acc:', res['acc'])
            print('f1_sc', res['f1_score'])
            # print(res['conf_mat'])

        # fea_impt = np.array([res['model'].feature_importances_ for res in ress])
        # fea_impt = np.mean(fea_impt, axis=0)
        # print('\n'.join(X.keys()))
        # print('\n'.join(fea_impt.astype(str)))

        acc = [res['acc'] for res in ress]
        f1_sc = [res['f1_score'] for res in ress]
        print('acc mean:', np.mean(acc))
        print('acc std:', np.std(acc))
        print('f1_sc mean:', np.mean(f1_sc))
        print('f1_sc std:', np.std(f1_sc))

        return ress


def rfe_exp(data_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    X, y = data_prep(data.fillna(0))

    estimator = xgb.XGBClassifier(max_depth=3, n_estimators=100, scale_pos_weight=8)
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)

    fea_sel = data.keys()[selector.get_support()]

    print(fea_sel.values.tolist())

if __name__ == '__main__':
    data_fn = 'MinimalFile_merged_with_CoP_sml.csv'
    test_fn = 'MinimalFile_2018_with_CoP_sml.csv'

    # learning experiment
    # clf_xgb_exp(data_fn, test_fn)
    # clf_xgb_exp(data_fn)

    rfe_exp(data_fn)

    pass
