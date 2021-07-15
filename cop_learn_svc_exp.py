import os
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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


scale_magic = 28


def clf_svc_exp(data_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    data.fillna(0, inplace=True)

    y = data[target]
    X = data.drop(target, axis=1)

    # one hot encoding
    X.drop(not_used, axis=1, inplace=True)
    for fea in cats:
        dummies = pd.get_dummies(X[fea], prefix=fea)
        X = pd.concat([X, dummies], axis=1)
    X.drop(cats, axis=1, inplace=True)

    # hide last round
    X.drop(to_hide, axis=1, inplace=True)

    # print('\n'.join(X.keys()))
    # return

    # CV separate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    # model = SVC(kernel='linear', verbose=True)
    model = SVC(kernel='rbf', verbose=True)
    model.fit(X_train, y_train)

    # test
    y_pred = model.predict(X_test)

    # eval
    print('acc', accuracy_score(y_test.values, y_pred))
    print('f1_sc', f1_score(y_test.values, y_pred))
    print(confusion_matrix(y_test.values, y_pred))


if __name__ == '__main__':
    data_fn = 'MinimalFile_merged_with_CoP.csv'

    # learning experiment
    clf_svc_exp(data_fn)
    pass
