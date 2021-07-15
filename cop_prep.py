import os
import pandas as pd
from externals.stucco import ContrastSetLearner
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

data_dir = 'data'
res_dir = 'result'
dt_fn = 'Column_types.csv'

target = 'Changed'
metric = any
uni_code = 'I32'
first_prefix = 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_'
second_prefix = 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_'
last_prefix = 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_'
suffix = '_Institution_Code'
high_range = ['1', '2', '3']


def cop_prep(data_fn, out_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
    # print('\n'.join(data.keys()))
    high_pref_data = pd.DataFrame()
    high_pref_data['chose_first'] = data.apply(lambda x: metric([x[first_prefix + i + suffix] == uni_code for i in high_range]), axis=1)
    high_pref_data['chose_second'] = data.apply(lambda x: metric([x[second_prefix + i + suffix] == uni_code for i in high_range]), axis=1)
    high_pref_data['chose_last'] = data.apply(lambda x: metric([x[last_prefix + i + suffix] == uni_code for i in high_range]), axis=1)
    changed = high_pref_data.apply(
        lambda x: (x['chose_first'] or x['chose_second']) and not x['chose_last'], axis=1
    )
    # print(sum(changed))
    data[target] = changed

    # # write the modified data to csv
    # data.to_csv(os.path.join(data_dir, out_fn), index=False)

    return data


def cop_inst_prep(data_fn, out_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    # get indices
    high_pref_data = pd.DataFrame()
    high_pref_data['chose_first'] = data.apply(lambda x: metric([x[first_prefix + i + suffix] == uni_code for i in high_range]), axis=1)
    high_pref_data['chose_second'] = data.apply(lambda x: metric([x[second_prefix + i + suffix] == uni_code for i in high_range]), axis=1)
    high_pref_data['not_chose_at_all'] = high_pref_data.apply(lambda x: not x['chose_first'] and not x['chose_second'], axis=1)

    # perform drop
    print('original size', data.shape[0])
    print('rows to drop', sum(high_pref_data['not_chose_at_all']))
    data.drop(high_pref_data[high_pref_data['not_chose_at_all']].index, inplace=True)
    print('final size', data.shape[0])


    # # write the modified data to csv
    # data.to_csv(os.path.join(data_dir, out_fn), index=False)

    return data


def cop_dt_prep(data_fn, out_fn):
    pd.set_option('display.max_rows', 200)

    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
    print(set(data.keys()) - set(dt.index.tolist()))
    print(data.dtypes)
    pass


if __name__ == '__main__':
    # # prepare columns
    data_fn = 'MinimalFile_merged.csv'
    out_fn = data_fn[:-4] + '_with_CoP.csv'
    data = cop_prep(data_fn, out_fn)

    # prepare instances
    sml_out_fn = data_fn[:-4] + '_with_CoP_sml.csv'
    cop_inst_prep(out_fn, sml_out_fn)

    # # prepare datatype TODO
    # dt_out_fn = dt_fn[:-4] + '_with_CoP.csv'
    # cop_dt_prep(out_fn, dt_out_fn)

    pass
