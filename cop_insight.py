import os
import itertools
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from matplotlib import pyplot as plt


pref_columns = [
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_1_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_1_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_2_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_2_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_3_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_3_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_4_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_4_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_5_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_5_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_6_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_6_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_7_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_7_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_8_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_8_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_1_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_1_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_2_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_2_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_3_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_3_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_4_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_4_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_5_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_5_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_6_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_6_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_7_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_7_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_8_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_8_Field_of_Study',
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


def cop_prune(data_fn, out_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    out_data = data.filter(regex='Undergraduate_preferences')
    out_data['ATAR'] = data['ATAR']

    # # write the modified data to csv
    # out_data.to_csv(os.path.join(data_dir, out_fn), index=False)

    return data


data_dir = 'data'
res_dir = 'result'
dt_fn = 'Column_types.csv'

prefixes = [
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_'
]
suffixes = [
    '_Institution_Code',
    '_Field_of_Study'
]
high_range = ['1', '2', '3']
comp_range = ['1', '2', '3', '4', '5', '6', '7', '8']


def cop_prep(data_fn, out_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
    # print('\n'.join(data.keys()))

    # add features for the majority of each round
    # take the 2nd round as preATAR and 3rd round as postATAR
    def get_major(row, prefix, suffix, range):
        keys = [prefix + i  + suffix for i in range]
        count = Counter(row[keys])
        if np.nan in count:
            if count[np.nan] == len(range):
                # if all prefs are nan, mark as undecided
                return 'UD'
            count.pop(np.nan)
        s = sum(count.values())
        code, c = count.most_common(1)[0]
        if c > s / 2:
            # if some pref is more than half, mark as majority
            # todo later can verify whether marked majorities keep unchanged after ATAR
            return code
        else:
            # if no pref is more than half, mark as multiple interests
            return 'MT'
        pass

    for prefix in prefixes:
        for suffix in suffixes:
            print('processing', prefix + 'Comp_Major' + suffix)
            data[prefix + 'Comp_Major' + suffix] = data.apply(lambda x: get_major(x, prefix, suffix, comp_range), axis=1)
            print('processing', prefix + 'High_Major' + suffix)
            data[prefix + 'High_Major' + suffix] = data.apply(lambda x: get_major(x, prefix, suffix, high_range), axis=1)
            pass

    # # write the modified data to csv
    # data.to_csv(os.path.join(data_dir, out_fn), index=False)

    return data


def cop_atar(atar_fn, data_fn, out_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    atar_data = pd.read_csv(os.path.join(data_dir, atar_fn), dtype=dict(dt['dtype']))
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    data['ATAR'] = atar_data['ATAR']

    # # write the modified data to csv
    # data.to_csv(os.path.join(data_dir, out_fn), index=False)


# int_vals = {
#     '_Institution_Code': ['I12', 'I14', 'I21', 'I28', 'I32', 'I34', 'I38', 'I43'],
#     '_Field_of_Study': ['F8', 'F10', 'F3', 'F9', 'F6', 'F4']
# }
#
#
# def cop_proc(data_fn, out_fn):
#     dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
#     data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
#
#     # pick the relevant institutions and FoSs
#     keys = data.keys()
#     major_keys = list(filter(lambda x: 'Major' in x, keys))
#
#     # print(major_keys)
#
#     def interp_idx(id, ref):
#         # other == -3, MT == -2, UD == -1
#         if id == 'UD':
#             return -1
#         if id == 'MT':
#             return -2
#         if id not in ref:
#             return -3
#         return ref.index(id)
#
#     source = 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major'
#     target = 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major'
#
#     res = {}
#
#     for suffix in suffixes:
#         idcs = int_vals[suffix]
#         count = np.zeros((len(idcs) + 3, len(idcs) + 3), dtype=int)
#
#         for _, row in data.iterrows():
#             sid = row[source + suffix]
#             tid = row[target + suffix]
#             si = interp_idx(sid, idcs)
#             ti = interp_idx(tid, idcs)
#             count[si, ti] += 1
#
#         res[suffix] = count
#
#     with open(os.path.join(data_dir, out_fn), 'wb') as f:
#         pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
#
#
# def cop_post(data_fn, out_fn):
#     with open(os.path.join(data_dir, data_fn), 'rb') as f:
#         data = pickle.load(f)
#
#     with open(os.path.join(data_dir, out_fn), 'w') as f:
#         for suffix in suffixes:
#             print(',', ','.join(columns[suffix]), file=f)
#             for c, row in zip(columns[suffix], data[suffix]):
#                 print(c, ',', ','.join(row.astype(str)), file=f)


columns = {
    '_Institution_Code': ['I12', 'I14', 'I21', 'I28', 'I32', 'I34', 'I38', 'I43', 'OTH', 'MT', 'UD'],
    '_Field_of_Study': ['F8', 'F10', 'F3', 'F9', 'F6', 'F4', 'OTH', 'MT', 'UD']
}


def cop_proc_upd(data_fn, out_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    # pick the relevant institutions and FoSs
    keys = data.keys()
    major_keys = list(filter(lambda x: 'Major' in x, keys))

    # print(major_keys)

    avg_atar = data.groupby([
        'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Institution_Code',
        'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Field_of_Study',
        'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Institution_Code',
        'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Field_of_Study'
    ])['ATAR'].mean()

    atar = data.groupby([
        'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Institution_Code',
        'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Field_of_Study',
        'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Institution_Code',
        'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Field_of_Study'
    ])['ATAR'].apply(list) #  .reset_index(name='ATAR')

    count = data.groupby([
        'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Institution_Code',
        'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Field_of_Study',
        'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Institution_Code',
        'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Field_of_Study'
    ]).size()

    def interp_idx(iid, fid):
        if iid in columns['_Institution_Code']:
            i = columns['_Institution_Code'].index(iid) * len(columns['_Field_of_Study'])
        else:
            i = columns['_Institution_Code'].index('OTH') * len(columns['_Field_of_Study'])
        if fid in columns['_Field_of_Study']:
            i += columns['_Field_of_Study'].index(fid)
        else:
            i += columns['_Field_of_Study'].index('OTH')
        return i

    m = len(columns['_Institution_Code']) * len(columns['_Field_of_Study'])
    count_arr = np.zeros((m, m), dtype=int)

    for id, c in count.iteritems():
        i = interp_idx(id[0], id[1])
        j = interp_idx(id[2], id[3])
        count_arr[i, j] += c

    avg_arr = np.zeros((m, m))

    # buggy for others which may have multiple entries
    for id, a in avg_atar.iteritems():
        i = interp_idx(id[0], id[1])
        j = interp_idx(id[2], id[3])
        avg_arr[i, j] += a

    atar_arr = np.empty((m, m), dtype=object)
    for id, a in atar.iteritems():
        i = interp_idx(id[0], id[1])
        j = interp_idx(id[2], id[3])
        if atar_arr[i, j] is None:
            atar_arr[i, j] = a
        else:
            atar_arr[i, j].extend(a)

    avg_arr = np.vectorize(lambda x: np.nanmean(x) if x is not None else np.nan)(atar_arr)
    std_arr = np.vectorize(lambda x: np.nanstd(x) if x is not None else np.nan)(atar_arr)

    with open(os.path.join(data_dir, out_fn), 'wb') as f:
        pickle.dump({'count': count_arr, 'atar': atar_arr, 'avg': avg_arr, 'std': std_arr}, f, pickle.HIGHEST_PROTOCOL)


columns_readable = {
    '_Institution_Code': [
        'Australian Catholic University',
        'Deakin University',
        'La Trobe University',
        'Monash University',
        'RMIT University',
        'Swinburne University of Technology',
        'University of Melbourne',
        'Victoria University',
        'Other',
        'Multiple',
        'Undecided'
    ],
    '_Field_of_Study': [
        'Management and Commerce',
        'Creative Arts',
        'Engineering and Related Technologies',
        'Society and Culture',
        'Health',
        'Architecture and Building',
        'Other',
        'Multiple',
        'Undecided'
    ]
}


def cop_post_upd(data_fn, out_fn, atar_fn):
    with open(os.path.join(data_dir, data_fn), 'rb') as f:
        data = pickle.load(f)

    idx_0 = np.repeat(columns_readable['_Institution_Code'], len(columns_readable['_Field_of_Study']))
    idx_1 = np.tile(columns_readable['_Field_of_Study'], len(columns_readable['_Institution_Code']))

    with open(os.path.join(data_dir, out_fn), 'w') as f:
        print(',,', ','.join(idx_0), file=f)
        print(',,', ','.join(idx_1), file=f)
        for i_0, i_1, row in zip(idx_0, idx_1, data['count']):
            print(i_0, ',', i_1, ',', ','.join(row.astype(str)), file=f)

    with open(os.path.join(data_dir, atar_fn), 'w') as f:
        print(',,', ','.join(idx_0), file=f)
        print(',,', ','.join(idx_1), file=f)
        for i_0, i_1, row in zip(idx_0, idx_1, data['avg']):
            print(i_0, ',', i_1, ',', ','.join(row.astype(str)), file=f)


def cop_post_flat(data_fn, out_fn, year):
    with open(os.path.join(data_dir, data_fn), 'rb') as f:
        data = pickle.load(f)

    idx_0 = np.repeat(columns_readable['_Institution_Code'], len(columns_readable['_Field_of_Study']))
    idx_1 = np.tile(columns_readable['_Field_of_Study'], len(columns_readable['_Institution_Code']))

    n = len(data['count'])

    with open(os.path.join(data_dir, out_fn), 'a') as f:
        for i in range(n):
            for j in range(n):
                print(', '.join([year, idx_0[i], idx_1[i], idx_0[j], idx_1[j], str(data['count'][i, j]), str(data['avg'][i, j]), str(data['std'][i, j])]), file=f)


out_columns = [
    'ATAR',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_High_Major_Field_of_Study',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_High_Major_Field_of_Study'
]

inst_code_dict = {
    'I12': 'Australian Catholic University',
    'I14': 'Deakin University',
    'I21': 'La Trobe University',
    'I28': 'Monash University',
    'I32': 'RMIT University',
    'I34': 'Swinburne University of Technology',
    'I38': 'University of Melbourne',
    'I43': 'Victoria University',
    # 'OTH': 'Other',
    'MT': 'Multiple',
    'UD': 'Undecided'
}

fos_code_dict = {
    'F8': 'Management and Commerce',
    'F10': 'Creative Arts',
    'F3': 'Engineering and Related Technologies',
    'F9': 'Society and Culture',
    'F6': 'Health',
    'F4': 'Architecture and Building',
    # 'OTH': 'Other',
    'MT': 'Multiple',
    'UD': 'Undecided'
}


def cop_interp(data_fn, out_fn, year):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    data = data[out_columns]
    # data['Year'] = year
    data[out_columns[1]] = data[out_columns[1]].apply(lambda x: inst_code_dict[x] if x in inst_code_dict else 'Other')
    data[out_columns[2]] = data[out_columns[2]].apply(lambda x: fos_code_dict[x] if x in fos_code_dict else 'Other')
    data[out_columns[3]] = data[out_columns[3]].apply(lambda x: inst_code_dict[x] if x in inst_code_dict else 'Other')
    data[out_columns[4]] = data[out_columns[4]].apply(lambda x: fos_code_dict[x] if x in fos_code_dict else 'Other')

    with open(os.path.join(data_dir, out_fn), 'a') as f:
        for i, row in data.iterrows():
            print(', '.join([year, row[out_columns[1]], row[out_columns[2]], row[out_columns[3]], row[out_columns[4]], str(row[out_columns[0]])]), file=f)


if __name__ == '__main__':

    flat_fn = 'PrefFlow_flat.csv'
    with open(os.path.join(data_dir, flat_fn), 'w') as f:
        print('Year, Institution_from, FoS_from, Institution_to, FoS_to, Count, ATAR_avg, ATAR_std', file=f)

    pref_out_fn = 'Pref_flat.csv'
    with open(os.path.join(data_dir, pref_out_fn), 'w') as f:
        print('Year, Institution_from, FoS_from, Institution_to, FoS_to, ATAR', file=f)

    for y in ['2016', '2017', '2018']:
        data_fn = 'ShortFile_' + y + '.csv'
        pref_fn = 'PrefOnly_' + y + '.csv'
        # cop_prune(data_fn, pref_fn)
        out_fn = 'PrefEngineered_' + y + '.csv'
        # cop_prep(pref_fn, out_fn)
        # cop_atar(pref_fn, out_fn, out_fn)

        # pkl_fn = 'PrefEngineered_' + y + '.pkl'
        # # cop_proc(out_fn, pkl_fn)
        # flow_fn = 'PrefFlow_' + y + '.csv'
        # cop_post(pkl_fn, flow_fn)

        pkl_fn = 'PrefEngineered_upd_' + y + '.pkl'
        # cop_proc_upd(out_fn, pkl_fn)
        flow_fn = 'PrefFlow_upd_' + y + '.csv'
        atar_fn = 'PrefFlow_upd_atar_' + y + '.csv'
        # cop_post_upd(pkl_fn, flow_fn, atar_fn)

        cop_post_flat(pkl_fn, flat_fn, y)

        cop_interp(out_fn, pref_out_fn, y)
