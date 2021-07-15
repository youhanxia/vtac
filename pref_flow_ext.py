import os
import numpy as np
import pandas as pd
import pickle
# import networkx as nx
from matplotlib import pyplot as plt


data_dir = 'data'
plt_dir = 'plots'
dt_fn = 'Column_types.csv'
uni_code = 'I32'

columns = np.array([
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_1_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_2_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_3_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_4_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_5_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_6_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_7_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_8_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_1_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_2_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_3_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_4_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_5_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_6_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_7_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_8_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_1_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_2_Institution_Code',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_3_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_4_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_5_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_6_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_7_Institution_Code',
    # 'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_8_Institution_Code',
    # 'Course_Code_Offer_1_Institution_Code',
    # 'Course_Code_Offer_2_Institution_Code',
    # 'Course_Code_Offer_3_Institution_Code',
    # 'Course_Code_Offer_4_Institution_Code',
    # 'Course_Code_Offer_5_Institution_Code',
    # 'Course_Code_Offer_6_Institution_Code',
    # 'Course_Code_Offer_7_Institution_Code',
    # 'Course_Code_Offer_8_Institution_Code',
])

institutions = np.sort(['I28', 'I14', 'I12', 'I21', 'I37', 'I43', 'I13', 'I38', 'I32', 'I92', 'I34', 'I91', 'I61', 'I51', 'I86', 'I69', 'I83', 'I96', 'I98', 'I95', 'I94', 'I99', 'I17', 'I81', 'I53', 'I19', 'I97', 'I82', 'I89', 'I63'])

lookup = {}

institution_names = np.array([
    'ACU',
    'Charles Sturt University',
    'Deakin U',
    'CQUniversity',
    'Australian Maritime College',
    'La Trobe U',
    'Monash U',
    'RMIT',
    'Swinburne U',
    'Federation University Australia',
    'UoM',
    'Victoria U',
    'Box Hill Institute of TAFE',
    'Kangan Batman Institute of TAFE',
    'Holmesglen Institute of TAFE',
    'Bendigo Regional Institute of TAFE',
    'Melbourne Polytechnic',
    'Academy of Information Technology',
    'Macleay College',
    'Goulburn Ovens Institute of TAFE',
    'Collarts (Australian College of the Arts)',
    'Australian Guild of Music Education',
    'Academy of Design Australia',
    'Australian College of Applied Psychology',
    'Melbourne Institute of Technology',
    'Academy of Interactive Entertainment',
    'Elly Lukas Beauty Therapy College',
    'Footscray City Films / Footscray City College',
    'Qantm College',
    'SAE Creative Media Institute'
])

def add_flow(a, b, count):
    src = list(a - b)
    if not src:
        return
    tgt = list(b - a)
    if not tgt:
        return
    val = len(src) / len(tgt)

    for x in src:
        for y in tgt:
            count[lookup[x]][lookup[y]] += val


def ext(data_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    data = data[columns]

    m = len(institutions)

    n = data.shape[0]

    # count prefs in each round
    pref_count = np.zeros((3, m), dtype=int)

    for i, col in enumerate(columns):
        count = data.groupby(col, as_index=False).size()
        for j, ins in enumerate(institutions):
            if ins not in count:
                continue
            pref_count[i // 3][j] += count[ins]

    # count the preference flow
    pref_flow = np.zeros((2, m, m), dtype=float)

    rnd_0 = data.apply(lambda x: set(x[columns[:3]]) - {np.nan}, axis=1)
    rnd_1 = data.apply(lambda x: set(x[columns[3:6]]) - {np.nan}, axis=1)
    rnd_2 = data.apply(lambda x: set(x[columns[6:]]) - {np.nan}, axis=1)

    for i in range(n):
        add_flow(rnd_0[i], rnd_1[i], pref_flow[0])
        add_flow(rnd_1[i], rnd_2[i], pref_flow[1])

    # store result
    with open(os.path.join(data_dir,'pref_count_' + data_fn[-8:-4] + '.pkl'), 'wb') as f:
        pickle.dump({'pref_count': pref_count, 'pref_flow': pref_flow}, f, protocol=pickle.HIGHEST_PROTOCOL)


def post_proc(res_fn):

    with open(os.path.join(data_dir,res_fn), 'rb') as f:
        res = pickle.load(f)

    m = len(institutions)

    # for i in range(m):
    #     print(institutions[i], res['pref_count'][0][i], res['pref_count'][1][i], res['pref_count'][2][i], sep='\t')
    #
    # for i in range(m):
    #     for j in range(m):
    #         print(institutions[i], institutions[j], res['pref_flow'][0][i][j], res['pref_flow'][1][i][j], sep='\t')

    pass


def bar_plot(res_fn, leave=False, ratio=True, sort_ratio=False, top_n=5):
    with open(os.path.join(data_dir,res_fn), 'rb') as f:
        res = pickle.load(f)

    title = 'pref_flow_' + res_fn[-8:-4] + ''

    if leave:
        data = res['pref_flow'][:, lookup[uni_code], :]
        title += '_from_RMIT'
    else:
        data = res['pref_flow'][:, :, lookup[uni_code]]
        title += '_to_RMIT'
        if ratio:
            if sort_ratio:
                title += '_top'
            title += '_ratio'

    if not top_n:
        top_n = data.shape[1]
    else:
        title += '_top_' + str(top_n)

    if sort_ratio:
        if not leave and ratio:
            denom = res['pref_count'][:2]
            data = np.divide(data, denom, out=np.zeros_like(data), where=denom!=0)

        idx = np.argsort(data[1])[-top_n:][::-1]
        data = data[:, idx]
    else:
        idx = np.argsort(data[1])[-top_n:][::-1]
        data = data[:, idx]

        if not leave and ratio:
            denom = res['pref_count'][:2, idx]
            data /= denom

    plt.figure(figsize=(10, 10))

    ind = np.arange(top_n)
    width = 0.35
    plt.bar(ind, data[0], width, label='1st-2nd')
    plt.bar(ind + width, data[1], width,
            label='2nd-3rd')

    plt.title(title)
    plt.xticks(ind + width / 2, institution_names[idx])
    plt.legend(loc='best')

    plt.savefig(os.path.join(plt_dir, title))


if __name__ == '__main__':
    for i, ins in enumerate(institutions):
        lookup[ins] = i

    years = [2016, 2017, 2018]
    for y in years:
        data_fn = 'ShortFile_' + str(y) + '.csv'
        # ext(data_fn)
        res_fn = 'pref_count_' + str(y) + '.pkl'
        # post_proc(res_fn)
        bar_plot(res_fn)
        bar_plot(res_fn, sort_ratio=True)
        bar_plot(res_fn, ratio=False)
        bar_plot(res_fn, leave=True)
