import os
import pandas as pd
from externals.stucco import ContrastSetLearner
import pickle

data_dir = 'data'
res_dir = 'result'
dt_fn = 'Column_types.csv'

target = 'Changed'


def con_min_exp(data_fn, res_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    learner = ContrastSetLearner(data, group_feature=target)
    learner.learn(max_length=3)
    output = learner.score(min_lift=3)
    with open(os.path.join(res_dir, res_fn), 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)


def con_min_post(res_fn):
    with open(os.path.join(res_dir, res_fn), 'rb') as f:
        res = pickle.load(f)
    print(res)


if __name__ == '__main__':
    data_fn = 'MinimalFile_merged_with_CoP.csv'

    # contrast pattern mining
    con_min_res_fn = 'con_min_res.pkl'
    con_min_exp(data_fn, con_min_res_fn)
    con_min_post(con_min_res_fn)
