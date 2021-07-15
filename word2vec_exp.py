import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from matplotlib import pyplot as plt


data_dir = 'data'
dt_fn = 'Column_types.csv'

model_fn = 'word2vec.model'


columns = [
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
    'Course_Code_Offer_1_Institution_Code',
    'Course_Code_Offer_1_Field_of_Study',
    'Course_Code_Offer_2_Institution_Code',
    'Course_Code_Offer_2_Field_of_Study',
    'Course_Code_Offer_3_Institution_Code',
    'Course_Code_Offer_3_Field_of_Study',
    'Course_Code_Offer_4_Institution_Code',
    'Course_Code_Offer_4_Field_of_Study',
    'Course_Code_Offer_5_Institution_Code',
    'Course_Code_Offer_5_Field_of_Study',
    'Course_Code_Offer_6_Institution_Code',
    'Course_Code_Offer_6_Field_of_Study',
    'Course_Code_Offer_7_Institution_Code',
    'Course_Code_Offer_7_Field_of_Study',
    'Course_Code_Offer_8_Institution_Code',
    'Course_Code_Offer_8_Field_of_Study',
]


def embedding(data_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    print(data.shape)

    # print('\'\n    \''.join(data.keys()))

    # convert to list of list
    data = data[columns].astype(str).values.tolist()

    # remove all nan
    data = [list(filter('nan'.__ne__, l)) for l in data]

    # train word2vec
    model = Word2Vec(data, min_count=1, size=10, workers=3, window=20, sg=1)

    model.save(model_fn)


def post_proc():
    model = Word2Vec.load(model_fn)
    vecs = []
    words = []

    for _, word in enumerate(model.wv.vocab):
        words.append(word)
        vecs.append(model.wv[word])

    vecs = np.array(vecs)
    tsne = TSNE(n_components=2)
    ld_vecs = tsne.fit_transform(vecs)

    xs = ld_vecs[:, 0]
    ys = ld_vecs[:, 1]

    plt.figure(figsize=(10, 10))
    plt.scatter(xs, ys)

    for i, word in enumerate(words):
        plt.annotate(word, (xs[i], ys[i]))
    plt.show()

    pass


if __name__ == '__main__':
    data_fn = 'ShortFile_merged.csv'

    # embedding(data_fn)
    post_proc()
    pass
