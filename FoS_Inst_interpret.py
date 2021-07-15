import os
import numpy as np
import pandas as pd


data_dir = 'data'

institutions = np.sort(['I28', 'I14', 'I12', 'I21', 'I37', 'I43', 'I13', 'I38', 'I32', 'I92', 'I34', 'I91', 'I61', 'I51', 'I86', 'I69', 'I83', 'I96', 'I98', 'I95', 'I94', 'I99', 'I17', 'I81', 'I53', 'I19', 'I97', 'I82', 'I89', 'I63'])


def inst_file_prep(data_fn, out_fn):

    inst_name = 'Institution.Name'
    inst_code = 'Institution.Code'

    data = pd.read_csv(os.path.join(data_dir, data_fn))
    data = data[[inst_code, inst_name]]
    data.drop_duplicates(subset=inst_code, inplace=True)

    # idx = map(lambda x: int(x[1:]), institutions)
    data[inst_code] = data[inst_code].apply(lambda x: 'I' + str(x))
    data = data[data[inst_code].isin(institutions)]
    data.sort_values(inst_code, inplace=True)

    # write the modified data to csv
    data.to_csv(os.path.join(data_dir, out_fn), index=False)


if __name__ == '__main__':
    inst_file_prep('CourseCode_FoS_Inst.csv', 'institution_names.csv')
