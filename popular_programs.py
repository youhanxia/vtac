import os
import numpy as np
import pandas as pd

data_dir = 'data'
dt_fn = 'Column_types.csv'


columns = np.array([
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_1',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_2',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_3',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_4',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_5',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_6',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_7',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_8',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_1',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_2',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_3',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_4',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_5',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_6',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_7',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_8',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_1',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_2',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_3',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_4',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_5',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_6',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_7',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_8',
])


def course_code_ext(data_fn, out_fn):
    # dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), low_memory=False, dtype=str)
    # print('\n'.join(data.keys()))

    data = data.filter(items=columns).astype(float).fillna(0).astype(int).astype(str)
    data.replace('0', np.nan, inplace=True)

    # write the modified data to csv
    data.to_csv(os.path.join(data_dir, out_fn), index=False)
    return data


institution_id = '32'

title = [
    'VTAC_timeley_applications_close',
    'VTAC_late_close_and_Early_worklist',
    'main_round_worklist'
]

def pop_course_count(data_fn, course_fn):
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=str)

    # prepare course code lookup
    out_data = pd.read_csv(os.path.join(data_dir, course_fn), dtype=str, index_col='VTAC.Course.Code')
    out_data = out_data.reset_index().drop_duplicates(subset='VTAC.Course.Code', keep='last').set_index('VTAC.Course.Code')
    out_data = out_data.filter(items=['Institution.Name', 'Course.Name'])
    out_data.columns = ['Institution_Name', 'Course_Name']
    out_data.index.names = ['Course_Code']

    pd.set_option('display.max_rows', 100)

    for i in range(3):
        print(title[i])
        total = pd.Series()
        for k in columns[i * 8: i * 8 + 3]:
            # pass low range preferences
            count = data[k].value_counts()
            # count = count.filter(regex='^' + institution_id)
            count = count[count.index.str.startswith(institution_id)]
            total = total.add(count, fill_value=0)
        total.index = total.index.astype(int)

        out_data['Count'] = total.astype(int)
        out_data['Institution_Name'] = out_data['Institution_Name'].apply(str.rstrip)
        out_data['Course_Name'] = out_data['Course_Name'].apply(str.rstrip)
        out_data = out_data.sort_values(by='Count', ascending=False)

        # write the modified data to csv
        out_data.to_csv(os.path.join(data_dir, 'RMIT_Program_popularity_' + title[i] + data_fn[-9:-4] + '.csv'))


if __name__ == '__main__':
    # course_code_ext('ReformattedFile_2016.csv', 'CourseCode_Only_2016.csv')
    # course_code_ext('ReformattedFile_2017.csv', 'CourseCode_Only_2017.csv')
    # course_code_ext('ReformattedFile_2018.csv', 'CourseCode_Only_2018.csv')

    pop_course_count('CourseCode_Only_2016.csv', 'CourseCode_FoS_Inst.csv')
    pop_course_count('CourseCode_Only_2017.csv', 'CourseCode_FoS_Inst.csv')
    pop_course_count('CourseCode_Only_2018.csv', 'CourseCode_FoS_Inst.csv')

    pass
