import os
import numpy as np
import pandas as pd

data_dir = 'data'
dt_fn = 'Column_types.csv'

# num_gender = {'F': -1.0, 'X': 0.0, 'M': 1.0}
# num_born = {'O': 0.0, 'A': 1.0}
# num_lang = {'O': 0.0, 'E': 1.0}
# num_school = {}


# def fos_uni_lookup_gen(data_fn):
#     data = pd.read_excel(os.path.join(data_dir, data_fn), sheet_name=None)['New.Field.Of.Study']
#     # print(data.keys())
#     data['Root.Field.Of.Study'] = data['New.Field.Of.Study'].map(lambda x: int(x / 1e4))
#     data['Institution.Code'] = data['VTAC.Course.Code'].map(lambda x: int(x / 1e8))
#     data.drop_duplicates('VTAC.Course.Code', inplace=True)
#     data.set_index('VTAC.Course.Code', inplace=True)
#     data.to_csv(os.path.join(data_dir, 'CourseCode_FoS_Inst.csv'))

to_remove = np.array([
    'VTAC_ID_Number', 'Surname', 'First_Given_Name', 'Second_Given_Name', 'Postal_address_line_1',
    'Postal_address_line_2', 'Postal_Suburb', 'Postal_State', 'Postal_Postcode', 'Postal_Overseas_State_Province',
    'Postal_Overseas_Zip_Postcode', 'Postal_Country', 'Residential_address_line_1', 'Residential_address_line_2',
    'Residential_Overseas_State_Province', 'Residential_Overseas_Zip_Postcode', 'Home_Phone_Number',
    'Business_Phone_Number', 'Mobile_Phone', 'Electronic_mail_address', 'Previous_Surname', 'Previous_First_Given_Name',
    'Previous_Second_Given_Name', 'CHESSN', 'Proxy_name', 'Proxy_relationship', 'Proxy_DOB', 'Proxy_phone',
    'Proxy_email', 'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_9',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_10',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_11',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_timeley_applications_close_12',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_9',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_10',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_11',
    'Undergraduate_preferences_as_at_the_time_of_VTAC_late_close_and_Early_worklist_12',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_9',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_10',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_11',
    'Undergraduate_preferences_as_at_the_time_of_main_round_worklist_12', 'Course_Code_Offer_9', 'Prerequisite_met__9',
    'Offer_Round_9', 'Enrolment_Status_9', 'Fee_eligibility_for_Type_4_enrolled_course_9', 'Course_Code_Offer_10',
    'Prerequisite_met__10', 'Offer_Round_10', 'Enrolment_Status_10', 'Fee_eligibility_for_Type_4_enrolled_course_10',
    'Course_Code_Offer_11', 'Prerequisite_met__11', 'Offer_Round_11', 'Enrolment_Status_11',
    'Fee_eligibility_for_Type_4_enrolled_course_11', 'Course_Code_Offer_12', 'Prerequisite_met__12', 'Offer_Round_12',
    'Enrolment_Status_12', 'Fee_eligibility_for_Type_4_enrolled_course_12', 'Offer_type_Supplement_Offer',
    'Course_Code_Supplement_Offer', 'Offer_Round_Supplement_Offer', 'Enrolment_Status_Supplement_Offer',
    'Fee_eligibility_for_Type_4_enrolled_course_Supplement_Offer', 'Worklist_remarks_Current_Undergraduate_offer',
    'Institutional_remarks_Current_Undergraduate_offer', 'Basis_of_Selection_Current_Undergraduate_offer',
    'Fee_eligibility_for_Type_4_offered_course', 'GET_preferences_as_at_the_time_of_VTAC_timely_applications_close',
    'GET_preferences_as_at_the_time_of_main_round_worklist', 'Course_Code_applied_1_teaching_preferences',
    'Offer_Rank_Current_Undergraduate_offer',
    'Overseas_Equivalent_Secondary_1',
    'Overseas_Equivalent_Secondary_2',
    'Overseas_Equivalent_Secondary_3',
    'Overseas_Equivalent_Secondary_4',
    'Specialist_teaching_area_A_applied_1_teaching_preferences',
    'Specialist_teaching_area_B_applied_1_teaching_preferences',
    'Applicant_will_consider_alternative_methods_1_teaching_preferences', 'Offer_Round_1_teaching_preferences',
    'Offered_specialist_teaching_area_A_1_teaching_preferences',
    'Offered_specialist_teaching_area_B_1_teaching_preferences', 'Enrolment_Status_1_teaching_preferences',
    'Enrolled_specialist_teaching_area_A_1_teaching_preferences',
    'Enrolled_specialist_teaching_area_B_1_teaching_preferences', 'Course_Code_applied_2_teaching_preferences',
    'Specialist_teaching_area_A_applied_2_teaching_preferences',
    'Specialist_teaching_area_B_applied_2_teaching_preferences',
    'Applicant_will_consider_alternative_methods_2_teaching_preferences', 'Offer_Round_2_teaching_preferences',
    'Offered_specialist_teaching_area_A_2_teaching_preferences',
    'Offered_specialist_teaching_area_B_2_teaching_preferences', 'Enrolment_Status_2_teaching_preferences',
    'Enrolled_specialist_teaching_area_A_2_teaching_preferences',
    'Enrolled_specialist_teaching_area_B_2_teaching_preferences', 'Course_Code_applied_3_teaching_preferences',
    'Specialist_teaching_area_A_applied_3_teaching_preferences',
    'Specialist_teaching_area_B_applied_3_teaching_preferences',
    'Applicant_will_consider_alternative_methods_3_teaching_preferences', 'Offer_Round_3_teaching_preferences',
    'Offered_specialist_teaching_area_A_3_teaching_preferences',
    'Offered_specialist_teaching_area_B_3_teaching_preferences', 'Enrolment_Status_3_teaching_preferences',
    'Enrolled_specialist_teaching_area_A_3_teaching_preferences',
    'Enrolled_specialist_teaching_area_B_3_teaching_preferences', 'Course_Code_applied_4_teaching_preferences',
    'Specialist_teaching_area_A_applied_4_teaching_preferences',
    'Specialist_teaching_area_B_applied_4_teaching_preferences',
    'Applicant_will_consider_alternative_methods_4_teaching_preferences', 'Offer_Round_4_teaching_preferences',
    'Offered_specialist_teaching_area_A_4_teaching_preferences',
    'Offered_specialist_teaching_area_B_4_teaching_preferences', 'Enrolment_Status_4_teaching_preferences',
    'Enrolled_specialist_teaching_area_A_4_teaching_preferences',
    'Enrolled_specialist_teaching_area_B_4_teaching_preferences', 'Offer_type', 'Course_Code', 'Offer_Round',
    'Offered_specialist_teaching_area_A', 'Offered_specialist_teaching_area_B', 'Enrolment_Status',
    'Enrolled_specialist_teaching_area_A', 'Enrolled_specialist_teaching_area_B', 'Worklist_remarks_Current_GET_offer',
    'Institutional_remarks_Current_GET_offer', 'UMAT_ISAT_identification', 'Scholarship_application__via_VTAC_',
    'Course_Code_applied', 'Specialist_teaching_area_A_applied', 'Specialist_teaching_area_B_applied',
    'Applicant_will_consider_alternative_methods', 'Offer_Round_9', 'Offered_specialist_teaching_area_A',
    'Offered_specialist_teaching_area_B', 'Enrolment_Status_9', 'Enrolled_specialist_teaching_area_A',
    'Enrolled_specialist_teaching_area_B', 'Course_Code_applied_1', 'Specialist_teaching_area_A_applied_1',
    'Specialist_teaching_area_B_applied_1', 'Applicant_will_consider_alternative_methods_1', 'Offer_Round_10',
    'Offered_specialist_teaching_area_A_1', 'Offered_specialist_teaching_area_B_1', 'Enrolment_Status_10',
    'Enrolled_specialist_teaching_area_A_1', 'Enrolled_specialist_teaching_area_B_1', 'Course_Code_applied_2',
    'Specialist_teaching_area_A_applied_2', 'Specialist_teaching_area_B_applied_2',
    'Applicant_will_consider_alternative_methods_2', 'Offer_Round_11', 'Offered_specialist_teaching_area_A_2',
    'Offered_specialist_teaching_area_B_2', 'Enrolment_Status_11', 'Enrolled_specialist_teaching_area_A_2',
    'Enrolled_specialist_teaching_area_B_2', 'Course_Code_applied_3', 'Specialist_teaching_area_A_applied_3',
    'Specialist_teaching_area_B_applied_3', 'Applicant_will_consider_alternative_methods_3', 'Offer_Round_12',
    'Offered_specialist_teaching_area_A_3', 'Offered_specialist_teaching_area_B_3', 'Enrolment_Status_12',
    'Enrolled_specialist_teaching_area_A_3', 'Enrolled_specialist_teaching_area_B_3', 'Offer_type_1', 'Course_Code_9',
    'Offer_Round_13', 'Offered_specialist_teaching_area_A_4', 'Offered_specialist_teaching_area_B_4',
    'Enrolment_Status_13', 'Enrolled_specialist_teaching_area_A_4', 'Enrolled_specialist_teaching_area_B_4',
    'Enrolment_Status_99', 'Offer_Round_99', 'Overseas_Equivalent_1', 'Overseas_Equivalent_2', 'Overseas_Equivalent_3',
    'Overseas_Equivalent_4', 'Overseas_Equivalent_5', 'Overseas_Equivalent_6', 'Overseas_Equivalent_7',
    'Overseas_Equivalent_8', 'Overseas_Equivalent_9', 'Overseas_Equivalent_10', 'Overseas_Equivalent_11',
    'Overseas_Equivalent_111', 'Overseas_Equivalent_222', 'Overseas_Equivalent_333', 'Overseas_Equivalent_444',
    'Post_Sec_Name_1', 'Post_Sec_Name_2', 'Post_Sec_Name_3', 'Post_Sec_Name_4', 'Post_Sec_Name_5', 'Post_Sec_Name_6',
    'Post_Sec_Name_7', 'Post_Sec_Name_8', 'Post_Sec_Name_9', 'Post_Sec_Name_10', 'Post_Sec_Name_11',
    'Proof_of_Completion_1', 'Proof_of_Completion_2', 'Proof_of_Completion_3', 'Proof_of_Completion_4',
    'Proof_of_Completion_5', 'Proof_of_Completion_6', 'Proof_of_Completion_7', 'Proof_of_Completion_8',
    'Proof_of_Completion_9', 'Proof_of_Completion_10', 'Proof_of_Completion_11', 'Residential_Suburb',
    'Overseas_year_12_award_name_Secondary_1', 'Overseas_year_12_award_name_Secondary_2',
    'Overseas_year_12_award_name_Secondary_3', 'Overseas_year_12_award_name_Secondary_4', 'School_Name_Secondary_1',
    'School_Name_Secondary_2', 'School_Name_Secondary_3', 'School_Name_Secondary_4',
    'Verification_of_claim_Secondary_1', 'Verification_of_claim_Secondary_2', 'Verification_of_claim_Secondary_3',
    'Verification_of_claim_Secondary_4', 'Post_Sec_Institution_Name_1', 'Post_Sec_Institution_Name_2',
    'Post_Sec_Institution_Name_3', 'Post_Sec_Institution_Name_4', 'Post_Sec_Institution_Name_5',
    'Post_Sec_Institution_Name_6', 'Post_Sec_Institution_Name_7', 'Post_Sec_Institution_Name_8',
    'Post_Sec_Institution_Name_9', 'Post_Sec_Institution_Name_10', 'Post_Sec_Institution_Name_11',
    'Overseas_Post_Sec__Country_1', 'Overseas_Post_Sec__Country_2', 'Overseas_Post_Sec__Country_3',
    'Overseas_Post_Sec__Country_4', 'Overseas_Post_Sec__Country_5', 'Overseas_Post_Sec__Country_6',
    'Overseas_Post_Sec__Country_7', 'Overseas_Post_Sec__Country_8', 'Overseas_Post_Sec__Country_9',
    'Overseas_Post_Sec__Country_10', 'Overseas_Post_Sec__Country_11', 'Post_Sec_Student_ID_Number_1',
    'Post_Sec_Student_ID_Number_2', 'Post_Sec_Student_ID_Number_3', 'Post_Sec_Student_ID_Number_4',
    'Post_Sec_Student_ID_Number_5', 'Post_Sec_Student_ID_Number_6', 'Post_Sec_Student_ID_Number_7',
    'Post_Sec_Student_ID_Number_8', 'Post_Sec_Student_ID_Number_9', 'Post_Sec_Student_ID_Number_10',
    'Post_Sec_Student_ID_Number_11', 'Verification_of_claim_1', 'Verification_of_claim_2', 'Verification_of_claim_3',
    'Verification_of_claim_4', 'Verification_of_claim_5', 'Verification_of_claim_6', 'Verification_of_claim_7',
    'Verification_of_claim_8', 'Verification_of_claim_9', 'Verification_of_claim_10', 'Verification_of_claim_11',
    'Name_and_location_of_indigenous_community'])


def csv_cleaning(data_fn, out_fn):
    global to_remove

    # read in
    data = pd.read_csv(os.path.join(data_dir, data_fn), low_memory=False)
    # print(data.dtypes)
    # print('\n'.join(data.keys()))

    # # reformat field names
    # ks = data.keys()
    # for c in ' ./?-()':
    #     ks = ks.map(lambda x: x.replace(c, '_'))
    # data.columns = ks

    # # remove irrelevant columns
    # data.drop(columns=to_remove, errors='ignore', inplace=True)

    # # eliminate blank columns, deprecated
    # data.dropna(axis=1, how='all', inplace=True)

    # # desplay the count of each repeating values in a column
    # print(data.pivot_table(index=['Study_Count'], aggfunc='size'))

    # # write the modified data to csv
    # data.to_csv(os.path.join(data_dir, out_fn), index=False)
    return data


bool_map = {
    'Tertiary_Entrance_Requirement_status': {
        np.nan: True,
        'P': False
    },
    # 'Gender': {
    #     'F': True,
    #     'M': False,
    #     'X': np.nan
    # },
    'Prerequisite_met__1': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__2': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__3': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__4': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__5': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__6': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__7': {
        'Y': True,
        np.nan: False
    },
    'Prerequisite_met__8': {
        'Y': True,
        np.nan: False
    },
    # 'Type_of_schooling_indicator_Secondary_1': {
    #     'S': True,
    #     'N': False
    # },
    # 'Type_of_schooling_indicator_Secondary_2': {
    #     'S': True,
    #     'N': False
    # },
    # 'Type_of_schooling_indicator_Secondary_3': {
    #     'S': True,
    #     'N': False
    # },
    # 'Type_of_schooling_indicator_Secondary_4': {
    #     'S': True,
    #     'N': False
    # },
    'Country_Secondary_1': {
        np.nan: True,
    },
    'Country_Secondary_2': {
        np.nan: True,
    },
    'Country_Secondary_3': {
        np.nan: True,
    },
    'Country_Secondary_4': {
        np.nan: True,
    },
    # 'UMAT_or_ISAT_type': {
    #     'U': True,
    #     'I': False
    # },
    'VTAC_personal_history_submission': {
        'Y': True,
        np.nan: False
    },
    'SEAS_application__Special_Entry_Access_Schemes_': {
        'Y': True,
        np.nan: False
    },
    'Where_Applicant_Was_Born': {
        'A': True,
        'O': False
    },
    'Principal_Language_Spoken_At_Home': {
        'E': True,
        'O': False
    },
    'Hearing_Problem': {
        'Y': True,
        np.nan: False
    },
    'Learning_Problem': {
        'Y': True,
        np.nan: False
    },
    'Medical_Problem': {
        'Y': True,
        np.nan: False
    },
    'Mobility_Problem': {
        'Y': True,
        np.nan: False
    },
    'Vision_Problem': {
        'Y': True,
        np.nan: False
    },
    'Other_Problem': {
        'Y': True,
        np.nan: False
    },
    'Applicant_requires_advice_on_disability_support_services': {
        'Y': True,
        np.nan: False
    },
    'VCAA_Enhancement_Higher_Education_studies_reported': {
        'Y': True,
        np.nan: False
    },
    'VET_studies_reported': {
        'Y': True,
        np.nan: False
    },
}

course_code_feas = [
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
    'Course_Code_Offer_1',
    'Course_Code_Offer_2',
    'Course_Code_Offer_3',
    'Course_Code_Offer_4',
    'Course_Code_Offer_5',
    'Course_Code_Offer_6',
    'Course_Code_Offer_7',
    'Course_Code_Offer_8']

post_sec_gpa_prefix = 'Post_Secondary_Grade_Point_Average_Result_'
post_sec_gpa_n = 11

sec_study_res_year_prefix = 'Result_Year_'
sec_study_score_prefix = 'Study_Score_or_old_Subject_Mark_'
sec_study_non_vce_type_prefix = 'Non_VCE_Study_Type_'
sec_study_score_dec_prefix = 'Study_Score_decoded_'
sec_study_n = 35

disabilities = [
    'Hearing_Problem',
    'Learning_Problem',
    'Medical_Problem',
    'Mobility_Problem',
    'Vision_Problem',
    'Other_Problem'
]


def fea_prep(data_fn, out_fn):
    # read in
    data = pd.read_csv(os.path.join(data_dir, data_fn), low_memory=False)
    # pd.set_option('display.max_rows', 600)

    # print('\n'.join(data.keys()))
    # print(data.groupby('VTAC_Indicator_for_Year_Secondary_1', as_index=False).size())
    # print(data['Post_Secondary_Grade_Point_Average_Result_1'].count())
    pass

    # # print datatype of each column
    # print(data.dtypes)

    # # decode boolean features
    # data.replace(bool_map, inplace=True)
    # for i in ['1', '2', '3', '4']:
    #     data['Country_Secondary_' + i] = data['Country_Secondary_' + i].replace(regex=r'\w+', value=False)
    # print(data.pivot_table(index=['Principal_Language_Spoken_At_Home'], aggfunc='size'))

    # # decode DoB
    # data['Date_of_Birth'] = pd.to_datetime(data['Date_of_Birth'])
    # data['Year_of_Birth'] = data['Date_of_Birth'].apply(lambda x: x.year)
    # data['Month_of_Birth'] = data['Date_of_Birth'].apply(lambda x: x.month)
    # # print(data.pivot_table(index=['Month_of_Birth'], aggfunc='size'))

    # # write the modified data to csv
    # data.to_csv(os.path.join(data_dir, out_fn), index=False)
    return data


def fea_eng(data_fn, out_fn, fos_fn):
    # read in
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    # data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
    data = pd.read_csv(os.path.join(data_dir, data_fn), low_memory=False)
    fos = pd.read_csv(os.path.join(data_dir, fos_fn), index_col='VTAC.Course.Code')
    # data = pd.read_csv(os.path.join(data_dir, data_fn), low_memory=False)

    # print('\n'.join(data.keys()))
    # print(data.groupby('VTAC_Indicator_for_Year_Secondary_1', as_index=False).size())
    # print(data['Post_Secondary_Grade_Point_Average_Result_1'].count())
    pass

    # # print datatype of each column
    # print(data.dtypes)

    # decode courese code into FoS and University code
    fos_root = dict(fos['Root.Field.Of.Study'])
    for fea in course_code_feas:
        data[fea + '_Institution_Code'] = data[fea].map(lambda x: 'I' + str(x)[:2] if float(x) in fos.index else np.nan,
                                                        na_action='ignore')
        data[fea + '_Field_of_Study'] = data[fea].map(
            lambda x: 'F' + str(fos_root[float(x)]) if float(x) in fos_root.keys() else np.nan, na_action='ignore')
    data.drop(columns=course_code_feas, inplace=True)

    # decode parent educational attainment
    def func(x):
        t = int(x) % 10
        return -1 if t == 9 else 6 - t

    data['fpe'] = data['Educational_attainment_of_first_parent_guardian'].map(func, na_action='ignore')
    data['spe'] = data['Educational_attainment_of_second_parent_guardian'].map(func, na_action='ignore')
    data['Educational_attainment_max'] = data[['fpe', 'spe']].max(axis=1)
    data['Educational_attainment_min'] = data[['fpe', 'spe']].min(axis=1)
    data.drop(columns=[
        'Educational_attainment_of_first_parent_guardian',
        'Educational_attainment_of_first_parent_guardian',
        'fpe', 'spe'], inplace=True)

    # compute school avg and std
    # nan atar score for school detected
    school_mean = data.groupby('School_Code_Secondary_1', as_index=False)['ATAR'].mean().set_index(
        'School_Code_Secondary_1')
    data['School_Average_Secondary_1'] = data['School_Code_Secondary_1'].map(lambda x: school_mean.loc[x]['ATAR'],
                                                                             na_action='ignore')

    # # qualification counts, secondary and post sec
    # pass

    # get sec span from sec course year
    sec_study_res_year = [sec_study_res_year_prefix + str(i + 1) for i in range(sec_study_n)]
    latest = data[sec_study_res_year].max(axis=1)
    earlist = data[sec_study_res_year].min(axis=1)
    data['Secondary_study_span'] = latest - earlist

    # decode sec study scores
    def decode_func(row, i):
        score = pd.to_numeric(row[sec_study_score_prefix + str(i)], errors='coerce')
        if row[sec_study_non_vce_type_prefix + str(i)] == 'E':
            score /= 2
        return score

    for i in range(1, sec_study_n + 1):
        scores = data.apply(lambda x: decode_func(x, i), axis=1)
        data[sec_study_score_dec_prefix + str(i)] = scores

    # get sec study avg score
    sec_study_score = [sec_study_score_dec_prefix + str(i + 1) for i in range(sec_study_n)]
    avg = data[sec_study_score].mean(axis=1)
    # print(avg.count())
    # print(avg)
    data['Secondary_study_average'] = avg
    # print(data.groupby('Secondary_study_average', as_index=False).size())

    # add any of the disabilities
    disability = data[disabilities].apply(any, axis=1)
    data['Disability'] = disability
    # print(data.groupby('Disability', as_index=False).size())

    pass

    # write the modified data to csv
    data.to_csv(os.path.join(data_dir, out_fn), index=False)
    return data


to_prune = [
    'Result_Year_Secondary_1',
    'Type_of_schooling_indicator_Secondary_1',
    'State_jurisdiction_of_year_12_claim_Secondary_1',
    'YR_12_eligibility_flag_as_reported_by_state_authority_Secondary_1',
    'VTAC_Indicator_for_Year_Secondary_1',
    'School_Code_Secondary_1',
    'School_Postcode_Secondary_1',
    'Country_Secondary_1',
    'Result_Year_Secondary_2',
    'Type_of_schooling_indicator_Secondary_2',
    'State_jurisdiction_of_year_12_claim_Secondary_2',
    'YR_12_eligibility_flag_as_reported_by_state_authority_Secondary_2',
    'VTAC_Indicator_for_Year_Secondary_2',
    'School_Code_Secondary_2',
    'School_Postcode_Secondary_2',
    'Country_Secondary_2',
    'Result_Year_Secondary_3',
    'Type_of_schooling_indicator_Secondary_3',
    'State_jurisdiction_of_year_12_claim_Secondary_3',
    'YR_12_eligibility_flag_as_reported_by_state_authority_Secondary_3',
    'VTAC_Indicator_for_Year_Secondary_3',
    'School_Code_Secondary_3',
    'School_Postcode_Secondary_3',
    'Country_Secondary_3',
    'Result_Year_Secondary_4',
    'Type_of_schooling_indicator_Secondary_4',
    'State_jurisdiction_of_year_12_claim_Secondary_4',
    'YR_12_eligibility_flag_as_reported_by_state_authority_Secondary_4',
    'VTAC_Indicator_for_Year_Secondary_4',
    'School_Code_Secondary_4',
    'School_Postcode_Secondary_4',
    'Country_Secondary_4',
    'Post_Sec_Year_Started_1',
    'Post_Sec_Year_Completed_1',
    'Post_Sec_Level_1',
    'Post_Sec_Institution_Code_1',
    'Post_Secondary_Completion_1',
    'Post_Secondary_Grade_Point_Average_Result_1',
    'Post_Sec_Year_Started_2',
    'Post_Sec_Year_Completed_2',
    'Post_Sec_Level_2',
    'Post_Sec_Institution_Code_2',
    'Post_Secondary_Completion_2',
    'Post_Secondary_Grade_Point_Average_Result_2',
    'Post_Sec_Year_Started_3',
    'Post_Sec_Year_Completed_3',
    'Post_Sec_Level_3',
    'Post_Sec_Institution_Code_3',
    'Post_Secondary_Completion_3',
    'Post_Secondary_Grade_Point_Average_Result_3',
    'Post_Sec_Year_Started_4',
    'Post_Sec_Year_Completed_4',
    'Post_Sec_Level_4',
    'Post_Sec_Institution_Code_4',
    'Post_Secondary_Completion_4',
    'Post_Secondary_Grade_Point_Average_Result_4',
    'Post_Sec_Year_Started_5',
    'Post_Sec_Year_Completed_5',
    'Post_Sec_Level_5',
    'Post_Sec_Institution_Code_5',
    'Post_Secondary_Completion_5',
    'Post_Secondary_Grade_Point_Average_Result_5',
    'Post_Sec_Year_Started_6',
    'Post_Sec_Year_Completed_6',
    'Post_Sec_Level_6',
    'Post_Sec_Institution_Code_6',
    'Post_Secondary_Completion_6',
    'Post_Secondary_Grade_Point_Average_Result_6',
    'Post_Sec_Year_Started_7',
    'Post_Sec_Year_Completed_7',
    'Post_Sec_Level_7',
    'Post_Sec_Institution_Code_7',
    'Post_Secondary_Completion_7',
    'Post_Secondary_Grade_Point_Average_Result_7',
    'Post_Sec_Year_Started_8',
    'Post_Sec_Year_Completed_8',
    'Post_Sec_Level_8',
    'Post_Sec_Institution_Code_8',
    'Post_Secondary_Completion_8',
    'Post_Secondary_Grade_Point_Average_Result_8',
    'Post_Sec_Year_Started_9',
    'Post_Sec_Year_Completed_9',
    'Post_Sec_Level_9',
    'Post_Sec_Institution_Code_9',
    'Post_Secondary_Completion_9',
    'Post_Secondary_Grade_Point_Average_Result_9',
    'Post_Sec_Year_Started_10',
    'Post_Sec_Year_Completed_10',
    'Post_Sec_Level_10',
    'Post_Sec_Institution_Code_10',
    'Post_Secondary_Completion_10',
    'Post_Secondary_Grade_Point_Average_Result_10',
    'Post_Sec_Year_Started_11',
    'Post_Sec_Year_Completed_11',
    'Post_Sec_Level_11',
    'Post_Sec_Institution_Code_11',
    'Post_Secondary_Completion_11',
    'Post_Secondary_Grade_Point_Average_Result_11',
    'Result_Year_1',
    'Study_Code_1',
    'Non_VCE_Study_Type_1',
    'Study_Score_or_old_Subject_Mark_1',
    'Unit_1_Result_1',
    'Unit_2_Result_1',
    'Unit_3_Result_1',
    'Unit_4_result_Old_Subject_Grade_1',
    'Result_Year_2',
    'Study_Code_2',
    'Non_VCE_Study_Type_2',
    'Study_Score_or_old_Subject_Mark_2',
    'Unit_1_Result_2',
    'Unit_2_Result_2',
    'Unit_3_Result_2',
    'Unit_4_result_Old_Subject_Grade_2',
    'Result_Year_3',
    'Study_Code_3',
    'Non_VCE_Study_Type_3',
    'Study_Score_or_old_Subject_Mark_3',
    'Unit_1_Result_3',
    'Unit_2_Result_3',
    'Unit_3_Result_3',
    'Unit_4_result_Old_Subject_Grade_3',
    'Result_Year_4',
    'Study_Code_4',
    'Non_VCE_Study_Type_4',
    'Study_Score_or_old_Subject_Mark_4',
    'Unit_1_Result_4',
    'Unit_2_Result_4',
    'Unit_3_Result_4',
    'Unit_4_result_Old_Subject_Grade_4',
    'Result_Year_5',
    'Study_Code_5',
    'Non_VCE_Study_Type_5',
    'Study_Score_or_old_Subject_Mark_5',
    'Unit_1_Result_5',
    'Unit_2_Result_5',
    'Unit_3_Result_5',
    'Unit_4_result_Old_Subject_Grade_5',
    'Result_Year_6',
    'Study_Code_6',
    'Non_VCE_Study_Type_6',
    'Study_Score_or_old_Subject_Mark_6',
    'Unit_1_Result_6',
    'Unit_2_Result_6',
    'Unit_3_Result_6',
    'Unit_4_result_Old_Subject_Grade_6',
    'Result_Year_7',
    'Study_Code_7',
    'Non_VCE_Study_Type_7',
    'Study_Score_or_old_Subject_Mark_7',
    'Unit_1_Result_7',
    'Unit_2_Result_7',
    'Unit_3_Result_7',
    'Unit_4_result_Old_Subject_Grade_7',
    'Result_Year_8',
    'Study_Code_8',
    'Non_VCE_Study_Type_8',
    'Study_Score_or_old_Subject_Mark_8',
    'Unit_1_Result_8',
    'Unit_2_Result_8',
    'Unit_3_Result_8',
    'Unit_4_result_Old_Subject_Grade_8',
    'Result_Year_9',
    'Study_Code_9',
    'Non_VCE_Study_Type_9',
    'Study_Score_or_old_Subject_Mark_9',
    'Unit_1_Result_9',
    'Unit_2_Result_9',
    'Unit_3_Result_9',
    'Unit_4_result_Old_Subject_Grade_9',
    'Result_Year_10',
    'Study_Code_10',
    'Non_VCE_Study_Type_10',
    'Study_Score_or_old_Subject_Mark_10',
    'Unit_1_Result_10',
    'Unit_2_Result_10',
    'Unit_3_Result_10',
    'Unit_4_result_Old_Subject_Grade_10',
    'Result_Year_11',
    'Study_Code_11',
    'Non_VCE_Study_Type_11',
    'Study_Score_or_old_Subject_Mark_11',
    'Unit_1_Result_11',
    'Unit_2_Result_11',
    'Unit_3_Result_11',
    'Unit_4_result_Old_Subject_Grade_11',
    'Result_Year_12',
    'Study_Code_12',
    'Non_VCE_Study_Type_12',
    'Study_Score_or_old_Subject_Mark_12',
    'Unit_1_Result_12',
    'Unit_2_Result_12',
    'Unit_3_Result_12',
    'Unit_4_result_Old_Subject_Grade_12',
    'Result_Year_13',
    'Study_Code_13',
    'Non_VCE_Study_Type_13',
    'Study_Score_or_old_Subject_Mark_13',
    'Unit_1_Result_13',
    'Unit_2_Result_13',
    'Unit_3_Result_13',
    'Unit_4_result_Old_Subject_Grade_13',
    'Result_Year_14',
    'Study_Code_14',
    'Non_VCE_Study_Type_14',
    'Study_Score_or_old_Subject_Mark_14',
    'Unit_1_Result_14',
    'Unit_2_Result_14',
    'Unit_3_Result_14',
    'Unit_4_result_Old_Subject_Grade_14',
    'Result_Year_15',
    'Study_Code_15',
    'Non_VCE_Study_Type_15',
    'Study_Score_or_old_Subject_Mark_15',
    'Unit_1_Result_15',
    'Unit_2_Result_15',
    'Unit_3_Result_15',
    'Unit_4_result_Old_Subject_Grade_15',
    'Result_Year_16',
    'Study_Code_16',
    'Non_VCE_Study_Type_16',
    'Study_Score_or_old_Subject_Mark_16',
    'Unit_1_Result_16',
    'Unit_2_Result_16',
    'Unit_3_Result_16',
    'Unit_4_result_Old_Subject_Grade_16',
    'Result_Year_17',
    'Study_Code_17',
    'Non_VCE_Study_Type_17',
    'Study_Score_or_old_Subject_Mark_17',
    'Unit_1_Result_17',
    'Unit_2_Result_17',
    'Unit_3_Result_17',
    'Unit_4_result_Old_Subject_Grade_17',
    'Result_Year_18',
    'Study_Code_18',
    'Non_VCE_Study_Type_18',
    'Study_Score_or_old_Subject_Mark_18',
    'Unit_1_Result_18',
    'Unit_2_Result_18',
    'Unit_3_Result_18',
    'Unit_4_result_Old_Subject_Grade_18',
    'Result_Year_19',
    'Study_Code_19',
    'Non_VCE_Study_Type_19',
    'Study_Score_or_old_Subject_Mark_19',
    'Unit_1_Result_19',
    'Unit_2_Result_19',
    'Unit_3_Result_19',
    'Unit_4_result_Old_Subject_Grade_19',
    'Result_Year_20',
    'Study_Code_20',
    'Non_VCE_Study_Type_20',
    'Study_Score_or_old_Subject_Mark_20',
    'Unit_1_Result_20',
    'Unit_2_Result_20',
    'Unit_3_Result_20',
    'Unit_4_result_Old_Subject_Grade_20',
    'Result_Year_21',
    'Study_Code_21',
    'Non_VCE_Study_Type_21',
    'Study_Score_or_old_Subject_Mark_21',
    'Unit_1_Result_21',
    'Unit_2_Result_21',
    'Unit_3_Result_21',
    'Unit_4_result_Old_Subject_Grade_21',
    'Result_Year_22',
    'Study_Code_22',
    'Non_VCE_Study_Type_22',
    'Study_Score_or_old_Subject_Mark_22',
    'Unit_1_Result_22',
    'Unit_2_Result_22',
    'Unit_3_Result_22',
    'Unit_4_result_Old_Subject_Grade_22',
    'Result_Year_23',
    'Study_Code_23',
    'Non_VCE_Study_Type_23',
    'Study_Score_or_old_Subject_Mark_23',
    'Unit_1_Result_23',
    'Unit_2_Result_23',
    'Unit_3_Result_23',
    'Unit_4_result_Old_Subject_Grade_23',
    'Result_Year_24',
    'Study_Code_24',
    'Non_VCE_Study_Type_24',
    'Study_Score_or_old_Subject_Mark_24',
    'Unit_1_Result_24',
    'Unit_2_Result_24',
    'Unit_3_Result_24',
    'Unit_4_result_Old_Subject_Grade_24',
    'Result_Year_25',
    'Study_Code_25',
    'Non_VCE_Study_Type_25',
    'Study_Score_or_old_Subject_Mark_25',
    'Unit_1_Result_25',
    'Unit_2_Result_25',
    'Unit_3_Result_25',
    'Unit_4_result_Old_Subject_Grade_25',
    'Result_Year_26',
    'Study_Code_26',
    'Non_VCE_Study_Type_26',
    'Study_Score_or_old_Subject_Mark_26',
    'Unit_1_Result_26',
    'Unit_2_Result_26',
    'Unit_3_Result_26',
    'Unit_4_result_Old_Subject_Grade_26',
    'Result_Year_27',
    'Study_Code_27',
    'Non_VCE_Study_Type_27',
    'Study_Score_or_old_Subject_Mark_27',
    'Unit_1_Result_27',
    'Unit_2_Result_27',
    'Unit_3_Result_27',
    'Unit_4_result_Old_Subject_Grade_27',
    'Result_Year_28',
    'Study_Code_28',
    'Non_VCE_Study_Type_28',
    'Study_Score_or_old_Subject_Mark_28',
    'Unit_1_Result_28',
    'Unit_2_Result_28',
    'Unit_3_Result_28',
    'Unit_4_result_Old_Subject_Grade_28',
    'Result_Year_29',
    'Study_Code_29',
    'Non_VCE_Study_Type_29',
    'Study_Score_or_old_Subject_Mark_29',
    'Unit_1_Result_29',
    'Unit_2_Result_29',
    'Unit_3_Result_29',
    'Unit_4_result_Old_Subject_Grade_29',
    'Result_Year_30',
    'Study_Code_30',
    'Non_VCE_Study_Type_30',
    'Study_Score_or_old_Subject_Mark_30',
    'Unit_1_Result_30',
    'Unit_2_Result_30',
    'Unit_3_Result_30',
    'Unit_4_result_Old_Subject_Grade_30',
    'Result_Year_31',
    'Study_Code_31',
    'Non_VCE_Study_Type_31',
    'Study_Score_or_old_Subject_Mark_31',
    'Unit_1_Result_31',
    'Unit_2_Result_31',
    'Unit_3_Result_31',
    'Unit_4_result_Old_Subject_Grade_31',
    'Result_Year_32',
    'Study_Code_32',
    'Non_VCE_Study_Type_32',
    'Study_Score_or_old_Subject_Mark_32',
    'Unit_1_Result_32',
    'Unit_2_Result_32',
    'Unit_3_Result_32',
    'Unit_4_result_Old_Subject_Grade_32',
    'Result_Year_33',
    'Study_Code_33',
    'Non_VCE_Study_Type_33',
    'Study_Score_or_old_Subject_Mark_33',
    'Unit_1_Result_33',
    'Unit_2_Result_33',
    'Unit_3_Result_33',
    'Unit_4_result_Old_Subject_Grade_33',
    'Result_Year_34',
    'Study_Code_34',
    'Non_VCE_Study_Type_34',
    'Study_Score_or_old_Subject_Mark_34',
    'Unit_1_Result_34',
    'Unit_2_Result_34',
    'Unit_3_Result_34',
    'Unit_4_result_Old_Subject_Grade_34',
    'Result_Year_35',
    'Study_Code_35',
    'Non_VCE_Study_Type_35',
    'Study_Score_or_old_Subject_Mark_35',
    'Unit_1_Result_35',
    'Unit_2_Result_35',
    'Unit_3_Result_35',
    'Unit_4_result_Old_Subject_Grade_35',
    'Study_Score_decoded_1',
    'Study_Score_decoded_2',
    'Study_Score_decoded_3',
    'Study_Score_decoded_4',
    'Study_Score_decoded_5',
    'Study_Score_decoded_6',
    'Study_Score_decoded_7',
    'Study_Score_decoded_8',
    'Study_Score_decoded_9',
    'Study_Score_decoded_10',
    'Study_Score_decoded_11',
    'Study_Score_decoded_12',
    'Study_Score_decoded_13',
    'Study_Score_decoded_14',
    'Study_Score_decoded_15',
    'Study_Score_decoded_16',
    'Study_Score_decoded_17',
    'Study_Score_decoded_18',
    'Study_Score_decoded_19',
    'Study_Score_decoded_20',
    'Study_Score_decoded_21',
    'Study_Score_decoded_22',
    'Study_Score_decoded_23',
    'Study_Score_decoded_24',
    'Study_Score_decoded_25',
    'Study_Score_decoded_26',
    'Study_Score_decoded_27',
    'Study_Score_decoded_28',
    'Study_Score_decoded_29',
    'Study_Score_decoded_30',
    'Study_Score_decoded_31',
    'Study_Score_decoded_32',
    'Study_Score_decoded_33',
    'Study_Score_decoded_34',
    'Study_Score_decoded_35',
]
further_prune = [
    'Prerequisite_met__1',
    'Offer_Round_1',
    'Enrolment_Status_1',
    'Fee_eligibility_for_Type_4_enrolled_course_1',
    'Prerequisite_met__2',
    'Offer_Round_2',
    'Offer_Round__Offer_2',
    'Enrolment_Status_2',
    'Fee_eligibility_for_Type_4_enrolled_course_2',
    'Prerequisite_met__3',
    'Offer_Round_3',
    'Enrolment_Status_3',
    'Fee_eligibility_for_Type_4_enrolled_course_3',
    'Prerequisite_met__4',
    'Offer_Round_4',
    'Enrolment_Status_4',
    'Fee_eligibility_for_Type_4_enrolled_course_4',
    'Prerequisite_met__5',
    'Offer_Round_5',
    'Enrolment_Status_5',
    'Fee_eligibility_for_Type_4_enrolled_course_5',
    'Prerequisite_met__6',
    'Offer_Round_6',
    'Enrolment_Status_6',
    'Fee_eligibility_for_Type_4_enrolled_course_6',
    'Prerequisite_met__7',
    'Offer_Round_7',
    'Enrolment_Status_7',
    'Fee_eligibility_for_Type_4_enrolled_course_7',
    'Prerequisite_met__8',
    'Offer_Round_8',
    'Enrolment_Status_8',
    'Fee_eligibility_for_Type_4_enrolled_course_8',
    'ATAR_Calculating_Authority',
    'STAT_Type',
    'STAT_Percentile',
    'UMAT_or_ISAT_type',
    'VTAC_personal_history_submission',
    'SEAS_application__Special_Entry_Access_Schemes_',
    'Educational_attainment_of_second_parent_guardian',
    'Hearing_Problem',
    'Learning_Problem',
    'Medical_Problem',
    'Mobility_Problem',
    'Vision_Problem',
    'Other_Problem',
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


def fea_prune(data_fn, out_fn, further=False):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
    # print('\',\n    \''.join(data.keys()))

    if further:
        data.drop(columns=further_prune, errors='ignore', inplace=True)
    else:
        data.drop(columns=to_prune, errors='ignore', inplace=True)

    # write the modified data to csv
    data.to_csv(os.path.join(data_dir, out_fn), index=False)
    return data


def observe(data_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))
    print('\n'.join(data.keys()))

    # print(data.groupby('ASGS_rating_of_residential_Australian_postcode', as_index=False).size())


years = [2016, 2017]

def split(data_fn):
    dt = pd.read_csv(os.path.join(data_dir, dt_fn), index_col='column')
    data = pd.read_csv(os.path.join(data_dir, data_fn), dtype=dict(dt['dtype']))

    for y in years:
        data_year = data[data['Year_of_application'] == y]

        # # write the modified data to csv
        # data_year.to_csv(os.path.join(data_dir, data_fn[:-10] + str(y) + '.csv'), index=False)

    return


if __name__ == '__main__':
    # data_2016 = csv_cleaning('ReformattedFile_2016.csv', 'UnifiedFile_2016.csv')
    # data_2017 = csv_cleaning('ReformattedFile_2017.csv', 'UnifiedFile_2017.csv')
    # data_2018 = csv_cleaning('ReformattedFile_2018.csv', 'UnifiedFile_2018.csv')
    # data = pd.concat([data_2016, data_2017], ignore_index=True)
    # data.to_csv(os.path.join(data_dir, 'UnifiedFile_merged.csv'), index=False)


    # data = fea_prep(
    #     'UnifiedFile_2018.csv',
    #     'UnifiedFile_2018.csv',
    # )

    # data = fea_eng(
    #     'UnifiedFile_2018.csv',
    #     'EngineeredFile_2018.csv',
    #     'CourseCode_FoS_Inst.csv'
    # )

    # data = fea_prune(
    #     'EngineeredFile_2018.csv',
    #     'ShortFile_2018.csv',
    # )
    #
    # data = fea_prune(
    #     'ShortFile_2018.csv',
    #     'MinimalFile_2018.csv',
    #     further=True
    # )

    observe('MinimalFile_2018.csv')
    # observe('MinimalFile_merged.csv')

    # split('ShortFile_merged.csv')
    pass


