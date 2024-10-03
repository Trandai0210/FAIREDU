import pandas as pd

import math, copy, os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import sys

sys.path.append(os.path.abspath('..'))
from utils import *

np.random.seed(13)

## Load dataset OULAD
dataset_orig = pd.read_csv('../datasets/student_oulad.csv')

## Drop NULL values
dataset_orig = dataset_orig.dropna(axis=0, how='any')
dataset_orig = dataset_orig.drop(['STT','Mã sinh viên','Họ và tên','Tên lớp'])

## Change symbolics to numericsk
dataset_orig['gender'] = np.where(dataset_orig['gender'] == 'F', 0, 1)
dataset_orig['disability'] = np.where(dataset_orig['disability'] == 'Y', 0, 1)

dataset_orig['final_result'] = dataset_orig['final_result'].apply(lambda x: 0 if x in ['Fail', 'Withdrawn'] else 1)

categorical_columns = ['code_module', 'code_presentation', 'region', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts']
numeric_column = 'studied_credits'
label_column = 'final_result'

# Initialize the LabelEncoder
label_encoders = {col: LabelEncoder() for col in categorical_columns}

# Apply label encoding to each column
for col in categorical_columns:
    dataset_orig[col] = label_encoders[col].fit_transform(dataset_orig[col])

dataset_orig = dataset_orig.drop(columns=['num_of_prev_attempts', 'code_module', 'code_presentation', 'highest_education'])

# Normalize the numeric column
scaler = StandardScaler()
dataset_orig[numeric_column] = scaler.fit_transform(dataset_orig[[numeric_column]])

dataset_orig = dataset_orig.rename(columns={'final_result': 'Probability'})

protected_attrs = ['gender', 'disability']

dataset_orig = dataset_orig.astype(float)

for i in protected_attrs:
    print(f"###### RESULT FROM {i} attribute #####")

    clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time = run_original_LTDD(
        dataset_orig, current_protected_attr=i, other_protected_attrs=protected_attrs, categorical_columns=[])

    print_results(clfs.keys(), aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a,
                  recall_a,
                  DI_a, SPD_a,running_time)
    print("#######################################")
