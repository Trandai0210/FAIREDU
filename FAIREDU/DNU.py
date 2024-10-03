import pandas as pd

import math, copy, os
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import sys

sys.path.append(os.path.abspath('..'))
from utils import *

np.random.seed(13)

## Load dataset DNU
dataset_orig = pd.read_csv('../datasets/DNU.csv')

dataset_orig = dataset_orig.drop(['adv_score'], axis=1)
## Change Column values
dataset_orig['gender'] = np.where(dataset_orig['gender'] == 'Male', 1, 0)

categorical_columns = []
numeric_columns = []
label_column = 'Probability'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

aod_b = {'logistic': [],
         'random_forest': [],
         'boost': [],
         'decision_tree': []}
eod_b = copy.deepcopy(aod_b)
aod_a = copy.deepcopy(aod_b)
eod_a = copy.deepcopy(aod_b)
acc_a = copy.deepcopy(aod_b)
acc_b = copy.deepcopy(aod_b)
ce_list = []
recall_a = copy.deepcopy(aod_b)
recall_b = copy.deepcopy(aod_b)
false_a = copy.deepcopy(aod_b)
false_b = copy.deepcopy(aod_b)
DI_a = copy.deepcopy(aod_b)
SPD_a = copy.deepcopy(aod_b)
DI_b = copy.deepcopy(aod_b)
SPD_b = copy.deepcopy(aod_b)
ce_times = []

protected_attrs = ['gender','age','birthplace']

for i in protected_attrs:
    print(f"###### RESULT FROM {i} attribute #####")

    clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time = run_original_LTDD(
        dataset_orig, current_protected_attr=i, other_protected_attrs=protected_attrs, categorical_columns=[], random_state=42)

    print_results(clfs.keys(), aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a,
                  recall_a,
                  DI_a, SPD_a,running_time)
    print("#######################################")