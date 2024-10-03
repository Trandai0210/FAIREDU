import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics

import sys
sys.path.append(os.path.abspath('..'))

from scipy import stats
from utils import print_results, run_original_LTDD

np.random.seed(13)

## Load dataset
from sklearn import preprocessing
dataset_orig = pd.read_csv('../datasets/student_dropout.csv')

## Drop NULL values
dataset_orig = dataset_orig.dropna()
dataset_orig = dataset_orig.drop(columns=['id_student'])

## Make goal column binary

dataset_orig['Probability'] = np.where(dataset_orig['Target'] == 'Graduate', 1, 0)
dataset_orig = dataset_orig.drop(['Target'],axis=1)

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

protected_attrs = ['Debtor', 'Gender']

for i in protected_attrs:
    print(f"###### RESULT FROM {i} attribute #####")

    clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time = run_original_LTDD(
        dataset_orig, current_protected_attr=i, other_protected_attrs=protected_attrs, categorical_columns=[])

    print_results(clfs.keys(), aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a,
                  recall_a,
                  DI_a, SPD_a,running_time)
    print("#######################################")