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
dataset_orig = pd.read_csv('../datasets/student_performance.csv')


dataset_orig = dataset_orig.drop(['school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()

## calculate mean of age column
mean = dataset_orig.loc[:,"age"].mean()

dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'M', 1, 0)

dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
dataset_orig['schoolsup'] = np.where(dataset_orig['schoolsup'] == 'yes', 1, 0)
dataset_orig['famsup'] = np.where(dataset_orig['famsup'] == 'yes', 1, 0)
dataset_orig['paid'] = np.where(dataset_orig['paid'] == 'yes', 1, 0)
dataset_orig['activities'] = np.where(dataset_orig['activities'] == 'yes', 1, 0)
dataset_orig['nursery'] = np.where(dataset_orig['nursery'] == 'yes', 1, 0)
dataset_orig['higher'] = np.where(dataset_orig['higher'] == 'yes', 1, 0)
dataset_orig['internet'] = np.where(dataset_orig['internet'] == 'yes', 1, 0)
dataset_orig['romantic'] = np.where(dataset_orig['romantic'] == 'yes', 1, 0)
dataset_orig['health'] = np.where(dataset_orig['health'] >= 4, 1, 0)

mean = dataset_orig.loc[:,"G1"].mean()
dataset_orig['G1'] = np.where(dataset_orig['G1'] >= mean, 1, 0)

mean = dataset_orig.loc[:,"G2"].mean()
dataset_orig['G2'] = np.where(dataset_orig['G2'] >= mean, 1, 0)

## Make goal column binary
mean = dataset_orig.loc[:,"Probability"].mean()
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] >= mean, 1, 0)

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

protected_attrs = ['sex', 'health']


print(f"###### RESULT FROM sex-health attribute #####")

clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time = run_original_LTDD(
    dataset_orig, current_protected_attr='sex', other_protected_attrs=['sex', 'health'], categorical_columns=[])

print_results(clfs.keys(), aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a,
              recall_a,
              DI_a, SPD_a, running_time)
print("#######################################\n")


print(f"###### RESULT FROM health-sex attribute #####")

clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time = run_original_LTDD(
    dataset_orig, current_protected_attr='health', other_protected_attrs=['health', 'sex'], categorical_columns=[])

print_results(clfs.keys(), aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a,
              recall_a,
              DI_a, SPD_a, running_time)
print("#######################################")