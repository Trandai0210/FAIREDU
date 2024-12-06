# FAIREDU: A Multiple Regression-Based Method for Enhancing Fairness in Machine Learning Models for Educational Applications

This repository stores our experimental codes for the paper “FAIREDU: A Multiple Regression-Based Method for Enhancing Fairness in Machine Learning Models for Educational Applications”，FAIREDU is the name of the method we propose in this paper.

<br/>

## Datasets

1> Adult - http://archive.ics.uci.edu/ml/datasets/Adult

2> COMPAS - https://github.com/propublica/compas-analysis

3> Default - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

4> Student Dropout - https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

5> Student Performance - https://archive.ics.uci.edu/dataset/320

6> OULAD - https://www.nature.com/articles/sdata2017171

7> DNU - The dataset in the "datasets" folder.

<br/>

## Codes for FAIREDU

You can easily reproduce our method, we provide it in the FAIREDU folder. 

The codes in the folder are named for the applicable scenarios.

The code contains data preprocessing, our method and the calculation of indicators. You can run these codes directly to get the experimental results.

<br/>

## Baseline methods

We compare our method with LTDD method:

LTDD: Linear-regression based Training Data Debugging.

We use the code they provided in the code repository: https://github.com/fairnesstest/LTDD

## Experimental settings
* Multiple datasets with sensitive attributes (e.g., gender, race, age) were used, focusing on fairness challenges in the educational domain.
* Preprocessing we remove Association in Training set and Test set.
* The model and baselines were implemented using Python with frameworks like sklearn and aif360.
* Performance metrics: Accuracy and F1-score to assess the trade-off between fairness and accuracy.
* All datasets, and preprocessing steps are provided in this repository to ensure full reproducibility.
* The code is structured to allow easy extension to other domains or datasets.
