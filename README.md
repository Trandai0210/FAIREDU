# FAIREDU: A Multiple Regression-Based Method for Enhancing Fairness in Machine Learning Models for Educational Applications

This is the official implementation and DNU dataset of the paper: “FAIREDU: A Multiple Regression-Based Method for Enhancing Fairness in Machine Learning Models for Educational Applications".

<br/>

## Requirement
* Python >= 3.9
* Aif360 == 0.6.1

## Datasets
 ### DNU Dataset: We publish the DNU dataset [here](./datasets/DNU.csv) 
 ### Additional datasets:

1. Adult - http://archive.ics.uci.edu/ml/datasets/Adult

2. COMPAS - https://github.com/propublica/compas-analysis

3. Default - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

4. Student Dropout - https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

5. Student Performance - https://archive.ics.uci.edu/dataset/320

6. OULAD - https://www.nature.com/articles/sdata2017171


<br/>

## Implementation

We provide the source code for reproduction in the folder ./FAIREDU.
<br/>

## Experimental settings
* Multiple datasets with sensitive attributes (e.g., gender, race, age) were used, focusing on fairness challenges in the educational domain.
* Preprocessing we remove Association in Training set and Test set.
* The model and baselines were implemented using Python with frameworks like sklearn and aif360.
* Performance metrics: Accuracy and F1-score to assess the trade-off between fairness and accuracy.
* All datasets, and preprocessing steps are provided in this repository to ensure full reproducibility.
* The code is structured to allow easy extension to other domains or datasets.

## Acknowledgement

[LTDD](https://github.com/fairnesstest/LTDD) The codebase and model we built upon.

## Citation
If you find our paper and code useful in your research, please consider citation:

```BibTeX
@article{pham2024fairedu,
      title={FAIREDU: A Multiple Regression-Based Method for Enhancing Fairness in Machine Learning Models for Educational Applications}, 
      author={Nga Pham and Minh Kha Do and Tran Vu Dai and Pham Ngoc Hung and Anh Nguyen-Duc},
      year={2024},
      eprint={2410.06423},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.06423}, 
}
```
