import copy
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import time


def print_results(clf_names, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a,
                  recall_a, DI_a, SPD_a, running_time=0):
    for clf_name in clf_names:
        print(f'\n\n\tRESULT FROM: {clf_name} model:')
        print('---Original---')
        print('Aod before:', np.mean(np.abs(aod_b[clf_name])))
        print('Eod before:', np.mean(np.abs(eod_b[clf_name])))
        print('Acc before:', np.mean(acc_b[clf_name]))
        print('Far before:', np.mean(false_b[clf_name]))
        print('recall before:', np.mean(recall_b[clf_name]))
        print('DI before:', np.mean(DI_b[clf_name]))
        print('SPD before:', np.mean(np.abs(SPD_b[clf_name])))

        print('---LTDD---')
        print('Aod after:', np.mean(np.abs(aod_a[clf_name])))
        print('Eod after:', np.mean(np.abs(eod_a[clf_name])))
        print('Acc after:', np.mean(acc_a[clf_name]))
        print('Far after:', np.mean(false_a[clf_name]))
        print('recall after:', np.mean(recall_a[clf_name]))
        print('DI after:', np.mean(DI_a[clf_name]))
        print('SPD after:', np.mean(np.abs(SPD_a[clf_name])))
    print('Average running time:', np.mean(running_time))


def run_original_LTDD(dataset_orig, current_protected_attr, other_protected_attrs, categorical_columns,random_state = None):
    np.random.seed(13)
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
    running_time = []
    ce_times = []

    iterations = 100
    for k in range(iterations):
        start_time = time.time()
        print('------the {}\{} th turn------'.format(k, iterations))

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.15, random_state=random_state,
                                                                 shuffle=True)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train[
            'Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test[
            'Probability']

        column_train = [column for column in X_train]

        clfs = {'logistic': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100),
                'random_forest': RandomForestClassifier(random_state=0, max_depth=3),
                'boost': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0),
                'decision_tree': DecisionTreeClassifier(max_depth=3)}

        from aif360.datasets import BinaryLabelDataset, StructuredDataset
        from aif360.algorithms.preprocessing import Reweighing
        from aif360.metrics import ClassificationMetric
        from aif360.metrics import BinaryLabelDatasetMetric

        dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                       unfavorable_label=0.0,
                                       df=dataset_orig_test,
                                       label_names=['Probability'],
                                       protected_attribute_names=[current_protected_attr],
                                       )
        y_preds = []
        for clf_name, clf_i in clfs.items():
            clf_i.fit(X_train, y_train)
            y_pred = clf_i.predict(X_test)

            dataset_pred = dataset_t.copy()
            dataset_pred.labels = y_pred
            attr = dataset_t.protected_attribute_names[0]
            idx = dataset_t.protected_attribute_names.index(attr)
            privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
            unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

            class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
            b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

            DI_b[clf_name].append(class_metrics.disparate_impact())
            SPD_b[clf_name].append(class_metrics.statistical_parity_difference())

            acc_b[clf_name].append(class_metrics.accuracy())
            recall_b[clf_name].append(class_metrics.recall())
            false_b[clf_name].append(class_metrics.false_positive_rate())
            aod_b[clf_name].append(class_metrics.average_odds_difference())
            eod_b[clf_name].append(class_metrics.equal_opportunity_difference())

        slope_store = []
        intercept_store = []
        rvalue_store = []
        pvalue_store = []
        column_u = []
        flag = 0
        ce = []
        times = 0

        def Linear_regression(x, slope, intercept):
            return x * slope + intercept

        current_running_time = 0
        for i in column_train:
            flag = flag + 1
            if i != current_protected_attr and i not in categorical_columns:
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(X_train[current_protected_attr], X_train[i])
                rvalue_store.append(rvalue)
                pvalue_store.append(pvalue)
                if i not in other_protected_attrs:
                    if pvalue < 0.05:
                        times = times + 1
                        column_u.append(i)
                        ce.append(flag)
                        slope_store.append(slope)
                        intercept_store.append(intercept)
                        X_train.loc[:, i] = X_train.loc[:, i] - Linear_regression(X_train[current_protected_attr],
                                                                                  slope, intercept)

        ce_times.append(times)
        ce_list.append(ce)

        X_train = X_train.drop([current_protected_attr], axis=1)

        for i in range(len(column_u)):
            X_test.loc[:, column_u[i]] = X_test.loc[:, column_u[i]] - Linear_regression(X_test[current_protected_attr],
                                                                                        slope_store[i],
                                                                                        intercept_store[i])

        X_test = X_test.drop([current_protected_attr], axis=1)

        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
        from aif360.metrics import BinaryLabelDatasetMetric

        dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                       unfavorable_label=0.0,
                                       df=dataset_orig_test,
                                       label_names=['Probability'],
                                       protected_attribute_names=[current_protected_attr],
                                       )
        for clf_name, clf_i in clfs.items():
            clf_i.fit(X_train, y_train)
            y_pred = clf_i.predict(X_test)
            dataset_pred = dataset_t.copy()
            dataset_pred.labels = y_pred
            attr = dataset_t.protected_attribute_names[0]
            idx = dataset_t.protected_attribute_names.index(attr)
            privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
            unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

            class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
            b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

            DI_a[clf_name].append(class_metrics.disparate_impact())
            SPD_a[clf_name].append(class_metrics.statistical_parity_difference())

            acc_a[clf_name].append(class_metrics.accuracy())
            recall_a[clf_name].append(class_metrics.recall())
            false_a[clf_name].append(class_metrics.false_positive_rate())
            aod_a[clf_name].append(class_metrics.average_odds_difference())
            eod_a[clf_name].append(class_metrics.equal_opportunity_difference())
        current_running_time += time.time() - start_time

    return clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time

def run_sequential_LTDD(dataset_orig, current_protected_attr, other_protected_attrs, is_sequencetial_run):
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
    running_time = []
    ce_times = []

    iterations = 100
    index = 1
    while(index <= iterations):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try: 
                for k in range(iterations):
                    start_time = time.time()

                    print('------the {}\{} th turn------'.format(k, iterations))

                    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.15, random_state=42,
                                                                            shuffle=True)

                    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train[
                        'Probability']
                    X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test[
                        'Probability']

                    column_train = [column for column in X_train]

                    clfs = {'logistic': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100),
                            'random_forest': RandomForestClassifier(random_state=0, max_depth=3),
                            'boost': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0),
                            'decision_tree': DecisionTreeClassifier(max_depth=3)}

                    from aif360.datasets import BinaryLabelDataset, StructuredDataset
                    from aif360.algorithms.preprocessing import Reweighing
                    from aif360.metrics import ClassificationMetric
                    from aif360.metrics import BinaryLabelDatasetMetric

                    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                                unfavorable_label=0.0,
                                                df=dataset_orig_test,
                                                label_names=['Probability'],
                                                protected_attribute_names=[current_protected_attr],
                                                )
                    y_preds = []
                    for clf_name, clf_i in clfs.items():
                        clf_i.fit(X_train, y_train)
                        y_pred = clf_i.predict(X_test)

                        dataset_pred = dataset_t.copy()
                        dataset_pred.labels = y_pred
                        attr = dataset_t.protected_attribute_names[0]
                        idx = dataset_t.protected_attribute_names.index(attr)
                        privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
                        unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

                        class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
                        b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

                        DI_b[clf_name].append(class_metrics.disparate_impact())
                        SPD_b[clf_name].append(class_metrics.statistical_parity_difference())

                        acc_b[clf_name].append(class_metrics.accuracy())
                        recall_b[clf_name].append(class_metrics.recall())
                        false_b[clf_name].append(class_metrics.false_positive_rate())
                        aod_b[clf_name].append(class_metrics.average_odds_difference())
                        eod_b[clf_name].append(class_metrics.equal_opportunity_difference())

                    slope_store = []
                    intercept_store = []
                    rvalue_store = []
                    pvalue_store = []
                    column_u = []
                    flag = 0
                    ce = []

                    def Linear_regression(x, slope, intercept):
                        return x * slope + intercept

                    current_running_time = 0
                    for i in column_train:
                        if i != current_protected_attr:
                            current_start_time = time.time()

                            slope, intercept, rvalue, pvalue, stderr = stats.linregress(X_train[current_protected_attr],
                                                                                        X_train[i])

                            rvalue_store.append(rvalue)
                            pvalue_store.append(pvalue)
                            if i not in other_protected_attrs:
                                if pvalue < 0.05:
                                    column_u.append(i)
                                    ce.append(flag)
                                    slope_store.append(slope)
                                    intercept_store.append(intercept)
                                    X_train.loc[:, i] = X_train.loc[:, i] - Linear_regression(X_train[current_protected_attr],
                                                                                            slope, intercept)

                    for i in range(len(column_u)):
                        X_test.loc[:, column_u[i]] = X_test.loc[:, column_u[i]] - Linear_regression(
                            X_test[current_protected_attr],
                            slope_store[i],
                            intercept_store[i])

                    from aif360.datasets import BinaryLabelDataset
                    from aif360.metrics import ClassificationMetric
                    from aif360.metrics import BinaryLabelDatasetMetric

                    dataset_t = BinaryLabelDataset(favorable_label=1.0,
                                                unfavorable_label=0.0,
                                                df=dataset_orig_test,
                                                label_names=['Probability'],
                                                protected_attribute_names=[current_protected_attr],
                                                )

                    for clf_name, clf_i in clfs.items():
                        clf_i.fit(X_train, y_train)
                        y_pred = clf_i.predict(X_test)
                        dataset_pred = dataset_t.copy()
                        dataset_pred.labels = y_pred
                        attr = dataset_t.protected_attribute_names[0]
                        idx = dataset_t.protected_attribute_names.index(attr)
                        privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
                        unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

                        class_metrics = ClassificationMetric(dataset_t, dataset_pred, unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
                        b_metrics = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
                        
                        DI_a[clf_name].append(class_metrics.disparate_impact())
                        SPD_a[clf_name].append(class_metrics.statistical_parity_difference())

                        acc_a[clf_name].append(class_metrics.accuracy())
                        recall_a[clf_name].append(class_metrics.recall())
                        false_a[clf_name].append(class_metrics.false_positive_rate())
                        aod_a[clf_name].append(class_metrics.average_odds_difference())
                        eod_a[clf_name].append(class_metrics.equal_opportunity_difference())
                    current_running_time += time.time() - start_time
                    index = index + 1
            except RuntimeWarning as e:
                continue
    return clfs, aod_b, eod_b, acc_b, false_b, recall_b, DI_b, SPD_b, aod_a, eod_a, acc_a, false_a, recall_a, DI_a, SPD_a, running_time