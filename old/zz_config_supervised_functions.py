"""

this first part is developed to support the models developed a posteriori.
cv_gs_results_func will export the results from the cross-validation made by the GridSearchCV.
the plot_confusion_matrix will plot which confusion matrix
the plot_roc_roc will make a roc_curve for each plot where the test_accuracy was higher or equal to 60% to make sure
this doensn't export a lot of plots! the plots are saved on the roc_plot_path path_variable (global)
the results_func will export the values of the accuracy of the models and will send the values for the results variable
from the main script

"""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score,
                             f1_score, confusion_matrix, recall_score,
                             precision_score, accuracy_score)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from xgboost import XGBClassifier
import numpy as np
import os
from zz_config_module import random_seed, stratified_cv, func_cv_results, plot_confusion_matrix, plot_roc_curve, results_func
from zz_config_features_importance import func_back_projection

roc_plot_path = '000_ROC_plots/'
confusion_matrix_plot_path = '000_CM_plots/'
os.makedirs(roc_plot_path, exist_ok=True)
os.makedirs(confusion_matrix_plot_path, exist_ok=True)


def random_forest(x_train, y_train, x_test, y_test, label_encoder,
                  sample_type, train_percentage, loadings, wavenumbers,
                  results, cross_validation_results, target_column,
                  back_projection, group_fam_to_use=None):
    model_name = "Random Forest"

    # GridSearchCV
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [4, 8, 12],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=random_seed),
                               param_grid_rf, scoring="balanced_accuracy", cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    test_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)

    lv_importance = best_model.feature_importances_
    top_wavenumbers, top_importances = func_back_projection(
        lv_importance, loadings, wavenumbers, test_acc, target_column,
        sample_type, train_percentage, f"{model_name} (GridSearchCV)",
        group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(
        cross_validation_results, sample_type, train_percentage, test_acc,
        f1, recall, precision, cm, grid_search, acc, f"{model_name} (GridSearchCV)"
    )

    plot_confusion_matrix(y_test, y_pred, label_encoder, test_acc, sample_type, train_percentage,
                          f"{model_name} (GridSearchCV)", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob, label_encoder, sample_type, train_percentage,
                             test_acc, f"{model_name} (GridSearchCV)", target_column, group_fam_to_use=group_fam_to_use)
    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"{model_name} (GridSearchCV)", test_acc, f1, roc_auc,
                                            top_wavenumbers, top_importances, back_projection)

    # Bayesian Optimization
    bayes_search = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=random_seed),
        search_spaces={
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 15),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'bootstrap': Categorical([True, False]),
        },
        n_iter=50,
        n_points=10,
        cv=stratified_cv,
        scoring="balanced_accuracy",
        random_state=random_seed
    )

    bayes_search.fit(x_train, y_train)
    best_bayes = bayes_search.best_estimator_

    y_pred = best_bayes.predict(x_test)
    y_prob = best_bayes.predict_proba(x_test)

    test_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)

    lv_importance = best_bayes.feature_importances_
    top_wavenumbers, top_importances = func_back_projection(
        lv_importance, loadings, wavenumbers, test_acc, target_column,
        sample_type, train_percentage, f"{model_name} (BayesOpt)", group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(
        cross_validation_results, sample_type, train_percentage, test_acc,
        f1, recall, precision, cm, bayes_search, acc, f"{model_name} (BayesOpt)")

    plot_confusion_matrix(y_test, y_pred, label_encoder, test_acc, sample_type, train_percentage,
                          f"{model_name} (BayesOpt)", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob, label_encoder, sample_type, train_percentage,
                             test_acc, f"{model_name} (BayesOpt)", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"{model_name} (BayesOpt)", test_acc, f1, roc_auc,
                                            top_wavenumbers, top_importances, back_projection)

    return results, cross_validation_results, back_projection


def mlp_classifier(x_train, y_train, x_test, y_test, label_encoder,
                   sample_type, train_percentage, loadings, wavenumbers,
                   results, cross_validation_results, target_column, back_projection,
                   group_fam_to_use=None):

    suffix = f"_{group_fam_to_use}" if group_fam_to_use else ""

    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }

    grid_search = GridSearchCV(MLPClassifier(max_iter=3000,
                                             random_state=random_seed,
                                             learning_rate_init=0.01,
                                             early_stopping=True,
                                             validation_fraction=0.1),
                                param_grid_mlp, cv=5, scoring='balanced_accuracy')
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score (y_test, y_pred)

    perm_importance = permutation_importance(best_model, x_test, y_test, n_repeats=10, random_state=random_seed)
    lv_importance = perm_importance.importances_mean

    top_wavenumbers, top_importances = func_back_projection(lv_importance, loadings, wavenumbers,
                                                             acc, target_column, sample_type, train_percentage,
                                                             f"MLP (Best Grid Search){suffix}")

    cross_validation_results = func_cv_results(cross_validation_results, sample_type, train_percentage,
                                               acc, f1, recall, precision, conf_matrix,
                                               grid_search,acc_score, f"MLP (Best Grid Search){suffix}")

    plot_confusion_matrix(y_test, y_pred, label_encoder, acc, sample_type, train_percentage,
                          f"MLP (Best Grid Search){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob, label_encoder, sample_type, train_percentage,
                              acc, f"MLP (Best Grid Search){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"MLP (Best Grid Search){suffix}", acc, f1,
                                            roc_auc, top_wavenumbers, top_importances, back_projection)

    # BAYESIAN OPTIMIZATION
    bayes_search = BayesSearchCV(
        MLPClassifier(random_state=random_seed, validation_fraction=0.1,
                      hidden_layer_sizes=(200,), early_stopping=True),
        {
            'activation': Categorical(['relu', 'tanh', 'logistic']),
            'solver': Categorical(['adam', 'sgd']),
            'alpha': Real(1e-6, 1e-1, prior='log-uniform'),
            'learning_rate_init': Real(1e-4, 1e-2, prior='log-uniform'),
            'max_iter': Integer(500, 3000)
        },
        n_iter=50, n_points=10, cv=stratified_cv,
        scoring='balanced_accuracy', random_state=random_seed
    )
    bayes_search.fit(x_train, y_train)
    best_model_bayes = bayes_search.best_estimator_
    y_pred_bayes = best_model_bayes.predict(x_test)
    y_prob_bayes = best_model_bayes.predict_proba(x_test)

    acc_bayes = balanced_accuracy_score(y_test, y_pred_bayes)
    f1_bayes = f1_score(y_test, y_pred_bayes, average='weighted')
    recall_bayes = recall_score(y_test, y_pred_bayes, average='weighted')
    precision_bayes = precision_score(y_test, y_pred_bayes, average='weighted')
    conf_matrix_bayes = confusion_matrix(y_test, y_pred_bayes)
    acc_score = accuracy_score (y_test, y_pred_bayes)

    perm_bayes = permutation_importance(best_model_bayes, x_test, y_test, random_state=random_seed)
    lv_importance = perm_bayes.importances_mean
    top_wavenumbers, top_importances = func_back_projection(lv_importance, loadings, wavenumbers,
                                                             acc_bayes, target_column, sample_type,
                                                             train_percentage, f"MLP (Bayesian Optimization){suffix}", group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(cross_validation_results, sample_type, train_percentage,
                                               acc_bayes, f1_bayes, recall_bayes, precision_bayes, conf_matrix_bayes,
                                               bayes_search,acc_score, f"MLP (Bayesian Optimization){suffix}")

    plot_confusion_matrix(y_test, y_pred_bayes, label_encoder, acc_bayes, sample_type, train_percentage,
                          f"MLP (Bayesian Optimization){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob_bayes, label_encoder, sample_type, train_percentage,
                              acc_bayes, f"MLP (Bayesian Optimization){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"MLP (Bayesian Optimization){suffix}", acc_bayes, f1_bayes,
                                            roc_auc, top_wavenumbers, top_importances, back_projection)

    return results, cross_validation_results, back_projection


def decision_tree(x_train, y_train, x_test, y_test, label_encoder,
                  sample_type, train_percentage, loadings, wavenumbers,
                  results, cross_validation_results, target_column, back_projection,
                  group_fam_to_use=None):

    suffix = f"_{group_fam_to_use}" if group_fam_to_use else ""

    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=random_seed),
                               param_grid_dt, cv=5, scoring='balanced_accuracy')
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score (y_test, y_pred)

    lv_importance = best_model.feature_importances_
    top_wavenumbers, top_importances = func_back_projection(lv_importance, loadings, wavenumbers,
                                                             acc, target_column, sample_type, train_percentage,
                                                             f"Decision Tree (Best Grid Search){suffix}", group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(cross_validation_results, sample_type, train_percentage,
                                               acc, f1, recall, precision, conf_matrix,
                                               grid_search, acc_score, f"Decision Tree (Best Grid Search){suffix}")

    plot_confusion_matrix(y_test, y_pred, label_encoder, acc, sample_type, train_percentage,
                          f"Decision Tree (Best Grid Search){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob, label_encoder, sample_type, train_percentage,
                              acc, f"Decision Tree (Best Grid Search){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"Decision Tree (Best Grid Search){suffix}", acc, f1,
                                            roc_auc, top_wavenumbers, top_importances, back_projection)

    # BAYESIAN OPTIMIZATION
    bayes_search = BayesSearchCV(
        DecisionTreeClassifier(random_state=random_seed),
        {
            'max_depth': Integer(1, 100),
            'min_samples_split': Integer(2, 50),
            'min_samples_leaf': Integer(1, 20),
            'criterion': Categorical(['gini', 'entropy']),
            'max_features': Categorical([None, 'sqrt', 'log2'])
        },
        n_iter=50, n_points=100, cv=stratified_cv,
        scoring='balanced_accuracy', random_state=random_seed
    )

    bayes_search.fit(x_train, y_train)
    best_model_bayes = bayes_search.best_estimator_
    y_pred_bayes = best_model_bayes.predict(x_test)
    y_prob_bayes = best_model_bayes.predict_proba(x_test)

    acc_bayes = balanced_accuracy_score(y_test, y_pred_bayes)
    f1_bayes = f1_score(y_test, y_pred_bayes, average='weighted')
    recall_bayes = recall_score(y_test, y_pred_bayes, average='weighted')
    precision_bayes = precision_score(y_test, y_pred_bayes, average='weighted')
    conf_matrix_bayes = confusion_matrix(y_test, y_pred_bayes)
    acc_score = accuracy_score (y_test, y_pred_bayes)

    lv_importance = best_model_bayes.feature_importances_
    top_wavenumbers, top_importances = func_back_projection(lv_importance, loadings, wavenumbers,
                                                             acc_bayes, target_column, sample_type, train_percentage,
                                                             f"Decision Tree (Bayesian Optimization){suffix}", group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(cross_validation_results, sample_type, train_percentage,
                                               acc_bayes, f1_bayes, recall_bayes, precision_bayes, conf_matrix_bayes,
                                               bayes_search, acc_score, f"Decision Tree (Bayesian Optimization){suffix}")

    plot_confusion_matrix(y_test, y_pred_bayes, label_encoder, acc_bayes, sample_type, train_percentage,
                          f"Decision Tree (Bayesian Optimization){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob_bayes, label_encoder, sample_type, train_percentage,
                              acc_bayes, f"Decision Tree (Bayesian Optimization){suffix}", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"Decision Tree (Bayesian Optimization){suffix}", acc_bayes, f1_bayes,
                                            roc_auc, top_wavenumbers, top_importances, back_projection)

    return results, cross_validation_results, back_projection


def xboost(x_train, y_train, x_test, y_test, label_encoder,
           sample_type, train_percentage, loadings, wavenumbers,
           results, cross_validation_results, target_column,
           back_projection, group_fam_to_use=None):
    model_name = "XGBoost"
    group_suffix = f"_{group_fam_to_use}" if group_fam_to_use else ""

    # GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=random_seed),
                               param_grid, cv=5, scoring="balanced_accuracy")
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    test_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    acc_score = accuracy_score (y_test, y_pred)

    booster = best_model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')
    lv_importance = np.zeros(x_train.shape[1])
    for i in range(len(lv_importance)):
        key = f"f{i}"
        if key in importance_dict:
            lv_importance[i] = importance_dict[key]
    lv_importance /= lv_importance.sum()

    top_wavenumbers, top_importances = func_back_projection(
        lv_importance, loadings, wavenumbers, test_acc, target_column,
        sample_type, train_percentage, f"{model_name} (GridSearchCV)", group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(
        cross_validation_results, sample_type, train_percentage, test_acc,
        f1, recall, precision, cm, grid_search, acc_score, f"{model_name} (GridSearchCV)"
    )

    plot_confusion_matrix(y_test, y_pred, label_encoder, test_acc, sample_type, train_percentage,
                          f"{model_name} (GridSearchCV)", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob, label_encoder, sample_type, train_percentage,
                             test_acc, f"{model_name} (GridSearchCV)", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"{model_name} (GridSearchCV)", test_acc, f1, roc_auc,
                                            top_wavenumbers, top_importances, back_projection)

    # Bayesian Optimization
    bayes_search = BayesSearchCV(XGBClassifier(eval_metric='logloss', random_state=random_seed), {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(1, 15),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'min_child_weight': Integer(1, 10),
        'gamma': Real(0, 5)
    }, n_iter=50, cv=stratified_cv, scoring="balanced_accuracy", random_state=random_seed)

    bayes_search.fit(x_train, y_train)
    best_model = bayes_search.best_estimator_

    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)

    test_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    acc_score = accuracy_score(y_test, y_pred)

    booster = best_model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')
    lv_importance = np.zeros(x_train.shape[1])
    for i in range(len(lv_importance)):
        key = f"f{i}"
        if key in importance_dict:
            lv_importance[i] = importance_dict[key]
    lv_importance /= lv_importance.sum()

    top_wavenumbers, top_importances = func_back_projection(
        lv_importance, loadings, wavenumbers, test_acc, target_column,
        sample_type, train_percentage, f"{model_name} (BayesOpt)", group_fam_to_use=group_fam_to_use)

    cross_validation_results = func_cv_results(
        cross_validation_results, sample_type, train_percentage, test_acc,
        f1, recall, precision, cm, bayes_search, acc_score,  f"{model_name} (BayesOpt)"
    )

    plot_confusion_matrix(y_test, y_pred, label_encoder, test_acc, sample_type, train_percentage,
                          f"{model_name} (BayesOpt)", target_column, group_fam_to_use=group_fam_to_use)

    roc_auc = plot_roc_curve(y_test, y_prob, label_encoder, sample_type, train_percentage,
                             test_acc, f"{model_name} (BayesOpt)", target_column, group_fam_to_use=group_fam_to_use)

    results, back_projection = results_func(results, sample_type, train_percentage,
                                            f"{model_name} (BayesOpt)", test_acc, f1, roc_auc,
                                            top_wavenumbers, top_importances, back_projection)

    return results, cross_validation_results, back_projection



