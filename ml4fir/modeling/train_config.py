from sklearn.model_selection import StratifiedKFold

from ml4fir.modeling.conf import random_seed

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)


grid_search_args = {
    "scoring": "balanced_accuracy",
    "cv": 5,
}
# WHY: why are the BayesSearchCV parameters different in each model? Like this you are not comparing
# the same search space for each model.
search_args = {
    "random_forest": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "mlp_classifier": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "decision_tree": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 100,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "xboost": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
}

# WHY: why are the model_args_conf parameters different in each model? Like this you are not comparing the same model!
model_args_conf = {
    "mlp_classifier": {
        "GridSearchCV": {
            "max_iter": 3000,
            "learning_rate_init": 0.01,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
        "BayesSearchCV": {
            "validation_fraction": 0.1,
            "hidden_layer_sizes": (200,),
            "early_stopping": True,
        },
    },
}
