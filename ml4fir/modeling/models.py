from typing import Dict, List, Optional, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class BaseModelConfig:
    """Base class for model configurations."""

    def __init__(
        self,
        name,
        model_fn,
        random_seed=54,
        desc_name: Optional[str] = None,
        model_args: Optional[Dict] = None,
    ):

        self.name = name or self.name
        self.desc_name = desc_name or self.name.replace("_", " ").title()
        self.random_seed = random_seed
        self.model_args = model_args or {}
        self.model_fn = model_fn

    def get_model(self, **kwargs):
        model_args = self.model_args.copy()
        model_args.update(kwargs)
        return self.model_fn(**model_args, random_state=self.random_seed)

    def _get_params(self, param_type: str, **kwargs):
        param_list = getattr(self, f"{param_type}", [])
        param = {}
        for arg in param_list:
            if hasattr(self, arg):
                attr = getattr(self, arg)
                if isinstance(attr, dict):
                    attr = attr[param_type]
                param[arg] = attr
        param.update(kwargs)
        return param

    def get_param_grid(self, **kwargs):
        return self._get_params(param_type="param_grid_args", **kwargs)

    def get_bayes_search_params(self, **kwargs):
        return self._get_params(param_type="bayes_search_params", **kwargs)


class RandomForestConfig(BaseModelConfig):
    param_grid_args = ["n_estimators", "max_depth", "max_features", "criterion"]
    bayes_search_params = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "bootstrap",
    ]

    def __init__(
        self,
        name="random_forest",
        desc_name="Random Forest",
        n_estimators: Optional[Union[List, Dict]] = None,
        max_depth: Optional[Union[List, Dict]] = None,
        max_features: Optional[Union[List, Dict]] = None,
        criterion: Optional[Union[List, Dict]] = None,
        min_samples_split: Optional[Union[List, Dict]] = None,
        min_samples_leaf: Optional[Union[List, Dict]] = None,
        bootstrap: Optional[Union[List, Dict]] = None,
        **kwargs,
    ):
        self.n_estimators = n_estimators or [100, 200]
        self.max_depth = max_depth or [4, 8, 12]
        self.max_features = max_features or ["sqrt", "log2"]
        self.criterion = criterion or ["gini", "entropy"]
        self.min_samples_split = min_samples_split or [2, 5, 10]
        self.min_samples_leaf = min_samples_leaf or [1, 2, 4]
        self.bootstrap = bootstrap or [True, False]

        super().__init__(
            name=name, desc_name=desc_name, model_fn=RandomForestClassifier, **kwargs
        )


class MLPConfig(BaseModelConfig):
    param_grid_args = [
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "learning_rate",
    ]
    bayes_search_params = [
        "activation",
        "solver",
        "alpha",
        "learning_rate_init",
        "max_iter",
    ]

    def __init__(
        self,
        name="mlp",
        desc_name="MLP",
        model_fn=MLPClassifier,
        model_args: Optional[Dict] = None,
        hidden_layer_sizes: Optional[Union[List, Dict]] = None,
        activation: Optional[Union[List, Dict]] = None,
        solver: Optional[Union[List, Dict]] = None,
        alpha: Optional[Union[List, Dict]] = None,
        learning_rate: Optional[Union[List, Dict]] = None,
        learning_rate_init: Optional[Union[List, Dict]] = None,
        max_iter: Optional[Union[List, Dict]] = None,
        **kwargs,
    ):
        model_args = model_args or {}
        defaul_model_args = {
            "max_iter": 3000,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "learning_rate_init": 0.01,
        }
        model_args.update(defaul_model_args)

        self.hidden_layer_sizes = hidden_layer_sizes or [(50,), (100,), (50, 50)]
        self.activation = activation or ["tanh", "relu"]
        self.solver = solver or ["sgd", "adam"]
        self.alpha = alpha or [0.0001, 0.001]
        self.learning_rate = learning_rate or ["constant", "adaptive"]
        self.learning_rate_init = learning_rate_init or [0.001, 0.01]
        self.max_iter = max_iter or [1000, 3000]

        super().__init__(
            name=name,
            desc_name=desc_name,
            model_fn=model_fn,
            model_args=model_args,
            **kwargs,
        )


class DecisionTreeConfig(BaseModelConfig):
    param_grid_args = [
        "criterion",
        "splitter",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
    ]
    bayes_search_params = [
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "criterion",
        "max_features",
    ]

    def __init__(
        self,
        name="decision_tree",
        desc_name="Decision Tree",
        model_fn=DecisionTreeClassifier,
        criterion: Optional[Union[List, Dict]] = None,
        splitter: Optional[Union[List, Dict]] = None,
        max_depth: Optional[Union[List, Dict]] = None,
        min_samples_split: Optional[Union[List, Dict]] = None,
        min_samples_leaf: Optional[Union[List, Dict]] = None,
        max_features: Optional[Union[List, Dict]] = None,
        **kwargs,
    ):
        self.criterion = criterion or ["gini", "entropy"]
        self.splitter = splitter or ["best", "random"]
        self.max_depth = max_depth or [None, 10, 20]
        self.min_samples_split = min_samples_split or [2, 5, 10]
        self.min_samples_leaf = min_samples_leaf or [1, 2, 4]
        self.max_features = max_features or [None, "sqrt", "log2"]

        super().__init__(name=name, desc_name=desc_name, model_fn=model_fn, **kwargs)


class XGBoostConfig(BaseModelConfig):
    param_grid_args = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
    ]
    bayes_search_params = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
    ]

    def __init__(
        self,
        name="xgboost",
        desc_name="XGBoost",
        model_fn=XGBClassifier,
        n_estimators: Optional[Union[List, Dict]] = None,
        max_depth: Optional[Union[List, Dict]] = None,
        learning_rate: Optional[Union[List, Dict]] = None,
        subsample: Optional[Union[List, Dict]] = None,
        colsample_bytree: Optional[Union[List, Dict]] = None,
        min_child_weight: Optional[Union[List, Dict]] = None,
        gamma: Optional[Union[List, Dict]] = None,
        model_args: Optional[Dict] = None,
        **kwargs,
    ):
        model_args = model_args or {}
        defaul_model_args = {"eval_metric": "logloss"}
        model_args.update(defaul_model_args)

        self.n_estimators = n_estimators or [50, 100]
        self.max_depth = max_depth or [3, 6, 10]
        self.learning_rate = learning_rate or [0.01, 0.1]
        self.subsample = subsample or [0.8, 1.0]
        self.colsample_bytree = colsample_bytree or [0.8, 1.0]
        self.min_child_weight = min_child_weight or [1, 5, 10]
        self.gamma = gamma or [0, 1, 5]

        super().__init__(
            name=name,
            desc_name=desc_name,
            model_fn=model_fn,
            model_args=model_args,
            **kwargs,
        )


def get_model_config(model_type):
    """
    Factory function to retrieve the appropriate model configuration class.

    Parameters:
        model_type (str): The type of model. Options are:
                          'random_forest', 'mlp', 'decision_tree', 'xgboost'.
        random_seed (int): The random seed for reproducibility.

    Returns:
        BaseModelConfig: An instance of the appropriate configuration class.
    """
    config_classes = {
        "random_forest": RandomForestConfig,
        "mlp": MLPConfig,
        "decision_tree": DecisionTreeConfig,
        "xgboost": XGBoostConfig,
    }

    if model_type not in config_classes:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return config_classes[model_type]
