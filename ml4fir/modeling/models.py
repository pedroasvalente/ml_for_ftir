import itertools

from alquimodelia.alquimodelia import ModelMagia
from keras.wrappers import (
    SKLearnClassifier,
    SKLearnRegressor,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from ml4fir.config import random_seed


class BaseModelConfig:
    """Base class for model configurations."""

    def __init__(
        self,
        name,
        model_fn,
        random_seed=random_seed,
        desc_name: str | None = None,
        model_args: dict | None = None,
    ):
        self.name = name or self.name
        self.desc_name = desc_name or self.name.replace("_", " ").title()
        self.random_seed = random_seed
        self.model_args = model_args or {}
        self.model_fn = model_fn
        # self._set_model_fn()

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
        params = self._get_params(param_type="bayes_search_params", **kwargs)
        params.update({"random_state": self.random_seed})
        return params

    def get_params(self, param_type, **kwargs):
        if param_type in ["grid", "param_grid_args", "GridSearchCV"]:
            param_type = "param_grid_args"
        elif param_type in ["bayes", "bayes_search_params", "BayesSearchCV"]:
            param_type = "bayes_search_params"
        else:
            raise ValueError(f"Unsupported param_type: {param_type}")
        return self._get_params(param_type=param_type, **kwargs)


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
        model_fn=RandomForestClassifier,
        n_estimators: list | dict | None = None,
        max_depth: list | dict | None = None,
        max_features: list | dict | None = None,
        criterion: list | dict | None = None,
        min_samples_split: list | dict | None = None,
        min_samples_leaf: list | dict | None = None,
        bootstrap: list | dict | None = None,
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
            name=name, desc_name=desc_name, model_fn=model_fn, **kwargs
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
        name="mlp_classifier",
        desc_name="MLP",
        model_fn=MLPClassifier,
        model_args: dict | None = None,
        hidden_layer_sizes: list | dict | None = None,
        activation: list | dict | None = None,
        solver: list | dict | None = None,
        alpha: list | dict | None = None,
        learning_rate: list | dict | None = None,
        learning_rate_init: list | dict | None = None,
        max_iter: list | dict | None = None,
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

        self.hidden_layer_sizes = hidden_layer_sizes or [
            (50,),
            (100,),
            (50, 50),
        ]
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
        criterion: list | dict | None = None,
        splitter: list | dict | None = None,
        max_depth: list | dict | None = None,
        min_samples_split: list | dict | None = None,
        min_samples_leaf: list | dict | None = None,
        max_features: list | dict | None = None,
        **kwargs,
    ):
        self.criterion = criterion or ["gini", "entropy"]
        self.splitter = splitter or ["best", "random"]
        self.max_depth = max_depth or [None, 10, 20]
        self.min_samples_split = min_samples_split or [2, 5, 10]
        self.min_samples_leaf = min_samples_leaf or [1, 2, 4]
        self.max_features = max_features or [None, "sqrt", "log2"]

        super().__init__(
            name=name, desc_name=desc_name, model_fn=model_fn, **kwargs
        )


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
        n_estimators: list | dict | None = None,
        max_depth: list | dict | None = None,
        learning_rate: list | dict | None = None,
        subsample: list | dict | None = None,
        colsample_bytree: list | dict | None = None,
        min_child_weight: list | dict | None = None,
        gamma: list | dict | None = None,
        model_args: dict | None = None,
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

    def _set_model_fn(self):
        self.model_fn = self.model_fn


class RandomForestRegressorConfig(RandomForestConfig):
    def __init__(self, **kwargs):
        super().__init__(
            name="random_forest_regressor",
            desc_name="Random Forest Regressor",
            model_fn=RandomForestRegressor,
            **kwargs,
        )


class DecisionTreeRegressorConfig(DecisionTreeConfig):
    def __init__(self, **kwargs):
        super().__init__(
            name="decision_tree_regressor",
            desc_name="Decision Tree Regressor",
            model_fn=DecisionTreeRegressor,
            **kwargs,
        )


class MLPRegressorConfig(MLPConfig):
    def __init__(self, **kwargs):
        super().__init__(
            name="mlp_regressor",
            desc_name="MLP Regressor",
            model_fn=MLPRegressor,
            **kwargs,
        )


class XGBoostRegressorConfig(XGBoostConfig):
    def __init__(self, **kwargs):
        model_args = kwargs.get("model_args", {})
        model_args.update({"objective": "reg:squarederror"})
        kwargs["model_args"] = model_args
        super().__init__(
            name="xgboost_regressor",
            desc_name="XGBoost Regressor",
            model_fn=XGBRegressor,
            **kwargs,
        )


class KerasMagiaConfig(BaseModelConfig):
    """
    Model config for Keras models built with ModelMagia.
    Expects model_fn to be ModelMagia and model_args to include 'model_arch' and any additional args.
    """

    # Common search params for all KerasMagia models
    search_params = [
        "model_kwargs",
        "compile_kwargs",
        "fit_kwargs",
    ]
    model_keys = [
        "activation",
        "dropout",
        "layers",
        "filters",
        "kernel_size",
        "pool_size",
        "input_dim",
    ]
    param_grid_args = search_params
    bayes_search_params = search_params
    compile_keys = ["optimizer", "loss", "metrics"]
    fit_args = ["epochs", "batch_size"]

    def __init__(
        self,
        name="keras_magia",
        desc_name="Keras ModelMagia",
        model_fn=ModelMagia,
        sklearn_wrapper=None,
        model_arch=None,
        model_kwargs=None,
        model_args=None,
        compile_kwargs=None,
        fit_kwargs=None,
        **kwargs,
    ):
        model_args = model_args or {}
        model_kwargs = model_kwargs or {}
        self.model_arch = model_arch
        self.model_kwargs = model_kwargs
        self.compile_kwargs = compile_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}

        self.sklearn_wrapper = sklearn_wrapper
        super().__init__(
            name=name,
            desc_name=desc_name,
            model_fn=model_fn,
            model_args=model_args,
            **kwargs,
        )

    def _get_params(self, param_type: str, **kwargs):
        param_list = getattr(self, f"{param_type}", [])
        param_grid = {}
        for arg in param_list:
            if hasattr(self, arg):
                attr = getattr(self, arg)
                sequences = (
                    attr[param_type] if hasattr(attr, param_type) else attr
                )

                keys = list(sequences.keys())
                values = [sequences[k] for k in keys]
                param_combinations = [
                    dict(zip(keys, combo, strict=False))
                    for combo in itertools.product(*values)
                ]
                param_grid[arg] = param_combinations

        model_kwargs_list = param_grid["model_kwargs"]  # List of dicts
        compile_kwargs_list = param_grid["compile_kwargs"]  # List of dicts

        # Generate all combinations
        all_combinations = [
            {**mk, "compile_kwargs": {**ck}}
            for mk, ck in itertools.product(
                model_kwargs_list, compile_kwargs_list
            )
        ]

        param = {
            "model_kwargs": all_combinations,
            "fit_kwargs": param_grid.get("fit_kwargs", [{}]),
        }
        param.update(kwargs)
        return param

    def build_model(self, **params):
        model_args = {}
        compile_kwargs = {}
        compile_kwargs.update(params.pop("compile_kwargs", {}))
        model_args.update(params)
        model_args.update(
            {
                "input_shape": self.input_shape,
                "output_shape": self.output_shape,
                "num_classes": self.num_classes,
            }
        )
        internal_model = self.model_fn(
            self.model_arch, **model_args, random_state=self.random_seed
        ).model
        internal_model.compile(**compile_kwargs)
        return internal_model

    def get_model(self, **params):
        return self.sklearn_wrapper(
            self.build_model,
            **params,
        )

    def set_databasedatributes(self, datahandler):
        """
        Set input and output shapes for ModelMagia models using the datahandler.
        """
        x_train = datahandler.x_train
        y_train = datahandler.y_train
        self.input_shape = x_train.shape[1:]
        # Optionally, store num_classes if relevant
        if hasattr(datahandler, "num_classes"):
            self.num_classes = datahandler.num_classes
        else:
            self.num_classes = 1
        self.output_shape = (
            y_train.shape[1:] if len(y_train.shape) > 1 else (self.num_classes,)
        )


class KerasMagiaClassifier(KerasMagiaConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sklearn_wrapper=SKLearnClassifier, **kwargs)


class KerasMagiaRegressor(KerasMagiaConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sklearn_wrapper=SKLearnRegressor, **kwargs)


names_dict = {
    "random_forest": "Random Forest",
    "mlp_classifier": "MLP",
    "decision_tree": "Decision Tree",
    "xgboost": "XGBoost",
    "fcnn":"FCNN",
    "cnn": "CNN",
    "transformer": "Transformer",
    "unet": "UNet",
}
