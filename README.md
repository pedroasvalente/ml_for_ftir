# ml4fir

[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-orange?logo=dagshub)](https://dagshub.com/pedroasvalente/ml_for_ftir)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/pedroasvalente/ml_for_ftir)

Machine learning pipeline for classifying athletes (football players, ultrarunners, sedentary controls) using **FTIR (Fourier-Transform Infrared Spectroscopy)** data from biological samples (capillary blood, plasma, saliva, serum, urine).

> **Code** is versioned on GitHub. **Experiment results** (metrics, models, artefacts) are tracked on DagsHub via MLflow.

---

## Setup

### Requirements
- Python >= 3.11
- CUDA-compatible GPU recommended (tested with NVIDIA MX550, CUDA 12.8)

### Installation

```bash
git clone https://github.com/pedroasvalente/ml_for_ftir.git
cd ml_for_ftir
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### DagsHub Authentication (first time only)

All MLflow experiment data is sent to DagsHub automatically. Authenticate once:

```python
import dagshub
dagshub.auth.add_app_token(token="YOUR_TOKEN")
```

Get your token at: **DagsHub → Settings → Access Tokens**

---

## Usage

### Train

Use an existing experiment config from `experiments/configs/simple_ml/`:

```bash
ml4fir train experiments/configs/simple_ml/exp_group_fam.json
```

After training, results are available at:
- **DagsHub:** https://dagshub.com/pedroasvalente/ml_for_ftir (metrics, models, artefacts)
- **Local:** `experiments/<target>/experiment_configs.csv` and `results/<target>/`

### Running on two machines in parallel

Split the workload by sample type:

```bash
# PC 1 — CAPILAR, PLASMA, SALIVA (576 runs)
ml4fir train experiments/configs/simple_ml/exp_group_fam_pc1.json

# PC 2 — SERUM, URINE (384 runs)
ml4fir train experiments/configs/simple_ml/exp_group_fam_pc2.json
```

Both machines send results to the same DagsHub experiment. No conflicts — each machine handles different sample types.

### Predict

```bash
ml4fir predict example.csv --target-to-predict=group_fam --sample-type=PLASMA
```

### Plot

```bash
ml4fir plot --target-to-predict=group_fam --sample-type=PLASMA
```

### Clear local MLflow runs

```bash
ml4fir clear_runs
```

---

## Experiment Configuration

Each experiment is defined by a JSON file. Example (`exp_group_fam.json`):

```json
{
    "experiment_name": "FTIR Supervised Training - GROUP_FAM",
    "run_name": "group_fam",
    "searchs_hipermetrics": ["grid", "bayes"],
    "model_types_to_train": ["mlp_classifier", "random_forest", "decision_tree", "xgboost"],
    "train_percentages": [0.8, 0.7, 0.6],
    "sample_types": ["CAPILAR", "PLASMA", "SALIVA", "SERUM", "URINE"],
    "targets_to_predict": ["group_fam"],
    "scale": [true, false],
    "apply_pls": [true, false],
    "apply_smote_resampling": [true, false],
    "n_components": [10],
    "num_classes": [3]
}
```

**Available models:** `random_forest`, `mlp_classifier`, `decision_tree`, `xgboost`  
**Hyperparameter search:** `grid` (GridSearchCV), `bayes` (BayesSearchCV)  
**36 prediction targets** available in `experiments/configs/simple_ml/`

---

## MLflow Run Hierarchy

```
main_run
└── sample_type_run  (CAPILAR | PLASMA | SALIVA | SERUM | URINE)
    └── search_run  (GridSearchCV | BayesSearchCV)
        └── model_run  (e.g. "Random Forest_pls_no-smote_scale")
```

Each **model_run** logs:
- **Parameters:** model, target, sample_type, scale, apply_pls, n_components, apply_smote, train_size, test_size, train_class_dist, test_class_dist
- **Metrics:** accuracy, balanced accuracy, F1, recall, precision, ROC AUC
- **Artefacts:** experiment config JSON, plots (confusion matrix, ROC, wavenumber importances)

---

## Project Structure

```
ml_for_ftir/
├── data/
│   └── processed/
│       └── 001_3_cleaned_FTIR.csv   <- FTIR dataset
├── experiments/
│   ├── configs/
│   │   ├── simple_ml/               <- 36 experiment JSON configs
│   │   └── deep_learning/
│   └── <target>/
│       └── experiment_configs.csv   <- run history per target
├── ml4fir/                          <- main package
│   ├── config.py                    <- paths, env vars, DagsHub init
│   ├── cli.py                       <- CLI entry points (typer)
│   ├── data/
│   │   ├── data.py                  <- DataHandler class
│   │   └── load_data.py             <- preprocessing pipeline
│   ├── modeling/
│   │   ├── train.py                 <- training orchestration
│   │   ├── train_utils.py           <- MLflow logging + model training
│   │   ├── models.py                <- model definitions + hyperparameters
│   │   ├── predict.py               <- inference pipeline
│   │   └── utils.py                 <- MLflow query utilities
│   └── ploting/
│       └── ploting_functions.py     <- ROC, confusion matrix, feature importance
├── results/                         <- summary CSVs per target
├── models/                          <- saved model artefacts (local)
├── reports/figures/                 <- generated plots
└── pyproject.toml
```

---

## Environment Variables (`ml4fir/.env`)

| Variable | Default | Description |
|---|---|---|
| `TRAINING_DATA_FILENAME` | `001_3_cleaned_FTIR.csv` | Input dataset filename |
| `RANDOM_SEED` | `52` | Global random seed |
| `GLOBAL_THRESHOLD` | `70` | Minimum accuracy threshold (%) |
| `MAIN_METRIC` | `acc` | Primary metric for classification |
| `MAIN_METRIC_LINEAR` | `rmse` | Primary metric for regression |

> `MLFLOW_TRACKING_URI` is set automatically by `dagshub.init()` — do not override it.

---

## License

MIT — Pedro Valente
