# ml4fir

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Short desc

## Running the Training Script

To run the `train.py` script, follow these steps:

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv name_venv
   ```

2. **Activate the Virtual Environment**:
   - On Linux/Mac:
     ```bash
     source name_venv/bin/activate
     ```
   - On Windows:
     ```bash
     name_venv\Scripts\activate
     ```

3. **Install the Project in Editable Mode**:
    With the terminal inside the project.
   ```bash
   pip install -e .
   ```

4. **Run the Training Script**:
   ```bash
   ml4fir train path_to_training.json
   ```

This will execute the training process as defined in `train.py`. Make sure to configure any necessary parameters or dependencies before running the script.

5. **Open the MLflow UI**:
   To monitor and visualize your training runs, open the MLflow UI:

   ```bash
   mlflow ui
   ```

This will start the MLflow tracking server. By default, the UI will be accessible at http://localhost:5000. Open this link in your browser to explore your experiment runs, metrics, parameters, and artifacts.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ml4fir and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ml4fir   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ml4fir a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

