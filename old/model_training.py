import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Enable auto-logging
mlflow.sklearn.autolog()

with mlflow.start_run():
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid={
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10]
        },
        cv=5
    )
    
    # Fit triggers automatic logging
    grid_search.fit(X_train, y_train)