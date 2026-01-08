# %% [markdown]
# ## Load Dataset

# %%
from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features      
y = wine.data.targets  
df = pd.concat([X, y], axis=1)
  
# metadata 
print(wine.metadata) 
  
# variable information 
print(wine.variables) 

# %% [markdown]
# ## EDA

# %%
df.columns

# %%
df.shape

# %%
# check data quality
from CustomUtils import DataQualityCheck

DataQualityCheck.data_quality_report(input_df=df, type='df')

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %% [markdown]
# ## Initialize ZenML

# %%
from zenml import pipeline, step
from zenml.client import Client
from typing import Tuple, Dict, List, Annotated
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from zenml.logger import get_logger
import numpy as np

logger = get_logger(__name__)

# Initialize ZenML client
client = Client()

# %% [markdown]
# ## Define ZenML Steps

# %%
@step
def load_data() -> Tuple[
    Annotated[pd.DataFrame, "features"],
    Annotated[pd.DataFrame, "targets"]
]:
    """Load the wine dataset."""
    from ucimlrepo import fetch_ucirepo
    
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets
    
    logger.info(f"Loaded data - X shape: {X.shape}, y shape: {y.shape}")
    return X, y

@step
def split_data(
    X: pd.DataFrame, 
    y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 64
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"]
]:
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split data - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

@step
def save_feature_names(X_train: pd.DataFrame) -> List[str]:
    """Save feature names to file and return them."""
    feature_names = X_train.columns.to_list()
    
    with open("feature_names.txt", "w") as f:
        for c in feature_names:
            f.write(c + "\n")
    
    logger.info(f"Saved {len(feature_names)} feature names")
    return feature_names

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 64
) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[Dict, "params"]
]:
    """Train a Random Forest model and return accuracy and params."""
    # Convert y to 1D array if needed
    y_train_array = y_train.values.ravel()
    y_test_array = y_test.values.ravel()
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train_array)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_array, preds)
    
    params = {"n_estimators": n_estimators, "random_state": random_state}
    
    logger.info(f'Accuracy: {acc:.4f}')
    print(f'Accuracy: {acc:.4f}')
    
    return acc, params

# %% [markdown]
# ## Simple Training Pipeline

# %%
@pipeline
def training_pipeline(
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 64
):
    """Simple training pipeline."""
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X=X, 
        y=y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Save feature names
    feature_names = save_feature_names(X_train=X_train)
    
    # Train model
    accuracy, params = train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_estimators=n_estimators,
        random_state=random_state
    )

# Run the pipeline
training_pipeline()

# %% [markdown]
# ## Grid Search Pipeline

# %%
@step
def grid_search_train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_estimators_list: List[int]
) -> Tuple[
    Annotated[float, "best_accuracy"],
    Annotated[Dict, "best_params"],
    Annotated[List[Dict], "all_results"]
]:
    """Perform grid search and return best model info and all results."""
    from sklearn.model_selection import GridSearchCV
    
    # Convert y to 1D array
    y_train_array = y_train.values.ravel()
    y_test_array = y_test.values.ravel()
    
    param_grid = {"n_estimators": n_estimators_list}
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=64, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="accuracy",
        return_train_score=True
    )
    
    grid.fit(X_train, y_train_array)
    
    all_results = []
    best_acc = 0
    best_params = {}
    
    # Iterate over each candidate
    for i in range(len(grid.cv_results_["params"])):
        params = grid.cv_results_["params"][i]
        mean_val = grid.cv_results_["mean_test_score"][i]
        std_val = grid.cv_results_["std_test_score"][i]
        
        # Build & refit the model manually for logging
        model = RandomForestClassifier(**params, random_state=64, n_jobs=-1)
        model.fit(X_train, y_train_array)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test_array, preds)
        
        result = {
            "params": params,
            "cv_mean_accuracy": float(mean_val),
            "cv_std_accuracy": float(std_val),
            "test_accuracy": float(acc)
        }
        all_results.append(result)
        
        logger.info(f"Model {i+1}: n_estimators={params['n_estimators']}, "
                   f"test_acc={acc:.4f}, cv_mean={mean_val:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = params
    
    logger.info(f"Best model: {best_params}, accuracy: {best_acc:.4f}")
    
    return best_acc, best_params, all_results

@pipeline
def grid_search_pipeline(
    test_size: float = 0.2,
    random_state: int = 64
):
    """Grid search training pipeline."""
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save feature names
    feature_names = save_feature_names(X_train=X_train)
    
    # Grid search
    n_estimators_list = [72, 100, 125, 150, 200, 250]
    best_acc, best_params, all_results = grid_search_train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_estimators_list=n_estimators_list
    )

# Run the grid search pipeline
grid_search_pipeline()

# %%
# Load feature names
with open("feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

print(f"Feature names: {feature_names}")

# %% [markdown]
# ## View ZenML Dashboard
# 
# To view your pipeline runs, models, and artifacts:
# 
# ```bash
# zenml up
# ```
# 
# The dashboard will show:
# - All pipeline runs with their steps
# - Artifacts (datasets, models, metrics)
# - Model versions and metadata
# - Lineage tracking

# %% [markdown]
# ## Model Deployment
# 
# For production deployment with ZenML, you can use various deployment integrations:
# 
# ```python
# # Example: Deploy with MLflow (requires mlflow integration)
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
# 
# @pipeline
# def deployment_pipeline():
#     # ... training steps ...
#     mlflow_model_deployer_step(
#         model=trained_model,
#         deploy_decision=True
#     )
# ```
# 
# Or use other deployers:
# - Seldon Core
# - KServe
# - BentoML
# - Custom deployers
# 
# Install integration: `zenml integration install mlflow`

print("✅ Pipelines completed! Check ZenML dashboard for results.")