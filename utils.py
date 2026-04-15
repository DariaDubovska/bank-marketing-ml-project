import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

def plot_numeric_block(df, features, target='y', bins=30, rename=None, q=20):
    """
    Plot distribution and target-rate trends for a list of numerical features.

    For each numerical feature, the function creates two plots:
    1. A histogram with KDE showing the feature distribution.
    2. A line plot showing the probability of the positive class
       (P(target='yes')) across binned feature values.

    Binning is performed with:
    - `pd.qcut` when the feature has enough unique values, so that bins contain
      roughly equal numbers of observations.
    - `pd.cut` when the number of unique values is small or when `qcut`
      cannot be applied.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        features (list[str]): List of numerical feature names to plot.
        target (str, optional): Name of the target column. Defaults to 'y'.
        bins (int, optional): Number of bins for the histogram. Defaults to 30.
        rename (dict, optional): Mapping of original feature names to display
            names for plot titles and axis labels. Defaults to None.
        q (int, optional): Number of bins for the target-rate plot.
            Recommended range is 15-30. Defaults to 20.

    Returns:
        None: Displays the plots.

    Notes:
        - The function assumes a binary target where the positive class is
          labeled as `'yes'`.
        - Missing values in the feature columns are excluded from plotting.
    """
    n = len(features)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = np.array([axes])

    y_yes = (df[target] == 'yes').astype(int)

    for i, f in enumerate(features):
        display_name = rename.get(f, f) if rename else f

        # Feature distribution
        sns.histplot(df[f].dropna(), bins=bins, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f"Distribution {display_name}")
        axes[i, 0].set_xlabel(display_name)

        # P(y='yes') by binned feature values
        x = df[f]

        # If the feature has only a few unique values, use regular cut;
        # otherwise use qcut to create quantile-based bins.
        try:
            if x.nunique(dropna=True) <= q:
                binned = pd.cut(x, bins=min(q, x.nunique(dropna=True)))
            else:
                binned = pd.qcut(x, q=q, duplicates='drop')
        except Exception:
            # Fallback in case qcut fails due to repeated values or similar issues
            binned = pd.cut(x, bins=q)

        rate = (
            pd.DataFrame({'bin': binned, 'y_yes': y_yes})
            .dropna(subset=['bin'])
            .groupby('bin', observed=True)['y_yes']
            .mean()
            .reset_index()
        )

        # Use interval midpoints as x-values for a cleaner line plot
        def midpoint(interval):
            return (interval.left + interval.right) / 2

        rate['x_mid'] = rate['bin'].apply(midpoint)

        sns.lineplot(data=rate, x='x_mid', y='y_yes', marker='o', ax=axes[i, 1])
        axes[i, 1].set_title(f"P(y='yes') by {display_name}")
        axes[i, 1].set_xlabel(display_name)
        axes[i, 1].set_ylabel("P(y='yes')")
        axes[i, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def plot_categorical_combined(df, feature, target='y'):
    """
    Plot distribution and target rate for a categorical feature.

    This function creates a combined visualization with:
    - A bar plot showing the count of observations per category.
    - A line plot showing the probability of the positive class 
      (P(target='yes')) for each category.

    The plot uses dual y-axes:
    - Left axis: count of observations
    - Right axis: probability of the positive class

    Args:
        df (pd.DataFrame): Input DataFrame containing the feature and target.
        feature (str): Name of the categorical feature to analyze.
        target (str, optional): Name of the target column. Defaults to 'y'.

    Returns:
        None: Displays the plot.

    Notes:
        - The function assumes a binary target where the positive class 
          is labeled as 'yes'.
        - Categories are sorted by the probability of the positive class 
          (descending) for better interpretability.
    """
    summary = (
        df
        .groupby(feature)[target]
        .agg(['count', lambda x: (x == 'yes').mean()])
        .rename(columns={'<lambda_0>': 'p_yes'})
        .reset_index()
        .sort_values('p_yes', ascending=False)
    )

    fig, ax1 = plt.subplots(figsize=(12,6))

    sns.barplot(data=summary, x=feature, y='count', ax=ax1, alpha=0.6)
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    sns.lineplot(data=summary, x=feature, y='p_yes', ax=ax2, marker='o', color='red')
    ax2.set_ylabel('P(y=yes)')

    plt.title(f"{feature}: distribution and P(y='yes')")
    plt.show()

def get_optimal_threshold(y_true, y_probs, metric=f1_score):
    """
    Find the optimal classification threshold based on a given metric.

    Iterates over thresholds from 0 to 1 and selects the one that maximizes
    the specified evaluation metric.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_probs (array-like): Predicted probabilities for the positive class.
        metric (callable, optional): Metric function to optimize. 
            Should take (y_true, y_pred) as input. Defaults to f1_score.

    Returns:
        tuple:
            float: Optimal threshold.
            float: Best metric score achieved at this threshold.
    """
    thresholds = np.linspace(0, 1, 100)
    scores = [metric(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

def evaluate_model_cv(estimator, X, y, cv, scoring, n_jobs=-1):
    """
    Evaluate a model using cross-validation.

    Computes mean cross-validation scores for the specified metrics.

    Args:
        estimator: Scikit-learn estimator or pipeline.
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        cv: Cross-validation splitter.
        scoring (dict): Dictionary of scoring metrics 
            (e.g., {"roc_auc": "roc_auc", "f1": "f1"}).
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

    Returns:
        dict: Mean cross-validation scores with keys formatted as 
            "cv_<metric_name>".
    """
    results = cross_validate(
      estimator=estimator,
      X=X,
      y=y,
      cv=cv,
      scoring=scoring,
      n_jobs=n_jobs
  )
    return {
      f"cv_{metric}": results[f"test_{metric}"].mean()
      for metric in scoring.keys()
  }

def evaluate_model_full(model_name, estimator, X_train, y_train,
                        X_valid, y_valid, cv, scoring, approach="baseline",
                         threshold=0.5, n_jobs=-1):
    """
    Perform full model evaluation including cross-validation and validation set.

    The function:
    - Computes cross-validation scores on training data
    - Fits the model on training data
    - Evaluates performance on validation data
    - Extracts model parameters

    Args:
        model_name (str): Name of the model.
        estimator: Scikit-learn estimator or pipeline.
        X_train (array-like): Training features.
        y_train (array-like): Training targets.
        X_valid (array-like): Validation features.
        y_valid (array-like): Validation targets.
        cv: Cross-validation splitter.
        scoring (dict): Dictionary of scoring metrics.
        approach (str, optional): Label for experiment type 
            (e.g., "baseline", "tuned"). Defaults to "baseline".
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

    Returns:
        tuple:
            dict: Evaluation results including CV and validation metrics.
            estimator: Fitted estimator.
    """
    cv_scores = evaluate_model_cv(
      estimator=estimator,
      X=X_train,
      y=y_train,
      cv=cv,
      scoring=scoring
    )

    estimator.fit(X_train, y_train)

    y_proba = estimator.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    result = {
      "model": model_name,
      "approach": approach,
      "threshold": threshold,
      **cv_scores
    }
    
    if "roc_auc" in scoring:
      result["valid_roc_auc"] = roc_auc_score(y_valid, y_proba)
    
    if "average_precision" in scoring:
      result["valid_average_precision"] = average_precision_score(y_valid, y_proba)
    
    if "f1" in scoring:
      result["valid_f1"] = f1_score(y_valid, y_pred)
    
    if "recall" in scoring:
      result["valid_recall"] = recall_score(y_valid, y_pred)
    
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps["model"]
    else:
        model = estimator
      
    result["params"] = model.get_params()
    
    return result, estimator


def evaluate_random_search(model_name, estimator, param_distributions,
                            X_train, y_train, X_valid, y_valid, cv, scoring,
                            n_iter=20, threshold=0.5, n_jobs=-1,
                            random_state=42):
    """
    Perform hyperparameter tuning using RandomizedSearchCV and evaluate the model.

    The function:
    - Runs RandomizedSearchCV to find best hyperparameters
    - Evaluates the best model using cross-validation
    - Evaluates performance on validation data

    Args:
        model_name (str): Name of the model.
        estimator: Scikit-learn estimator or pipeline.
        param_distributions (dict): Parameter distributions for random search.
        X_train (array-like): Training features.
        y_train (array-like): Training targets.
        X_valid (array-like): Validation features.
        y_valid (array-like): Validation targets.
        cv: Cross-validation splitter.
        scoring (dict): Dictionary of scoring metrics.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 20.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple:
            dict: Evaluation results including CV and validation metrics.
            RandomizedSearchCV: Fitted search object.
    """

    search = RandomizedSearchCV(
      estimator=estimator,
      param_distributions=param_distributions,
      n_iter=n_iter,
      scoring="roc_auc",
      cv=cv,
      n_jobs=n_jobs,
      random_state=random_state,
      verbose=1,
      refit=True
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    
    cv_scores = evaluate_model_cv(
      estimator=best_model,
      X=X_train,
      y=y_train,
      cv=cv,
      scoring=scoring
    )
    
    y_proba = best_model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    result = {
      "model": model_name,
      "approach": "random_search",
      "threshold": threshold,
      **cv_scores
    }
    
    if "roc_auc" in scoring:
      result["valid_roc_auc"] = roc_auc_score(y_valid, y_proba)
    
    if "average_precision" in scoring:
      result["valid_average_precision"] = average_precision_score(y_valid, y_proba)
    
    if "f1" in scoring:
      result["valid_f1"] = f1_score(y_valid, y_pred)
    
    if "recall" in scoring:
      result["valid_recall"] = recall_score(y_valid, y_pred)
    
    result["params"] = search.best_params_
    
    return result, search


