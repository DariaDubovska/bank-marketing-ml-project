import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

def get_optimal_threshold(y_true, y_probs, metric=f1_score):
  # finding optimal threshold 
  thresholds = np.linspace(0, 1, 100)
  scores = [metric(y_true, (y_probs >= t).astype(int)) for t in thresholds]
  best_idx = np.argmax(scores)
  return thresholds[best_idx], scores[best_idx]

def evaluate_model_cv(estimator, X, y, cv, scoring, n_jobs=-1):
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


