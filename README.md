# Bank Marketing Midterm Project

## Project Overview
This project focuses on predicting whether a client will subscribe to a term deposit using machine learning methods.

## Dataset
The dataset used in this project is available on Kaggle:  
https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv  

Download the dataset and place it in the project root directory before running the notebook.

The target variable is binary and indicates whether the client subscribed to a term deposit.

## What Was Done
- exploratory data analysis (EDA)
- data preprocessing
- baseline model training
- hyperparameter tuning (RandomizedSearchCV and Hyperopt)
- model interpretation using SHAP
- error analysis of misclassified observations

## Models Used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- XGBoost

## Results Summary

| Experiment | Model | ROC-AUC | Average Precision | F1 | Recall | Notes |
|----------|------|--------|------------------|----|--------|------|
| Baseline | Logistic Regression | 0.802 | 0.444 | 0.463 | 0.640 | Best baseline model |
| Baseline | XGBoost | 0.791 | 0.452 | 0.473 | 0.609 | Strong baseline |
| Tuning | XGBoost (Random Search + Threshold) | 0.815 | 0.484 | 0.524 | 0.583 | Improved F1 |
| Tuning | XGBoost (Reduced Features) | 0.813 | 0.472 | 0.475 | 0.642 | Better recall |
| Final | XGBoost (Hyperopt + Reduced Features) | 0.812 | 0.473 | 0.475 | 0.645 | Best balance between metrics |

## Final Model
The final selected model is XGBoost tuned with Hyperopt. It demonstrated the best balance of quality metrics and was used for further interpretation and error analysis.

## Conclusions
XGBoost outperformed the baseline models and provided the best predictive performance after hyperparameter tuning. Error analysis showed that the model struggles with borderline cases and observations with similar feature patterns.

## Possible Improvements
- additional feature engineering
- improved handling of categorical variables
- further threshold optimization
- better handling of class imbalance

## Repository Structure
- `bank_marketing.ipynb` — main project notebook  
- `utils.py` — helper functions 
