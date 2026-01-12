# Predicting Tsunamis using Machine Learning

## Project Overview
This project focuses on predicting whether an earthquake will generate a tsunami using seismic and geospatial features.  
The task is formulated as a **binary classification problem**, aiming to support early-warning decision systems with data-driven models.

## Dataset
The dataset contains historical earthquake events with:
- Seismic features (magnitude, depth, intensity)
- Geographic coordinates (latitude, longitude)
- Event significance and impact indicators
- Binary target label: tsunami (0/1)

Data preprocessing includes:
- Missing value handling
- Feature scaling
- Train/validation/test splitting
Link dataset: "https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset"
## Exploratory Data Analysis (EDA)
Key analyses include:
- Statistical distribution of seismic features
- Correlation analysis between variables
- Geographic clustering of tsunami-prone regions
- Magnitude–depth interaction patterns

Insights from EDA guide feature engineering and model selection.

## Feature Engineering

### 1. Physics-Informed Seismic Features
- Energy-related transformations of magnitude
- Depth-weighted intensity measures

### 2. Spherical Earth Representation
- Conversion of latitude/longitude into 3D Cartesian coordinates
- Preserves spatial relationships on Earth’s surface

### 3. Adaptive Spherical RBF Encoding
- Radial Basis Functions to model regional seismic patterns
- Helps capture localized tsunami risk zones

## Models Evaluated
Multiple classifiers were compared under the same experimental setup:
- Logistic Regression
- Decision tree
- Random Forest
- Gradient Boosting
- XGBoost
- AdaBoost
- LightGBM
- SVM

Evaluation metrics:
- Accuracy
- F1-score, F2-score
- Confusion Matrix

## Hyperparameter Optimization
- Grid Search on Random Forest model.
- Optimization focused on F2-Score.

## Feature Selection & Interpretability

### Feature Importance
- Mean Decrease in Impurity (MDI)
- SHAP (SHapley Additive exPlanations)

### Explainability Goals
- Identify key seismic and spatial drivers of tsunami risk
- Provide both global and local explanations

Low-impact features were pruned to reduce model complexity.

## Decision Threshold Optimization
Instead of using the default 0.5 threshold:
- Threshold tuned to maximize F1-score
- Better balance between false positives and false negatives

## Final Evaluation on Test Set
Final model evaluated using:
- Accuracy: 0.95
- F2-score: 0.96
- Precision: 0.9
- Recall: 0.98

Results show improved recall for tsunami events while maintaining acceptable precision.

