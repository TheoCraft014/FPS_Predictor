# FPS_Predictor
Evaluate machine learning regression models to predict Frame Rate (FPS) in Unity-rendered 3D scenes based on various rendering load settings.

## Motivation
Real-time 3D applications must maintain frame rates above specific thresholds to deliver smooth user experiences. Rendering performance depends on multiple interacting parameters, such as polygon count, lighting, particle effects, and global quality presets. Manually profiling every configuration is time-consuming and computationally expensive.  
This project implements a predictive framework $f: D \to \text{FPS}$ using machine learning algorithms to estimate FPS across unseen rendering configurations before executing full rendering sweeps.

## Dataset
Was generated using a custom Unity script designed to run systematic sweeps across rendering parameters while capturing frame rate measurements.  Training Set (FPS_Combinedx6.csv): 72,576 rows across 12,096 unique rendering configurations.  Hidden Test Set (FPS_hidden.csv): 17,280 rows reserved as an untouched test set featuring unseen polygon/particle grid points to measure true model generalization. 

## Features

FeatureTypeValues / RangeDescriptionQualityCategorical0, 1, 2, 3, 4, 5Unity's global quality preset  PolygonsContinuous0 to 250,000Polygon count in the scene  LightsContinuous0, 1, 2, 4, 8, 12, 16Active light sources  ShadowsBoolean0 or 1Shadow map rendering toggle  ParticlesContinuous0 to 3,000Active particle system count  IsComplexMaterialBoolean0 or 1High-complexity shader flag  FPS (Target)Continuous0.56 to 158.21Recorded frames per second

## Preprocessing and validation

Grouped Cross-Validation (GroupKFold):
To prevent data leakage across cross-validation splits, a group_id is computed from the unique combination of the input features. GroupKFold(n_splits=5) guarantees that all duplicate entries for a given configuration belong exclusively to either the training fold or the validation fold.

## Used Models and optimal hyperparameter result

The pipeline evaluates baseline models alongside hyperparameter-tuned advanced regressors:

1. Lifted Linear RegressionPipeline: StandardScaler $\to$ PolynomialFeatures(degree=2) $\to$ LinearRegression   

2. Random Forest RegressorPipeline: RandomForestRegressor tuned via GridSearchCV using GroupKFold  Best Params: max_depth: 10, max_features: 1.0, min_samples_leaf: 4, n_estimators: 300 

3. Decision Tree RegressorPipeline: DecisionTreeRegressor tuned via GridSearchCV  Best Params: criterion: 'absolute_error', max_depth: 10, min_samples_leaf: 1, min_samples_split: 

4. Polynomial Ridge RegressionPipeline: One-hot encoded quality presets $\to$ Degree-3 polynomial expansion $\to$ L2-regularized Ridge regression. 

## Results
All models were evaluated on the untouched hidden test set (FPS_hidden.csv):

ModelHidden R2Hidden RMSE (FPS)Hidden MAE (FPS)Median AE (FPS)Hidden MAPEBase Lifted Linear Regression  0.4286  25.04  18.20  13.26  247.78%  Polynomial Ridge Regression  0.2340  29.00  17.47  —273.28%  Support Vector Regression (SVR)  0.8093  14.47  9.35  5.52  95.59%  Decision Tree Regressor  0.8236  13.92  8.26  3.94  49.83%  Random Forest Regressor (Best)  0.8400  13.25  7.73  3.64  48.29%

## Key Observations

1. Non-Linearity Dominance: Tree-based ensemble models (Random Forest, $R^2 = 0.8400$) vastly outperform linear and polynomial variants due to their capacity to map abrupt rendering performance drops and hardware floors.

2. Polynomial Failure: Feature space expansion (Polynomial Ridge) underperformed standard baselines due to polynomial coefficient blowups at large polygon counts (up to 250,000).
