<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/756124c0-6220-499f-a457-0c0496e66b8e" /># FPS_Predictor
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

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/71cc0889-74bd-4fb1-8fc4-0f6181aa68f5" />

## Key Observations

1. Non-Linearity Dominance: Tree-based ensemble models (Random Forest, $R^2 = 0.8400$) vastly outperform linear and polynomial variants due to their capacity to map abrupt rendering performance drops and hardware floors.

2. Polynomial Failure: Feature space expansion (Polynomial Ridge) underperformed standard baselines due to polynomial coefficient blowups at large polygon counts (up to 250,000).
