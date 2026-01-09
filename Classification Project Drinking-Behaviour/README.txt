
Drinking Behavior Classification
======================================================

----------------
Project Overview
----------------
This classification task uses a dataset from Kaggle:
https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

The dataset was collected from the National Health Insurance Service in Korea. All personal information and sensitive data were excluded.

Total entries: 991,346
Total columns: 24 (including target variable)

The objective of this project is to predict drinking behaviour (Yes / No) using various body signals such as sensory functions, cardiovascular status, kidney and liver functions, blood-based metabolic signals, and so on, by machine learning classification modelling.

The details of the interpretation of the results and content of the codes can refer to as follows.



-------------------
Dataset Description
-------------------
Target Variable:
- DRK_YN: Drinker or Not (Yes / No)

Feature descriptions include demographics, vision, hearing, blood pressure, blood chemistry, liver enzymes, smoking status, and other health indicators.

Detailed description of the dataset can be found in Jupyter notebook, or in the PDF / HTML files.



-----------------
Project Structure
-----------------
code.ipynb:
Complete codes for EDA, feature selection, feature engineering, model training, and evaluation.

drinking_split.pkl:
Dataset used for EDA, feature selection, and engineering.

drinking_finalized.pkl:
Final dataset for model training.

catboost.pkl / decision_tree.pkl / random_forest.pkl / xgboost.pkl:
Saved randomized CV search results, runtime records, and best parameters, for model evaluation and final test.

To use the codes without training model, please run the pkl files directly. Detailed comments can be found in the code.ipynb.



----------------------
Data Preparation & EDA
----------------------
Dataset split before EDA to avoid data leakage:
- Train: 594,807 rows (60%)
- Validation: 198,269 rows (20%)
- Test: 198,270 rows (20%)

- Univariate analysis for target variable, categorical variables and numeric variables.
- Bivariate analysis with target feature for numeric and categorical variables were conducted one by one separately using boxplots, kernel density estimation plots and barplots separately, for inspecting the relationship between input variables and target variable carefully.

EDA findings:
- Balanced target distribution
- Many numeric outliers
- Most numeric features are non-normally distributed



-----------------
Feature Selection
-----------------
Methods used:
- Spearman correlation (for numeric only due to high amount of outliers)
- ANOVA for categorical target and continuous predictors
- Cramér’s V for categorical target and nominal variables
- Spearman correlation for ordinal variables

Five variables were dropped based on statistical significance (F-statistics, higher, better) and correlation strength.



-------------------
Feature Engineering
-------------------
Log transformation applied to skewed numeric features.
Same transformations applied consistently across train, validation, and test sets.



---------------------------
Model Training & Evaluation
---------------------------
Models used:
- Decision Tree (baseline)
- Random Forest
- XGBoost
- CatBoost

Randomized cross validation search was applied to minimize the computational cost.

Neural networks were not considered, as they typically underperform tree-based ensembles on structured tabular data because of limited feature dimensionality and frequency of data, and require significantly greater tuning and computational resources without clear performance gains.

ROC-AUC used as the primary evaluation metric because it evaluates the model’s ability to discriminate between drinkers and non-drinkers across all decision thresholds. Balanced accuracy was additionally reported to assess classification fairness at a fixed threshold while ensuring equal importance of both classes. Accuracy, precision, and recall were used for optional model references because they depend on an arbitrary cutoff but not reflect ranking performance.


Decision Tree (Baseline)
------------------------
TRAIN ROC-AUC: 0.8140
VALID ROC-AUC: 0.8083

Best Parameters: {'min_samples_split': 1000, 'min_samples_leaf': 500, 'max_depth': 13, 'criterion': 'entropy'}

- CART-based decision tree classifier was used as the baseline model.
- To control overfitting, strong pre-pruning regularization was applied through constraints on maximum depth, minimum samples per split, and minimum samples per leaf.
- The optimal model, selected via 5-fold stratified cross-validation, used entropy as the splitting criterion with a maximum depth of 13, requiring at least 1000 samples to split a node and 500 samples per leaf.
- Resulted in stable generalization performance.
- Computational cost is low (1.5 mins).
- Visualizations of performance v.s. depths and tree plots.

Interpretations of the visualizations of the model:

- As tree depth increased, model performance improved rapidly at shallow depths and then stabilized which indicated that most predictive structure was captured by moderate model complexity. 
- Validation ROC-AUC increased from approximately 0.78 at depth 4 to around 0.81 by depth 10 – 11, after which further increases in depth did not yield meaningful improvement, with values fluctuating above or below within a narrow range of 0.002. 
- A similar pattern was observed for BA, which rose from about 0.69 at depth 3 to approximately 0.73 at depth 10, and then plateaued, which means that deeper trees did not improve fair classification across classes. 
- Overall Accuracy followed the same trend, reaching roughly 0.73 and remaining stable thereafter. 
- Training and validation curves closely overlapped across all three metrics, with no divergence at higher depths, which demonstrates effective regularization and the absence of overfitting. 
- In short, these results showed that increasing tree depth beyond approximately 10 – 12 levels did not provide additional discriminative or classification benefits, and that the decision tree achieved stable generalization performance with ROC-AUC around 0.81 and BA around 0.73.


XGBoost
-------
TRAIN ROC-AUC: 0.8272
VALID ROC-AUC: 0.8205

Best Parameters: {'subsample': 0.9, 'reg_lambda': 5, 'reg_alpha': 0.5, 'n_estimators': 400, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.5, 'colsample_bytree': 0.7}

- Captured complex non-linear interactions through gradient-boosted decision trees.
- Run time is short (4.5 mins).
- Strongly regularized using constraints on tree depth, minimum child weight, split gain (gamma), feature subsampling, and L1/L2 penalties.
- Hyperparameter tuning moderated learning rates combined with split regularization yielded optimal performance, minimized train–validation gap and stable generalization.
- Compared with different parameters e.g. learning rates and gamma, generated slightly different results, but more or less similar.

Interpretations of the visualizations of the model:

- Cross-validated ROC-AUC increased substantially as the learning rate rose from 0.01 around 0.809 to 0.05 around 0.819 and then stabilized, with only marginal gains observed at higher learning rates which implies that diminishing returns beyond moderate step sizes.
- Analysis of misclassified cases showed that both false positives and false negatives were concentrated near the decision boundary which is the predicted probabilities close to 0.5, and thus the errors primarily arose from the overlap in predictors distributions rather than systematic model bias.


CatBoost
--------
TRAIN ROC-AUC: 0.8313
VALID ROC-AUC: 0.8210

- Advanced gradient boosting model leveraging ordered boosting and stochastic sampling to reduce prediction shift and overfitting.
- The optimal configuration which are a moderate learning rate with deeper trees and strong regularization.
- Compared to XGBoost, CatBoost provided marginal but consistent performance improvements which means that the dataset benefits from its handling of feature interactions and regularization strategy, but the computation time is a lot longer.
- Compared with different parameters generated slightly wore results, but more or less similar.
- Long run time around 38 mins.

Interpretations of the visualizations of the model:

- Analysis of misclassified cases also showed that both false positives and false negatives were concentrated near the decision boundary which is the predicted probabilities close to 0.5, and thus the errors primarily arose from the overlap in predictors distributions rather than systematic model bias.
- Threshold sensitivity analysis further showed that BA peaked near a threshold of 0.5, reaching approximately 0.74, and declined symmetrically as the threshold moved away from this value, which demonstrates stable performance across a moderate threshold range.


Random Forest
-------------
TRAIN ROC-AUC: 0.8181
VALID ROC-AUC: 0.8141

Best parameters: {'n_estimators': 500, 'min_samples_split': 300, 'min_samples_leaf': 100, 'max_features': 0.3, 'max_depth': 25}

- A bagging-based ensemble to reduce variance relative to a single decision tree.
- Subsample used to reduce the run time (total required for 45 mins).
- Bootstrap sampling and feature subsampling used to decorrelate trees, and out-of-bag estimation provided an internal generalization check.
- Compared with different parameters generated slightly wore results, but more or less similar.

Interpretations of the visualizations of the model:

- The out-of-bag (OOB) ROC-AUC was notably lower at approximately 0.7325, which was expected given that OOB predictions are based on individual trees trained on bootstrap samples rather than the full ensemble and therefore provide a conservative, approximate estimate of generalization performance.
- Validation ROC-AUC increased from approximately 0.8137 with 100 trees to 0.8141 with 300 trees, with only a gain to around 0.8143 at 500 trees which means that ensemble diversity was largely saturated beyond 300 estimators.
- Variation in max_features had minimal impact on performance with mean cross-validated ROC-AUC remaining close to 0.81 across values of 0.2, 0.3, and sqrt which shows the robustness to feature subsampling choices.


----------------
Model Comparison
----------------
Across all evaluated models, XGBoost consistently achieved near-top predictive performance while maintaining strong generalization. 

Although CatBoost achieved a slightly higher ROC-AUC, the improvement over XGBoost was marginal on the order of 0.0005 to 0.001. 

It indicates that both models operate near the performance ceiling for the given feature set. 

The train–validation curves (can be found of the jupyter notebook) show that XGBoost demonstrates a small and stable generalization gap across all metrics. 

It shows that XGBoost achieves an effective bias–variance trade-off and has improved upon Random Forest and Decision Tree while remaining as stable as CatBoost. 

Regarding the computational cost, in large-scale settings (600k training samples), the efficiency difference is practically significant. 

Referring the feature importance, sex and age were the most influential predictors in the Decision Tree and Random Forest, while XGBoost relied more heavily on biochemical markers, and overall the averaged importance scores confirmed sex, BLDS_log, age, and gamma-GTP as the most influential predictors overall, while other variables contributed only on average.

Thus, XGBoost offers a more favorable balance between accuracy, interpretability, and efficiency. 



------------------
Final Test Results
------------------
XGBoost selected as the final model. Misclassification confidence plot was produced. 

TRAIN ROC-AUC: 0.8272
TEST ROC-AUC: 0.8199

The final XGBoost model demonstrates strong generalization on the held-out test set with minimal performance degradation relative to training and validation results. 

Precision and recall are well balanced across both classes which means no systematic classification bias. 

Analysis of prediction confidence reveals that most misclassifications occur near the decision threshold. 

There are well-calibrated probability estimates and limited high-confidence errors. 

In short, these results confirm the robustness and reliability of the selected model.
