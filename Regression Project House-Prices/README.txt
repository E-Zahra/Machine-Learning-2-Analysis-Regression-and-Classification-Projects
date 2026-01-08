House Prices Regression Project (Ames Housing Dataset) – README
=======================================================

Hello Professor,
This folder contains my full regression project for“House Prices: 
Advanced Regression Techniques”. I tried to keep everything
clean and modular, so each notebook has a clear role and the outputs from one
step are reused in the next step.

────────────────────────────────────────────────────────────────────────────
1) Project Goal
────────────────────────────────────────────────────────────────────────────
Goal: predict house sale prices (SalePrice) from the Ames Housing dataset.

Main focus of my work:
- build pipeline (cleaning → preprocessing → features → modeling)
- avoid data leakage and compare models fairly
- reduce overfitting (because the dataset is not huge but has many features)

Evaluation metric:
- log-RMSE (Kaggle-style, done in log-space because prices are skewed)

────────────────────────────────────────────────────────────────────────────
2) Dataset Details (Shapes)
────────────────────────────────────────────────────────────────────────────
Raw Kaggle files:
- train.csv: 1460 rows × 81 columns (includes target SalePrice)
- test.csv : 1459 rows × 80 columns (no SalePrice)

After separating target from features:
- X (features) in train: 1460 rows × 80 columns

Train/Validation split used in the notebooks (80/20):
- X_train: 1168 × 80
- X_val  :  292 × 80

After preprocessing + encoding (Notebook 1):
- X_train_preprocessed: 1168 × 200
- X_val_preprocessed  :  292 × 200

After feature engineering cleanup (Notebook 2) and saving again:
- modeling matrix used in Notebook 3 starts at: 1168 × 208 (train), 292 × 208 (val), 1459 × 208 (test)

After feature selection (LASSO in Notebook 3):
- selected features: 58
- final shapes for modeling:
  • full train: 1460 × 58
  • full test : 1459 × 58

────────────────────────────────────────────────────────────────────────────
3) Files Included (What is inside this folder)
────────────────────────────────────────────────────────────────────────────
Notebooks (main work):
1) 1- PreproccesingRegression.ipynb
2) 2- FeatureEngRegression.ipynb
3) 3-ModelingRegression.ipynb

Exports for easier viewing (no running needed):
- PDF and HTML exports of each notebook are included in the GitHub folder
  (same names as the notebooks, but .pdf and .html).

Data / artifacts:
- train.csv, test.csv
- prepared_data.pkl (saved object used to reuse preprocessing/feature engineering)
- submission_xgboost.csv (final prediction file: Id + SalePrice)

────────────────────────────────────────────────────────────────────────────
4) How to Run the Project (Step-by-Step)
────────────────────────────────────────────────────────────────────────────
Option A — Reproduce results by running the notebooks:

(1) Download/clone the repository and go to:
    Regression Project House-Prices/

(2) Run notebooks in THIS order (important because outputs are reused):
    1) 1- PreproccesingRegression.ipynb
    2) 2- FeatureEngRegression.ipynb
    3) 3-ModelingRegression.ipynb

Option B — Only review (no code execution):
- open the exported PDF files (or download the HTML files and open them in a browser)

────────────────────────────────────────────────────────────────────────────
5) What I Did in Each Notebook (Steps + Outputs)
────────────────────────────────────────────────────────────────────────────

Notebook 1 — Preprocessing (1- PreproccesingRegression.ipynb)
-------------------------------------------------------------
Main steps:
- Load train.csv and test.csv
- Separate target (SalePrice) and apply log transformation to target (log1p)
- Split train into train/validation (80/20 → 1168 train, 292 val)
- Handle missing values using consistent rules (so train/val/test are treated the same)
- Fix data types and clean categories (so encoding works correctly)
- Treat skewness in numeric features (log / shifted-log for very skewed columns)
- Identify binary columns and one-hot encode nominal categorical features
- Make sure no missing values remain at the end (train/val/test)

Key output/result:
- Final encoded feature matrices with 200 columns
- Saved the prepared data object to prepared_data.pkl for reuse in later notebooks

Notebook 2 — Feature Engineering + Correlation Checks (2- FeatureEngRegression.ipynb)
-----------------------------------------------------------------------------------
Main steps:
- Load prepared_data.pkl
- Check feature types again + sanity checks for encoded/binary variables
  (example: verifying that binary encoding matches price differences)
- Detect and remove an unexpected non-numeric column (“Condition2”) after encoding
- Engineer new features to capture stronger patterns:
  Examples of engineered ideas included:
  - TotalArea, TotalBath, TotalPorch
  - Age-related features (HouseAge / RemodAge / GarageAge)
  - Quality × size features (like QualTotalArea / QualGrLivArea)
  - Density / indicator features (like RoomDensity, HasFireplace)
- Drop features that were constant or too weak (low usefulness / redundancy)
- Re-check correlations after engineering (to confirm the new features add signal)

Key output/result:
- Final engineered dataset used for modeling starts with 208 features
- Saved the updated prepared_data.pkl for the modeling notebook

Notebook 3 — Modeling + Evaluation + Prediction (3-ModelingRegression.ipynb)
---------------------------------------------------------------------------
Main steps:
- Load the updated prepared_data.pkl
- Feature selection using LASSO (to reduce noise and overfitting):
  → selected 58 final features
- Train and compare strong tree-based models using 5-fold CV with log-RMSE:
  • XGBoost
  • LightGBM
  • CatBoost
  • Random Forest
  • Ensemble = average of the 3 boosting models

Extra evaluation focus (not only “lowest error”):
- Overfit gap = (Validation log-RMSE − Train log-RMSE)
- Validation standard deviation (stability across folds)
- Residual analysis (residual plots/histograms and “worst error” examples)
  In the residual plots, XGBoost and the ensemble looked more compact/centered
  compared to models that showed wider spreads.

Cross-validation summary (mean values):
Model                        Train log-RMSE  Val log-RMSE      Gap   Val Std
----------------------------------------------------------------------------
CatBoost                           0.0753       0.1209   0.0456   0.0163
Ensemble (avg XGB+LGBM+CAT)        0.0876       0.1236   0.0359   0.0170
XGBoost                            0.1046       0.1302   0.0257   0.0183
LightGBM                           0.0952       0.1304   0.0352   0.0176
Random Forest                      0.0687       0.1384   0.0697   0.0170

Model choice (final):
- CatBoost and the ensemble had noticeably low validation scores and low validation std,
  but their overfit gaps were still larger than XGBoost.
- Random Forest had the largest overfit gap (strong memorization + weaker generalization).
- LightGBM also showed noticeable overfitting and not-the-best fold stability.
- XGBoost gave the most balanced result: smallest + most stable overfit gap.
  So I chose XGBoost as the final model.

Final output/result:
- Fit the final XGBoost model on the full training data (1460 rows, 58 selected features)
- Predicted the test set (1459 rows)
- Saved predictions to: TestPredictions_xgboost.csv (Id, SalePrice)

────────────────────────────────────────────────────────────────────────────
6) Notes About Reproducibility / Good Practice
────────────────────────────────────────────────────────────────────────────
- I tried to prevent data leakage by fitting preprocessing steps on training data and
  applying the same fitted transformations to validation and test.
- I used structured preprocessing so the workflow stays consistent across splits.
- I used 5-fold cross-validation to avoid relying on one split, and compared models
  using the same metric and folds.
- I also looked at overfit gap + stability (std) to choose a model that generalizes better.

────────────────────────────────────────────────────────────────────────────
7) Data Source
────────────────────────────────────────────────────────────────────────────
Kaggle competition: House Prices – Advanced Regression Techniques
(Train/Test files come directly from this competition.)

Thank you for reviewing my project!
