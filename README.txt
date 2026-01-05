House Prices Regression Project (Ames Housing Dataset) – README

Hello Professor,
This folder contains my full project for the Kaggle competition “House Prices: Advanced Regression Techniques”.
I tried to keep the work clean and modular, so each notebook has a clear role, and I also exported each notebook to PDF for easier reviewing.

────────────────────────────────────────────────────────────────────────────
1) Project Goal
────────────────────────────────────────────────────────────────────────────
The goal is to predict house sale prices (SalePrice) using the Ames Housing dataset.
I focus on:
- data cleaning and preprocessing
- feature engineering and correlation checks
- feature selection (LASSO) and model training (tree-based models, XGBM,CatBoost,LightGBM)
- producing a final Kaggle-style submission file

────────────────────────────────────────────────────────────────────────────
2) Files Included
────────────────────────────────────────────────────────────────────────────
Notebooks (and their PDF versions):
- 1- regression.ipynb  +  1- regression.pdf
  Purpose: initial data loading, missing-value handling, skewness handling, and building a preprocessing pipeline.
  Output: a prepared dataset saved for reuse.
- Loads train.csv / test.csv (Kaggle data)  
- Separates target (SalePrice) and features  
- Handles missing values, categorical encoding, and scaling using Pipeline + - ColumnTransformer  
- Applies log transform for the target / skewed numeric features (where appropriate)  
- Saves the processed splits and metadata as prepared_data.pkl for reuse

- 2- regression.ipynb  +  2- regression.pdf
  Purpose: feature engineering and feature quality checks (especially correlation analysis and multicollinearity checks).
  Output: an updated prepared dataset saved for modeling.
- Loads prepared_data.pkl  
- Performs correlation analysis with the target (to understand strongest predictors)  
- Checks for redundancy / multicollinearity patterns in numeric features  
- Adds engineered features (e.g., totals, ages, interactions/ratios, “has feature” indicators)  
- Performs sanity checks on encodings (e.g., verifying binary features match their meaning)  
- Saves an updated prepared dataset for modeling (depending on your notebook settings, it may overwrite or create a new pickle)

- 3-regressionC.ipynb  +  3-regressionC.pdf
  Purpose: modeling stage. Includes feature selection using LASSO and training/evaluating models (and creating predictions).
- Trains and evaluates multiple regression models  
- Uses LASSO (and related linear models) for feature selection / shrinkage  
- Trains stronger non-linear models (e.g., XGBoost, LightGBM, CatBoost) and compares performance  
- Produces the final submission file submission_xgboost.csv

Data / saved artifacts:
- train.csv
- test.csv
- prepared_data.pkl
  A saved “prepared data” object used to avoid repeating preprocessing across notebooks.
- submission_xgboost.csv
  Final submission file with two columns: Id and SalePrice.

────────────────────────────────────────────────────────────────────────────
3) How to Run the Project (Recommended Order)
────────────────────────────────────────────────────────────────────────────
Please run the notebooks in this order:

1) 1- regression.ipynb
2) 2- regression.ipynb
3) 3-regressionC.ipynb

Each notebook saves outputs that the next notebook loads, so the order matters.

If you want you can use the PDF files from the notebooks:
1- regression.pdf
2- regression.pdf
3-regressionC.pdf

────────────────────────────────────────────────────────────────────────────
4) Notes About Evaluation and Reproducibility
────────────────────────────────────────────────────────────────────────────
- I tried to avoid data leakage as much as possible. For example, whenever I needed to learn something from the data (imputation values, scaling parameters, encoding, etc.), I fit those steps on the training data and then applied the same fitted transformations to validation and test.

- I used structured preprocessing (pipelines / transformers) so the exact same steps are applied consistently. This helps keep the workflow reproducible and reduces the chance of “accidental” differences between train/val/test processing.

- In notebook 2, I do feature engineering based on correlation checks and sanity checks (for example I verified suspicious correlations such as CentralAir and fixed the encoding when needed). I also check for highly correlated features (multicollinearity) to reduce redundancy before modeling.

- In notebook 3, I focus on modeling:
  • I work in log-space (predicting the log of SalePrice) and convert predictions back to the original scale at the end, because house prices are skewed.
  • I evaluate models using cross-validation (K-Fold) so the score is more reliable than a single split.
  • For boosting models, I use early stopping during CV to reduce overfitting and choose a good number of trees.
  • I compare multiple models fairly (same folds / same metric) including regularized linear models and tree-based models like XGBoost, LightGBM, and CatBoost.
  • I apply feature selection using LASSO to keep only the most useful predictors and reduce noise. The selected feature set is then reused for each models and in the final training step for the chosen model.
  • After choosing the best-performing model, I refit it on the combined training + validation data and generate the final test predictions and the submission file (`submission_xgboost.csv`).

- I set random seeds (where possible) and keep the workflow split into 3 notebooks so it is easy to rerun, debug, and review each stage separately.


────────────────────────────────────────────────────────────────────────────
5) Data Source
────────────────────────────────────────────────────────────────────────────
Dataset: Kaggle – House Prices: Advanced Regression Techniques
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Thank you for reviewing my project!
