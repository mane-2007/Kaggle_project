![](UTA-DataScience-Logo.png)

# Washington House Prices Prediction

This repository holds an attempt to apply XGBoost regression to predict house sale prices in Washington State using tabular real estate data from Kaggle.

## Overview

The task is to predict the final sale price (`lastSoldPrice`) of residential properties in Washington State given features such as square footage, number of bedrooms/bathrooms, year built, property type, and ZIP code. The approach formulates this as a regression problem, using XGBoost as the primary model after establishing a Ridge Regression baseline. The target variable was log-transformed to handle price skewness before training. The best model achieved an R² of 0.66 and a Mean Absolute Error of ~$105,000 on the held-out test set.

## Summary of Workdone

### Data

* **Type:** CSV file of tabular real estate listing features; target is a continuous numerical value (sale price)
* **Size:** 12,017 rows, 15 features before cleaning
* **Split:** 7,075 training / 1,769 validation / 2,211 test

#### Preprocessing / Clean up

* Dropped columns that would cause data leakage: `listPrice`, `list_to_sold_ratio`, `price_per_sqft`, `baths_full`, and `sanitized_text` — these were either derived from the target or not useful for generalization
* Dropped the 15 rows where `lastSoldPrice` was null (target variable)
* Filled `garage` missing values with 0, assuming no garage when not listed (over 80% missing)
* Filled remaining missing numerical values with the column median to preserve typical values
* Removed outliers using IQR (1.5x multiplier) on `lastSoldPrice` to reduce the influence of extreme luxury properties that were not representative of the general market
* One-hot encoded `type` (property type) and ZIP code at two levels: full ZIP and 3-digit regional prefix to capture both neighborhood and broader region signal
* Applied StandardScaler to numerical features — fit only on training data to prevent data leakage into validation and test sets

#### Data Visualization

Feature distributions were compared before and after cleaning to confirm outlier removal and imputation worked as expected. After cleaning, histograms were compared across three price tiers (Low / Mid / High) to identify which features carried the most signal. Square footage showed the clearest separation between tiers, followed by year built and number of bathrooms. A correlation heatmap on the raw data confirmed which features were leaking information about the target.

<img width="1587" height="1418" alt="image" src="https://github.com/user-attachments/assets/0e76a900-eaa2-42f1-afdc-c4404e4a426e" />
<img width="1587" height="1418" alt="image" src="https://github.com/user-attachments/assets/f8b69fc2-8323-42a0-b339-efc99973fc6b" />
<img width="878" height="784" alt="image" src="https://github.com/user-attachments/assets/59a76715-1c49-47ad-a467-e962772feec7" />




### Problem Formulation

* **Input:** Numerical and categorical features describing a residential property (sqft, beds, baths, year built, stories, garage, ZIP code, property type)
* **Output:** Predicted sale price in dollars
* **Models:**
  * *Ridge Regression* — used as a baseline linear model. Achieved R² of 0.62 and MAE of ~$113,000. Limited by its assumption of linear relationships between features and price.
  * *XGBoost Regressor* — selected as the primary model because house prices have non-linear relationships with features (e.g. price doesn't scale linearly with sqft). XGBoost captures these patterns through ensembled decision trees and handles the mix of sparse one-hot encoded columns and dense numerical features well.
* **Target transformation:** `np.log1p` applied to sale price before training to reduce skewness; predictions converted back with `np.expm1`
* **Hyperparameters:** `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=20`

### Training

* Trained locally using Python / Jupyter Notebook
* XGBoost training completed in under 2 minutes on CPU
* Early stopping monitored validation MAE — training halted automatically when no improvement was seen for 20 consecutive trees, preventing overfitting
* No major difficulties after fixing a data leakage issue where the scaler was initially fit on the full dataset before splitting

### Performance Comparison

Primary metrics: R² (proportion of variance explained) and MAE (average dollar error)

| Model | Validation R² | Validation MAE | Test R² | Test MAE |
|---|---|---|---|---|
| Ridge Regression | 0.6820 | $113,156 | 0.6287 | $112,559 |
| XGBoost | 0.7338 | $105,701 | 0.6648 | $105,205 |

XGBoost outperformed Ridge on both metrics. The consistency between validation and test scores indicates the model generalizes well and is not overfitting.

### Conclusions

* XGBoost meaningfully outperforms linear regression for house price prediction, confirming that the relationship between property features and sale price is non-linear
* Square footage, year built, and number of bathrooms were the most informative features
* Geographic encoding at two levels (ZIP + region) added useful location signal
* A ~$105k average error on a median home price of ~$568k represents roughly 18% average error — reasonable for a single-model baseline without feature engineering

### Future Work

* Engineer additional features such as price per sqft by ZIP code or age of home at time of sale
* Experiment with lower `max_depth` to close the gap between validation and test R²
* Try ensemble stacking combining Ridge and XGBoost predictions
* Incorporate the `sanitized_text` listing description column using NLP features (TF-IDF or embeddings) as additional input

## How to Reproduce Results

1. Clone this repository
2. Download the dataset from Kaggle (link below) and place `washington_ultimate.csv` in the root directory
3. Install required packages (see Software Setup)
4. Run `real_estate.ipynb` top to bottom — all steps from loading to evaluation are contained in a single notebook

### Overview of Files in Repository

* `real_estate.ipynb` — main notebook containing all steps: data loading, cleaning, visualization, model training, and evaluation
* `washington_ultimate.csv` — raw dataset (download from Kaggle separately)
* `README.md` — this file

### Software Setup

Required packages:
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
Install all with: pip install pandas numpy matplotlib seaborn scikit-learn xgboost

### Data

* Dataset: [Washington State Real Estate — Kaggle](https://www.kaggle.com/datasets/kanchana1990/washington-real-estate-sold-properties-data-2026)
* No additional preprocessing steps needed beyond running the notebook

### Training

* Open `real_estate.ipynb` in Jupyter Notebook or JupyterLab
* Run all cells in order
* XGBoost training cell will print progress every 50 trees and stop automatically via early stopping

#### Performance Evaluation

* Validation and test metrics (R² and MAE) are printed at the end of the notebook
* To evaluate on new data, apply the trained `model` object and `scaler` to your feature matrix following the same column structure

## Citations

* [Washington State Real Estate dataset — Kaggle](https://www.kaggle.com/datasets/kanchana1990/washington-real-estate-sold-properties-data-2026)
* XGBoost documentation: https://xgboost.readthedocs.io
* Scikit-learn documentation: https://scikit-learn.org
