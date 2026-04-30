![](UTA-DataScience-Logo.png)

# Washington House Prices Prediction

This repository holds an attempt to apply XGBoost regression to predict house sale prices in Washington State using tabular real estate data from Kaggle.

## Overview

The task is to predict the final sale price (`lastSoldPrice`) of residential properties in Washington State given features such as square footage, number of bedrooms/bathrooms, year built, property type, and ZIP code. The approach formulates this as a regression problem, using XGBoost as the primary model after establishing a Ridge Regression baseline. The target variable was log-transformed to handle price skewness before training. Our best model achieved an R² of 0.66 and a Mean Absolute Error of ~$105,000 on the held-out test set.

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

Feature d







