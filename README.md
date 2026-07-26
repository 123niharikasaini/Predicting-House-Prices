# California House Price Prediction
 
A beginner machine learning project that predicts median house prices in California using the built-in `sklearn` California Housing dataset. The project walks through the full regression workflow: EDA, outlier treatment, feature correlation analysis, model training, evaluation, and diagnostic residual analysis.
 
## Dataset
 
The dataset is loaded directly from `sklearn.datasets.fetch_california_housing`. It contains 20,640 rows describing California census block groups, with features such as median income, house age, average rooms/bedrooms, population, average occupancy, and location (latitude/longitude). The target variable is the median house value for the block group (in units of $100,000), which is capped at 5.00001 (i.e. $500,001+).
 
## Project Workflow
 
### 1. Data Preparation
- Loaded the dataset and combined the feature matrix with the target into a single DataFrame.
- Reviewed descriptive statistics (`.describe()`) to understand feature scales and ranges.
### 2. Exploratory Data Analysis (EDA)
- Used `.corr()` and boxplots to inspect feature relationships and spot outliers.
- Identified apparent multicollinearity between `AveRooms` and `AveBedrms`, and between `Latitude` and `Longitude`.
### 3. Outlier Treatment
- Used the IQR method to detect outliers in each numeric feature.
- Capped (rather than removed) extreme values in `AveRooms`, `AveBedrms`, `Population`, and `AveOccup` using `.clip()`, since these are ratio-based features prone to distortion when a block group has very few households.
- Left `MedInc` and `HouseAge` untouched, since their extreme values represent plausible real-world variation rather than data artifacts.
- **Re-ran correlation analysis after cleaning** — the `AveRooms`/`AveBedrms` correlation dropped from 0.85 to 0.18 once outlier artifacts were removed, showing the original high correlation was distorted by a handful of rows with unreliable ratios. No columns were dropped as a result.
### 4. Preprocessing
- Split the data into training and test sets (`train_test_split`, `test_size=0.3`) **before** scaling, to avoid data leakage.
- Applied `StandardScaler`, fitting only on the training set and transforming both sets.
### 5. Modeling
- Trained a baseline `LinearRegression` model on the cleaned, scaled data.
### 6. Evaluation
 
| Metric | Before outlier treatment | After outlier treatment |
|---|---|---|
| RMSE | — | 0.6664 |
| MSE | 0.5306 | 0.4441 |
| MAE | 0.5384 | 0.4925 |
| R² | 0.5958 | 0.6617 |
| Adjusted R² | 0.5953 | 0.6612 |
| MAPE | — | ~30% |
 
Outlier capping alone improved R² by roughly 6.6 percentage points, showing that a large share of the model's original error came from a small number of distorted feature values rather than the model itself.
 
### 7. Diagnostics
 
**Residual analysis** (residuals vs. predicted values) revealed two distinct patterns:
- A **funnel/cone shape**, indicating heteroscedasticity — error size grows with predicted price.
- A **straight diagonal line**, caused by the dataset's price ceiling ($500,001+ homes are all recorded as exactly 5.00001), which is a structural artifact of the data and not fixable through modeling choices.
**Train vs. test RMSE** were nearly identical (0.6676 vs. 0.6664), indicating the model is **not overfitting** — the performance ceiling is a bias (underfitting) problem, not a variance problem.
 
### 8. Regularization (Ridge & Lasso)
- Applied `Ridge` and `Lasso` regression as a check against overfitting.
- Results confirmed the train/test diagnosis: Ridge produced virtually no change (RMSE ≈ 0.666), and Lasso performed worse (RMSE ≈ 0.769) by zeroing out coefficients that still carried useful signal.
- **Conclusion:** regularization does not help here, since there is no meaningful overfitting for it to correct.
### 9. Log-Transform Experiment
- Applied `np.log1p()` to the target to test whether it would resolve the heteroscedasticity seen in residuals.
- Result: RMSE and R² both got slightly **worse** (RMSE 0.719, R² 0.606) once predictions were converted back to price scale (`np.expm1()`). MAE improved marginally.
- **Reason:** the exponential inverse-transform amplifies errors on the price-capped rows, worsening their contribution to squared-error metrics even though it may have improved relative error for typical (non-capped) homes.
- **Decision:** log-transform was not adopted for the final model, since it did not improve the model's error profile in this case.
## Key Learnings
 
- Outlier treatment should happen **before** correlation analysis and scaling — both are sensitive to extreme values and can lead to incorrect conclusions if run on unclean data.
- High pairwise correlation can be an artifact of outliers rather than genuine multicollinearity; always re-check correlations after cleaning before dropping a feature.
- Regularization solves overfitting (high variance), not underfitting (high bias) — check train vs. test error before applying it.
- A technique that is theoretically appropriate (like log-transforming a skewed target) does not always improve results in practice, especially when a dataset has structural artifacts like a capped target value.
- Accuracy is a classification metric and does not apply to regression; R², RMSE, MAE, and MAPE are the appropriate tools for evaluating a regression model, each with different sensitivities.
## Next Steps
 
- Explore polynomial features and non-linear models (Random Forest, Gradient Boosting) to address the underfitting identified via the bias/variance diagnosis.
- Consider feature engineering (e.g. `population_per_household`, `bedrooms_per_room`) to derive more predictive ratios.
- Investigate cross-validation for more robust performance estimates and for hyperparameter tuning (e.g. `RidgeCV`, `LassoCV`) once a more flexible model is introduced.
## Tech Stack
 
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
## Status
 
This is an ongoing beginner ML project. The current model is a baseline linear regression with a documented, evidence-based diagnostic process. Non-linear modeling is planned as the next phase.
