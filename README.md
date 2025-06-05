# Credit Risk Prediction Model 

Credit risk is the financial loss a lender faces when a borrower fails to meet their debt obligations. To manage this, lenders rely on predictive models to assess the likelihood of default. This project tackles that challenge by developing a robust machine learning model that predicts a borrower's probability of serious delinquency.

The model transforms raw borrower data into two key outputs: a precise **probability of default** and an intuitive, consumer-style **credit score (300-850)**, where a lower score signifies higher risk. It demonstrates a systematic approach to building and optimizing a high-performing classification model using the "Give Me Some Credit" Kaggle dataset (150,000 records). The focus is on the rigorous development of the predictive model, from data analysis to final evaluation, making it a strong showcase of applied machine learning skills for quantitative and technical roles.

## Table of Contents
- [Key Achievements](#-key-achievements)
- [Methodology Overview](#-methodology-overview)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Cleaning & Preprocessing](#2-data-cleaning--preprocessing)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Development & Optimization](#4-model-development--optimization)
- [Final Model Performance](#-final-model-performance)
- [Credit Score Conversion](#-credit-score-conversion)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage Example](#-usage-example)

## Key Achievements

*   **High Predictive Accuracy**: Achieved a **Test ROC AUC of 0.8662**, demonstrating strong discrimination between defaulting and non-defaulting borrowers.
*   **Effective Imbalance Handling**: Successfully managed a severe class imbalance (6.7% default rate) using XGBoost's `scale_pos_weight` parameter, resulting in **78% recall** for the crucial default class. This shows a practical approach to real-world skewed datasets.
*   **Rigorous Modeling Process**: Systematic EDA, robust data cleaning, impactful feature engineering (e.g., `TotalPastDue`), and hyperparameter tuning (GridSearchCV).
*   **Quantifiable Impact**: The model can identify nearly 4 out of 5 actual defaulters, a significant improvement for risk mitigation strategies.
*   **Reproducible & Documented**: The entire workflow is documented across Jupyter notebooks, ensuring clarity and reproducibility. Model artifacts (trained model and scaler) are saved for use.
*   **Practical Application Insight**: Includes conversion of default probabilities to industry-standard credit scores (300-850 range), showcasing an understanding of real-world deployment considerations.

## Methodology Overview

The project follows a structured machine learning pipeline:

### 1. Exploratory Data Analysis
*   **Dataset**: [Give Me Some Credit](https://www.kaggle.com/c/give-me-some-credit/data) Kaggle competition (150,000 records, 11 features).
*   **Key Findings**:
    *   **Class Imbalance**: Target variable `SeriousDlqin2yrs` highly imbalanced (6.7% defaults). *(See distribution below)*
    *   **Missing Data**: Significant missingness in `MonthlyIncome` (19.8%) and `NumberOfDependents` (2.6%).
    *   **Outliers & Anomalies**: Identified in `age`, `DebtRatio`, and `RevolvingUtilizationOfUnsecuredLines`.
*   *Detailed analysis in `notebooks/01_Data_Exploration.ipynb`.*

<p align="center">
  <img src="images/target_distribution.png" alt="Target Variable Distribution">
</p>

### 2. Data Cleaning & Preprocessing
*   **Outlier Treatment**: Capped extreme values for `DebtRatio` (99th percentile) and `RevolvingUtilizationOfUnsecuredLines` (at 10). Filtered invalid `age` entries.
*   **Missing Value Imputation**:
    *   `NumberOfDependents`: Imputed with 0.
    *   `MonthlyIncome`: Imputed with 0, and a binary flag `MonthlyIncomeMissing` was created to capture the information content of missingness.
*   **Data Type Correction**: Ensured appropriate data types for all features.
*   *Implementation details in `notebooks/02_Data_Cleaning.ipynb`.*

### 3. Feature Engineering
*   **Key Engineered Features**:
    *   `TotalPastDue`: Sum of `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, and `NumberOfTimes90DaysLate`. This proved to be the most impactful feature.
    *   `AgeGroup`: Categorical feature derived from `age` (Young, Adult, Senior).
*   **Encoding**: One-hot encoded `AgeGroup` (dropping the first category).
*   **Scaling**: Applied `StandardScaler` to all numerical features.
*   *Details in `notebooks/03_Feature_Engineering.ipynb`. The scaler is saved as `models/scaler.pkl`.*

### 4. Model Development & Optimization
*   **Algorithm Selection**: Evaluated Logistic Regression, Decision Tree, Random Forest, and XGBoost. XGBoost demonstrated superior baseline performance.
*   **Imbalance Handling Strategy**:
    *   Initial exploration included SMOTE.
    *   The final, best-performing model utilizes XGBoost's `scale_pos_weight` parameter (calculated as ratio of negative to positive class instances in the training set) *without* SMOTE. This provided a better balance of performance and generalization.
*   **Hyperparameter Tuning**: Employed `GridSearchCV` with 5-fold stratified cross-validation, optimizing for ROC AUC.
    *   **Final XGBoost Parameters**:
        ```python
        {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.05, 
         'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 200, 
         'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8}
        ```
    *   **Best CV ROC AUC Score**: 0.8636 (well-aligned with test performance).
*   *Model training and evaluation are in `notebooks/04_Model_Training.ipynb`. The trained model is saved as `models/credit_model.pkl`.*

## Final Model Performance

The optimized XGBoost model (using `scale_pos_weight` only) achieved the following on the hold-out test set:

| Metric                      | Value    | Notes                                                                 |
| :-------------------------- | :------- | :-------------------------------------------------------------------- |
| **Test ROC-AUC**            | **0.8662** | Strong overall model discrimination.                                  |
| **Recall (Default Class)**  | **0.78**   | Correctly identifies 78% of actual defaulters.                        |
| **Precision (Default Class)** | 0.22     | Of those predicted to default, 22% actually do.                     |
| **F1-Score (Default Class)**  | 0.35     | Harmonic mean, reflecting the balance for the critical default class. |
| **Overall Accuracy**        | 0.80     | Percentage of all correct predictions.                                |

*Note: Precision, Recall, and F1-Score are reported for the positive class (`SeriousDlqin2yrs = 1`), which is paramount for credit risk assessment.*

### Visualizing Model Performance

<p align="center">
  <img src="images/roc_curve_final.png" alt="Final Model ROC Curve">
</p>


<p align="center">
  <img src="images/confusion_matrix_final.png" alt="Final Model Confusion Matrix">
</p>

### Top 10 Feature Importances (Final Model)

![Final Model Feature Importance](images/feature_importance_final.png)

1.  **`TotalPastDue`** (Importance: ~0.670)
2.  **`RevolvingUtilizationOfUnsecuredLines`** (Importance: ~0.085)
3.  **`NumberOfTimes90DaysLate`** (Importance: ~0.043)
4.  **`NumberOfTime30-59DaysPastDueNotWorse`** (Importance: ~0.036)
5.  **`NumberRealEstateLoansOrLines`** (Importance: ~0.033)
6.  **`age`** (Importance: ~0.022)
7.  **`NumberOfOpenCreditLinesAndLoans`** (Importance: ~0.019)
8.  **`MonthlyIncomeMissing`** (Importance: ~0.018)
9.  **`NumberOfTime60-89DaysPastDueNotWorse`** (Importance: ~0.018)
10. **`DebtRatio`** (Importance: ~0.017)

## Credit Score Conversion

The model's raw output is a probability of default. To make this more interpretable for business use, it is converted into a consumer-style credit score (300-850) using a standard logistic scaling formula.

The score is calculated as follows:

$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

**Where:**

**Odds**: $\frac{1 - P(\text{Default})}{P(\text{Default})}$

**Factor**: $\frac{\text{PDO}}{\ln(2)}$ (where PDO = 50, Points to Double the Odds)

**Offset**: $\text{Base Score} - \text{Factor} \times \ln(\text{Base Odds})$ (Base Score = 600, Base Odds = 50)

This transformation ensures that a higher probability of default logarithmically maps to a lower credit score, which is the industry standard.

## Project Structure

```
credit-risk-prediction-model/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   └── 05_Model_Deployment.ipynb
├── models/
│   ├── credit_model.pkl
│   └── scaler.pkl
├── src/
│   └── predict.py              # Core script for preprocessing and prediction logic.
├── images/
│   ├── roc_curve_final.png
│   ├── confusion_matrix_final.png
│   ├── feature_importance_final.png
│   └── target_distribution.png
├── requirements.txt
├── score.py                    # Command-line interface to score a new applicant.
├── README.md
└── LICENSE
```

## Getting Started

### Prerequisites
*   Python 3.8+
*   Git
*   Sufficient RAM (e.g., 8GB+) for model training with the full dataset.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mustafaelzowawi/credit-risk-prediction-model.git
    cd credit-risk-prediction-model
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage Example

The primary way to use the trained model is via the command-line interface `score.py`. This script loads the trained model and scaler, preprocesses the input data, and returns a prediction for the probability of default and a corresponding credit score.

### Command-Line Execution
You can score a new applicant by providing their information as command-line arguments.

**Example command:**
```bash
python3 score.py \
    --revolving-utilization 0.5 \
    --age 45 \
    --past-due-30-59 1 \
    --debt-ratio 0.3 \
    --monthly-income 5000 \
    --open-credit-lines 5 \
    --past-due-90 0 \
    --real-estate-loans 1 \
    --past-due-60-89 0 \
    --dependents 2
```

**Expected Output:**
```
--- Credit Score Prediction Result ---
  Probability of Default: 53.01%
  Predicted Credit Score: 309
--------------------------------------
```

You can see all available options by running:
```bash
python score.py --help
```
