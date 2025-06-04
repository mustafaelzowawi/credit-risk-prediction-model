# Credit Risk Prediction Model

A comprehensive machine learning pipeline for predicting credit default risk and generating consumer-style credit scores. This project demonstrates end-to-end ML development, from data exploration to model deployment, using the "Give Me Some Credit" dataset.

## 🎯 Project Overview

This project builds a production-ready credit scoring system that:
- Processes raw borrower data through a complete ML pipeline
- Handles class imbalance and missing data with advanced techniques
- Delivers state-of-the-art prediction performance (0.87 ROC-AUC)
- Converts probabilities to interpretable 300-850 credit scores
- Provides reproducible, deployment-ready model artifacts

## 📊 Dataset

**Source**: "Give Me Some Credit" Kaggle Competition
- **Size**: 150,000 rows
- **Features**: 11 borrower characteristics
- **Target**: Default probability (binary classification)
- **Challenges**: 
  - Heavy class imbalance (~6% default rate)
  - 20% missing values in MonthlyIncome
  - Extreme outliers requiring careful handling

## 🛠️ Technical Implementation

### Data Processing Pipeline
- **Outlier Treatment**: Statistical capping of extreme values
- **Missing Value Imputation**: Median imputation with missing indicators
- **Feature Engineering**: Created business-relevant predictors
- **Class Balancing**: SMOTE for handling imbalanced data
- **Feature Scaling**: StandardScaler for numeric features

### Key Engineered Features
- `TotalPastDue`: Sum of all delinquency categories
- `DebtRatio`: Monthly debt payments to income ratio
- `AgeGroup`: Binned age categories with one-hot encoding
- `MonthlyIncomeMissing`: Missing value indicator flag

### Model Development
- **Algorithm Selection**: Benchmarked multiple algorithms
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Final Model**: XGBoost Classifier
- **Performance**: 0.87 ROC-AUC on hold-out test set
- **Validation**: Stratified train-test split

### Credit Score Conversion
Transforms model probabilities into consumer-friendly scores using logistic scaling:
- **Range**: 300-850 (industry standard)
- **Base Score**: 600
- **Points to Double Odds (PDO)**: 50
- **Base Odds**: 50:1

## 📁 Project Structure

```
├── notebooks/
│   ├── 01_Data_Exploration.ipynb      # EDA and data understanding
│   ├── 02_Data_Cleaning.ipynb         # Data preprocessing
│   ├── 03_Feature_Engineering.ipynb   # Feature creation and selection
│   ├── 04_Model_Training.ipynb        # Model development and tuning
│   └── 05_Model_Deployment.ipynb      # Production pipeline setup
├── ml_model/
│   ├── predict.py                     # Prediction functions
│   ├── preprocess.py                  # Data preprocessing pipeline
│   └── train.py                       # Model training pipeline
├── models/
│   ├── credit_model.pkl               # Trained XGBoost model
│   └── scaler.pkl                     # Fitted StandardScaler
├── data/                              # Dataset directory
├── tests/                             # Unit tests
└── requirements.txt                   # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Data Exploration
```bash
jupyter notebook notebooks/01_Data_Exploration.ipynb
```

#### 2. Run Complete Pipeline
```python
from ml_model.predict import predict_credit_score

# Example borrower data
borrower_data = {
    'RevolvingUtilizationOfUnsecuredLines': 0.3,
    'age': 35,
    'NumberOfTime30-59DaysPastDueNotWorse': 0,
    'DebtRatio': 0.4,
    'MonthlyIncome': 5000,
    'NumberOfOpenCreditLinesAndLoans': 8,
    'NumberOfTimes90DaysLate': 0,
    'NumberRealEstateLoansOrLines': 1,
    'NumberOfTime60-89DaysPastDueNotWorse': 0,
    'NumberOfDependents': 2
}

credit_score = predict_credit_score(borrower_data)
print(f"Credit Score: {credit_score}")
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.87 |
| Precision | 0.82 |
| Recall | 0.79 |
| F1-Score | 0.80 |

### Feature Importance (Top 5)
1. RevolvingUtilizationOfUnsecuredLines
2. age
3. NumberOfTime30-59DaysPastDueNotWorse
4. DebtRatio
5. NumberOfTimes90DaysLate

## 🔧 Technical Details

### Libraries Used
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib

### Key Techniques
- **SMOTE**: Synthetic Minority Oversampling Technique
- **GridSearchCV**: Automated hyperparameter optimization
- **Stratified Sampling**: Maintains class distribution in splits
- **Feature Scaling**: StandardScaler for numeric features
- **Cross-Validation**: 5-fold stratified CV for robust evaluation

## 🎯 Production Considerations

- **Reproducibility**: All random states fixed, complete pipeline saved
- **Scalability**: Efficient preprocessing for batch predictions
- **Monitoring**: Feature drift detection capabilities
- **Interpretability**: Feature importance and SHAP value support
- **API-Ready**: Modular design for microservice deployment

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📋 Requirements

See `requirements.txt` for complete dependency list. Key packages:
- pandas>=1.3.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- imbalanced-learn>=0.8.0
- numpy>=1.21.0

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📧 Contact

[Your Name] - [your.email@example.com]

Project Link: [https://github.com/yourusername/credit-risk-prediction](https://github.com/yourusername/credit-risk-prediction)

---

*This project demonstrates advanced machine learning techniques for credit risk assessment, showcasing end-to-end ML pipeline development from data exploration to production deployment.*
