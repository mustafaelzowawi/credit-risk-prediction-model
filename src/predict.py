import joblib
import pandas as pd
import numpy as np
import os


# Load the trained model and scaler
model = joblib.load("models/credit_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Preprocesses raw input data for prediction by applying the same transformations
    used in the training pipeline.

    Args:
        input_data (dict): A dictionary containing the raw feature values.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for the model.
    """
    # The expected order of columns in the input dictionary for DataFrame creation
    feature_names = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
    ]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # 1. Feature Engineering
    input_df['TotalPastDue'] = (
        input_df['NumberOfTime30-59DaysPastDueNotWorse'] +
        input_df['NumberOfTime60-89DaysPastDueNotWorse'] +
        input_df['NumberOfTimes90DaysLate']
    )
    input_df['MonthlyIncomeMissing'] = input_df['MonthlyIncome'].isna().astype(int)
    input_df['MonthlyIncome'] = input_df['MonthlyIncome'].fillna(0)

    bins = [0, 30, 50, 120]
    labels = ['Young', 'Adult', 'Senior']
    input_df['AgeGroup'] = pd.cut(input_df['age'], bins=bins, labels=labels, right=False)
    input_df = pd.get_dummies(input_df, columns=['AgeGroup'], drop_first=True)

    # 2. Align Columns
    final_feature_names = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
        'MonthlyIncomeMissing', 'TotalPastDue', 'AgeGroup_Adult', 'AgeGroup_Senior'
    ]
    input_df = input_df.reindex(columns=final_feature_names, fill_value=0)

    # 3. Scale Numerical Features
    numerical_features = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome',
        'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents', 'TotalPastDue'
    ]
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    return input_df

def probability_to_score(probability: float, base_point: int = 600, pdo: int = 50) -> int:
    """
    Converts a probability of default into a credit score using logistic scaling.

    Args:
        probability (float): The model's predicted probability of default.
        base_point (int): The base score, a reference point.
        pdo (int): Points to Double the Odds to default.

    Returns:
        int: The calculated credit score, typically between 300 and 850.
    """
    # Constants for scaling
    base_odds = 50 

    # Ensure probability is within a valid range to avoid math errors
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    
    odds = (1 - probability) / probability
    
    factor = pdo / np.log(2)
    offset = base_point - factor * np.log(base_odds)
    
    score = offset + factor * np.log(odds)
    
    # Cap and floor the score to a standard 300-850 range
    return int(np.clip(score, 300, 850))


def calculate_credit_score(input_data: dict) -> dict:
    """
    A wrapper function that takes raw data, preprocesses it, predicts the
    probability of default, and converts it to a credit score.

    Args:
        input_data (dict): A dictionary of raw feature values.

    Returns:
        dict: A dictionary containing the probability of default and the credit score.
    """
    preprocessed_df = preprocess_input(input_data)
    
    # Predict the probability of the positive class (default)
    probability = model.predict_proba(preprocessed_df)[:, 1][0]
    
    credit_score = probability_to_score(probability)
    
    result = {
        'Probability of Default': round(probability, 4),
        'Credit Score': credit_score
    }
    
    return result 