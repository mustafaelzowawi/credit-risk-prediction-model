import argparse
import json
from src.predict import calculate_credit_score

def main():
    """
    Command-line interface for the credit risk model.

    This script allows users to get a credit score prediction by providing
    applicant data either as command-line arguments or as a JSON string.
    """
    parser = argparse.ArgumentParser(
        description="Calculate credit score based on applicant data.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Arguments
    parser.add_argument("--revolving-utilization", type=float, required=True, help="Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits.")
    parser.add_argument("--age", type=int, required=True, help="Age of borrower in years.")
    parser.add_argument("--past-due-30-59", type=int, required=True, help="Number of times borrower has been 30-59 days past due but no worse in the last 2 years.")
    parser.add_argument("--debt-ratio", type=float, required=True, help="Monthly debt payments, alimony, living costs divided by monthly gross income.")
    parser.add_argument("--monthly-income", type=float, help="Monthly income. Can be left empty.")
    parser.add_argument("--open-credit-lines", type=int, required=True, help="Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards).")
    parser.add_argument("--past-due-90", type=int, required=True, help="Number of times borrower has been 90 days or more past due.")
    parser.add_argument("--real-estate-loans", type=int, required=True, help="Number of mortgage and real estate loans including home equity lines of credit.")
    parser.add_argument("--past-due-60-89", type=int, required=True, help="Number of times borrower has been 60-89 days past due but no worse in the last 2 years.")
    parser.add_argument("--dependents", type=int, help="Number of dependents in family excluding themselves (spouse, children etc.). Can be left empty.")

    args = parser.parse_args()

    # Create the input dictionary for the prediction function
    sample_input = {
        'RevolvingUtilizationOfUnsecuredLines': args.revolving_utilization,
        'age': args.age,
        'NumberOfTime30-59DaysPastDueNotWorse': args.past_due_30_59,
        'DebtRatio': args.debt_ratio,
        'MonthlyIncome': args.monthly_income,
        'NumberOfOpenCreditLinesAndLoans': args.open_credit_lines,
        'NumberOfTimes90DaysLate': args.past_due_90,
        'NumberRealEstateLoansOrLines': args.real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': args.past_due_60_89,
        'NumberOfDependents': args.dependents
    }

    try:
        # Get the prediction result
        result = calculate_credit_score(sample_input)

        # Print the results in a clean, readable format
        print("\n--- Credit Score Prediction Result ---")
        print(f"  Probability of Default: {result['Probability of Default']:.2%}")
        print(f"  Predicted Credit Score: {result['Credit Score']}")
        print("--------------------------------------\n")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        print("Please ensure all required arguments are provided and are in the correct format.")

if __name__ == "__main__":
    main() 