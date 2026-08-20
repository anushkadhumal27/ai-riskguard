# AI RiskGuard — Model Bias & Risk Auditor

**Track 2: AI Risk Manager**

## Problem
Organizations deploying ML models often lack a fast, automated way to catch
bias, class imbalance, missing-data issues, and drift *before* a model goes
into production — leading to unfair or unreliable outcomes downstream.

## Solution
AI RiskGuard is a Streamlit app that takes a tabular dataset (with an
optional target/label column and sensitive feature columns like gender,
age group, or region) and produces an automated risk report:

- **Missing value risk** — flags columns with significant missingness
- **Class imbalance risk** — flags skewed target distributions
- **Sensitive feature bias** — compares outcome rates across groups
  (e.g. approval rate by gender) to flag potential unfairness
- **Data drift** *(optional)* — compares a new dataset against a reference/
  training set to flag statistical shifts
- **Overall Risk Score (0–100)** — a single aggregated score with a
  Low/Medium/High verdict, plus supporting charts

## Tech Stack
Python, pandas, numpy, scikit-learn-style logic, Streamlit, matplotlib

## Project Structure
```
ai-riskguard/
├── app.py              # Streamlit UI
├── risk_auditor.py      # Core auditing logic (RiskAuditor class)
├── data/
│   └── sample_loan_data.csv   # Sample dataset with a deliberate gender bias, for demo
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then upload `data/sample_loan_data.csv` in the app, select `loan_approved`
as the target column and `gender`/`region` as sensitive features, and click
**Run Risk Audit**.

## Example Output
On the included sample dataset, AI RiskGuard correctly flags a **High risk**
gender bias (a ~30-point gap in loan approval rate between groups) and
scores the dataset **45/100 — High overall risk**.

## Future Improvements
- Support for direct model file uploads (e.g. `.pkl`) instead of just data
- More fairness metrics (equal opportunity, disparate impact ratio)
- PDF export of the risk report
