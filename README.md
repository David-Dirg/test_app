# Customer Churn Predictive Analysis 

## Project Overview
This project demonstrates an end-to-end data pipeline to predict customer attrition (churn). By analyzing customer tenure, billing information, and support interactions, the model identifies high-risk customers, allowing business stakeholders to deploy targeted retention strategies.

## Key Skills Demonstrated
* **Data Wrangling:** Handling simulated business datasets using `pandas` and `numpy`.
* **Predictive Modeling:** Building and evaluating a Logistic Regression classifier using `scikit-learn`.
* **Business Intelligence Insight:** Translating model coefficients into actionable business metrics (Feature Importance).

## How to Run This Project
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the analysis pipeline:
   ```bash
   python churn_analysis.py
   ```
## Results & Business Impact
The model highlights that Support Tickets and Monthly Charges are the strongest predictors of a customer leaving. Recommendations include offering proactive discounts to customers with high recent ticket volumes.
