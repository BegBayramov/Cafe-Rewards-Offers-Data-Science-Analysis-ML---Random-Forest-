☕ Cafe Rewards Offers — Data Science Analysis

📌 Project Overview
This project analyzes customer behavior and marketing offers in a cafe rewards program.
The goal is to understand which customers complete offers and build a machine learning model to predict offer completion. 

The dataset contains information about:
- marketing offers
- customer demographics
- customer events (transactions, offer received, viewed, completed)
The project follows a full Data Science pipeline, from data cleaning to modeling and evaluation.

📂 Dataset Description
The data consists of three main tables:
1️⃣ Offers
Information about marketing offers sent to customers.  

| Column     | Description                                   |
| ---------- | --------------------------------------------- |
| offer_id   | Unique offer ID                               |
| offer_type | Type of offer (bogo, discount, informational) |
| difficulty | Minimum amount required to complete the offer |
| reward     | Reward in dollars                             |
| duration   | Days available to complete the offer          |
| channels   | Marketing channels used                       |

2️⃣ Customers
Demographic information about customers. 
| Column           | Description               |
| ---------------- | ------------------------- |
| customer_id      | Unique customer ID        |
| became_member_on | Date when customer joined |
| gender           | Customer gender           |
| age              | Customer age              |
| income           | Estimated annual income   |


3️⃣ Events
Customer activity during the 30-day period.
| Column      | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| customer_id | Customer identifier                                           |
| event       | transaction / offer received / offer viewed / offer completed |
| value       | JSON-like field with event details                            |
| time        | Hours since start of the period                               |

🧹 Data Cleaning & Preparation
🔹 Parsing JSON-like data

The value column contains dictionaries stored as objects.
They were efficiently transformed into columns using: 

pd.json_normalize(events['value']) 

This approach is faster and safer than .apply(pd.Series). 

🔹 Handling offer identifiers

The dataset contained both offer id and offer_id fields.
- offer_id was used as the standard identifier
- redundant columns were removed 

🔹 Target variable creation

A binary target variable offer_completed was created:
- 1 — if a customer completed a received offer
- 0 — otherwise
Logic:
- Select all offer received events
- Check if the same customer later has offer completed for the same offer

📊 Exploratory Data Analysis (EDA)

The analysis includes:
- distribution of events
- customer age and income analysis
- correlation between demographics and offer completion
- offer completion rates
Example:
- events['event'].value_counts()


  🤖 Machine Learning Model
🔹 Model Choice

A Random Forest Classifier was used because:
- it handles non-linear relationships well
- it works with mixed feature types
- it is interpretable and robust


🔹 Features Used

- customer age
- customer income
- offer difficulty
- offer reward
- offer duration

🔹 Train/Test Split
train_test_split(test_size=0.2, random_state=42) 

🔹 Model Training
RandomForestClassifier(
    n_estimators=100,
    random_state=42
) 

📈 Model Evaluation
Classification Report
precision    recall  f1-score   support

0.0       0.80      0.77      0.78
1.0       0.80      0.84      0.82

accuracy                        0.80 

Interpretation:
- Accuracy: 80% — good overall performance
- Precision — how many predicted completions were correct
- Recall — how many real completions were detected
- Balanced performance between classes


 ✅ Key Results

- Customer income and age influence offer completion
- Certain offer types perform better than others
- The model successfully predicts offer completion with 80% accuracy

 🚀 Tools & Technologies
- Python
- pandas, numpy
- matplotlib
- scikit-learn
- Random Forest Classifier

  🎯 Conclusion
- This project demonstrates:
- strong data cleaning skills
- understanding of customer behavior
- ability to build and evaluate ML models
- practical application of Data Science to marketing analytics

 



 












