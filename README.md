🚢 Titanic Survival Prediction

This project uses machine learning to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, fare, and more.
The Titanic dataset is a classic beginner-friendly dataset for classification tasks.

📁 Project Structure
├── Titanic-Dataset.csv
├── titanic_model.py  (or .ipynb)
└── README.md

🎯 Objective

Build a prediction model that determines whether a Titanic passenger survived using supervised machine learning techniques.

📊 Dataset Information

The dataset contains the following important features:

Survived – 0 = No, 1 = Yes

Pclass – Passenger class (1, 2, 3)

Sex – Male/Female

Age – Age of passenger

SibSp – Number of siblings/spouses aboard

Parch – Number of parents/children aboard

Fare – Ticket price

Embarked – Port (C, Q, S)

Cabin – Cabin number (many values missing)

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-Learn

🧹 Data Preprocessing

Filled missing Age, Fare, Embarked values

Dropped Cabin due to too many missing values

Encoded categorical features (Sex, Embarked)

Selected relevant features for training

Train–test split (80/20)

🤖 Machine Learning Models
1️⃣ Logistic Regression

Baseline classification model

Accuracy typically ~80%

2️⃣ Random Forest Classifier

Better performance

Accuracy typically ~85%

Used as final model

📈 Model Evaluation

Accuracy Score

Classification Report

Confusion Matrix (visualized)

▶️ How to Run
Install dependencies:
pip install pandas numpy scikit-learn matplotlib

Run the Python script:
python titanic_model.py

🎉 Results Summary

Random Forest performed better than Logistic Regression

Important features affecting survival:

Sex

Pclass

Fare

Age

📝 Future Improvements

Add EDA visualizations

Hyperparameter tuning

Try XGBoost / SVM

Improve feature engineering

👩‍💻 Author

Shivangi Soni
Machine Learning & Data Analysis Enthusiast
