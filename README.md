
# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect fraudulent transactions from a dataset of credit card transactions. Using various machine learning techniques and deep learning models, the goal is to predict whether a transaction is fraudulent or not.

## Table of Contents
- [Overview]
- [Dataset]
- [Technologies Used]
- [Installation]
- [Usage]
- [Modeling Approach]
- [Results]
- [Contributing]
- [License]

## Overview
Credit card fraud is a major concern in the financial industry. This project leverages machine learning algorithms to detect anomalies in credit card transactions that could be indicative of fraudulent behavior.

The dataset consists of real-world anonymized credit card transactions. Each transaction includes 30 features (such as `V1`, `V2`, ..., `V28`, Time, and Amount) and a label indicating whether the transaction is fraudulent (`1`) or not (`0`).

## Dataset
The dataset used in this project contains:
- **128,821 rows** and **31 columns**.
- Each row represents a credit card transaction.
- The `Class` column indicates fraud (1) or non-fraud (0).

The dataset is highly imbalanced, with only **0.17%** fraudulent transactions.

## Technologies Used
The project is implemented using:
- **Python** (3.x)
- **Pandas** for data manipulation
- **Scikit-learn** for data preprocessing and model training
- **Imbalanced-learn (SMOTE)** for handling class imbalance
- **Keras/TensorFlow** for deep learning (LSTM model)
- **XGBoost** for gradient boosting classification
- **Matplotlib** and **Seaborn** for data visualization

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/CreditCardFraudDetection.git
   cd CreditCardFraudDetection
   ```


2. **Dataset:**
   Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

## Usage

1. **Data Preprocessing:**
   - Run the Jupyter notebook `CreditCardFraudDetection.ipynb` to perform data cleaning, normalization, and feature engineering.
   
2. **Model Training:**
   - Multiple models are trained in this project:
     - A basic Logistic Regression model.
     - An LSTM model to capture sequential patterns in the transactions.
     - XGBoost for gradient boosting classification.
   - The dataset is oversampled using SMOTE to handle class imbalance.

3. **Evaluation:**
   - Use the confusion matrix, ROC-AUC curve, and classification reports to evaluate the performance of each model.

4. **Run the Notebook:**
   To execute the project notebook, open and run the `CreditCardFraudDetection.ipynb` notebook using Google Colab or Jupyter Notebook.

## Modeling Approach

1. **Data Preprocessing:**
   - Normalization of `Time` and `Amount` columns.
   - Removal of outliers using Interquartile Range (IQR).
   - Handling missing values by interpolation and filling missing values with median values.

2. **Models Used:**
   - **LSTM (Long Short-Term Memory)**: Used to capture the sequential nature of transactions.
   - **XGBoost**: Used for classification with the help of gradient boosting.
   - **MLP (Multi-Layer Perceptron)**: An alternative approach using neural networks.

3. **Handling Imbalance:**
   - **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training dataset to handle the class imbalance problem.

## Results
- The **LSTM model** achieved an accuracy of `99.85%` on the test set.
- The **XGBoost model** achieved an accuracy of `99.94%` and an ROC-AUC score of `0.9858`.
- Performance metrics include:
  - **Accuracy**: The percentage of correct predictions.
  - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
  - **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
  - **ROC-AUC**: The area under the ROC curve.

