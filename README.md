# Online Shopper Purchase Intention Prediction Using Machine Learning

## 1. Project Overview

This research project explores how machine learning techniques can predict whether an online shopping session will result in a purchase.

The study uses the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

The main goal is to compare different supervised machine learning algorithms and identify which model provides the most useful predictions.

## 2. Research Question

How effectively can supervised machine learning models predict whether an online shopping session results in a purchase?

## 3. Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**Dataset link:** https://doi.org/10.24432/C5F88Q

Dataset characteristics:

* 12,330 online shopping sessions
* 18 attributes
* 17 predictor variables
* 1 target variable: Revenue
* Binary classification problem

The target variable indicates whether a shopping session resulted in a purchase.

## 4. Machine Learning Models

The study will compare:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

## 5. Evaluation Metrics

Because the dataset contains fewer purchase sessions than non-purchase sessions, accuracy alone may not provide a complete evaluation.

The study will use:

* Precision
* Recall
* F1-score
* Precision-Recall AUC (primary metric)
* ROC-AUC
* Confusion Matrix

## 6. Project Structure

```text
online-shopper-purchase-prediction/
│
├── README.md
├── research/
│   └── research_protocol.docx
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   └── model_training.py
├── data/
│   └── README.md
└── requirements.txt
```

## 7. Tools and Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* GitHub

## 8. Research Contribution

This research will compare multiple supervised machine learning models for predicting online shopping purchase intention. It will also examine the challenges of imbalanced data and evaluate the trade-offs between identifying potential purchasers and generating false purchase alerts.

## 9. Reproducibility

The project will include the dataset source, Python environment requirements, exploratory data analysis, model development steps, and evaluation results to support reproducibility.

## 10. Author

Priyanka Yakkateela
