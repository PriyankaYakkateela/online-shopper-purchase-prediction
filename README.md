# Online Shopper Purchase Intention Prediction Using Machine Learning

## 1. Project Overview

This research project explores how machine learning techniques can predict whether an online shopping session will result in a purchase.

The study uses the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

The main goal is to compare different supervised machine learning algorithms, with particular attention to performance on the minority purchase class and to how much the models depend on the PageValues feature.

## 2. Research Question

How effectively can supervised machine learning models predict whether an online shopping session results in a purchase, and how does excluding PageValues affect predictive performance?

*Supporting questions:*

1. Which of the four models provides the most useful balance of precision and recall for the purchase class?
2. How much does model performance change when PageValues is excluded?

## 3. Dataset

*Dataset:* Online Shoppers Purchasing Intention Dataset

*Source:* UCI Machine Learning Repository (Sakar & Kastro, 2018)

*Dataset link:* https://doi.org/10.24432/C5F88Q

Dataset characteristics:

* 12,330 online shopping sessions
* 18 attributes
* 17 predictor variables
* 1 target variable: Revenue
* Binary classification problem
* Class imbalance: 84.5% no purchase, 15.5% purchase
* No missing values; 125 exact duplicate rows (retained in the main analysis)

The target variable indicates whether a shopping session resulted in a purchase.

The dataset is *not stored in this repository*. It is downloaded directly from UCI in the notebook using ucimlrepo. See data/README.md.

## 4. Machine Learning Models

The study will compare:

* Logistic Regression (baseline)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

## 5. Methodology

* 80/20 stratified train/test split (random_state=42)
* Stratified 5-fold cross-validation on the training set only
* Preprocessing (scaling, one-hot encoding) inside a scikit-learn Pipeline to prevent data leakage
* *PageValues ablation:* every model is trained with and without PageValues on the same split
* Predictions are made on completed sessions and are not real-time forecasts

## 6. Evaluation Metrics

Because the dataset contains fewer purchase sessions than non-purchase sessions, accuracy alone may not provide a complete evaluation.

The study will use:

* Precision-Recall AUC (primary metric)
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

## 7. Project Structure

text
online-shopper-purchase-prediction/
│
├── README.md                       # Project overview
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes .venv, checkpoints, data files
├── data/
│   └── README.md                   # Dataset source and download instructions
└── notebooks/
    └── Online_Shopper_EDA.ipynb    # Exploratory data analysis


Model training code (src/) will be added in the modelling phase.

## 8. How to Run

bash
git clone https://github.com/PriyankaYakkateela/online-shopper-purchase-prediction.git
cd online-shopper-purchase-prediction
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebooks/Online_Shopper_EDA.ipynb


## 9. Tools and Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* ucimlrepo
* Jupyter Notebook
* GitHub

## 10. Research Contribution

This research will compare multiple supervised machine learning models for predicting online shopping purchase intention. It will examine the challenges of imbalanced data, evaluate the trade-offs between identifying potential purchasers and generating false purchase alerts, and measure how much predictive performance depends on PageValues.

## 11. Reproducibility

The repository includes the dataset source, Python environment requirements and the exploratory data analysis notebook. A fixed random seed and identical train/test split are used for all experiments. Model development steps and evaluation results will be added as the experiments are completed, and all reported results come from executed code.

## 12. Author

Priyanka Yakkateela