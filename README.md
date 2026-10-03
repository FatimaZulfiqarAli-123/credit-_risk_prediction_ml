# Credit Risk Prediction — Machine Learning Competition

## 📌 Project Overview

This project develops a machine learning solution for predicting the probability that a loan applicant will experience payment difficulties.

The project follows a complete end-to-end machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Missing-value treatment
* Categorical feature encoding
* Feature engineering
* Feature scaling
* Multiple machine learning models
* Model evaluation using ROC-AUC
* Hyperparameter tuning
* Ensemble learning
* Final prediction generation
* Competition submission

The main objective is to build a robust classification model that can distinguish between applicants with different levels of credit risk.

---

## 🎯 Problem Statement

The task is a **binary classification problem**.

For each applicant, the model predicts the probability of the target class:

* `0` → No payment difficulties
* `1` → Payment difficulties

Because the competition focuses on probability predictions rather than only class labels, **ROC-AUC** is used as the primary evaluation metric.

---

## 📊 Dataset

The project uses three CSV files:

```text
input/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Training Data

The training dataset contains:

* Applicant information
* Financial information
* Employment/organization information
* Previous external-score information
* Vehicle ownership information
* Target variable: `TARGET`

### Test Data

The test dataset contains the same predictive features but does not include the target variable.

### Sample Submission

The sample submission file provides the required competition submission format.

---

## 🔎 Features Used

The project starts with important applicant-level features, including:

| Feature              | Description                              |
| -------------------- | ---------------------------------------- |
| `NAME_CONTRACT_TYPE` | Type of loan contract                    |
| `AMT_INCOME_TOTAL`   | Applicant's total income                 |
| `EXT_SOURCE_2`       | External credit-related score            |
| `OWN_CAR_AGE`        | Age of the applicant's car               |
| `ORGANIZATION_TYPE`  | Applicant's organization/employment type |

Additional engineered features are created to improve predictive performance.

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes several steps.

### 1. Categorical Encoding

`NAME_CONTRACT_TYPE` is converted into numerical values.

`ORGANIZATION_TYPE` is processed using frequency encoding and one-hot encoding.

### 2. Missing Values

Missing values are handled using appropriate statistical techniques such as:

* Mean imputation
* Median imputation
* Missing-value indicator features

### 3. Car Age Processing

Extreme values in `OWN_CAR_AGE` are treated as missing values.

The feature is then grouped into meaningful age ranges and one-hot encoded.

### 4. Boolean Conversion

Boolean features generated during preprocessing are converted into integer values so that they can be used consistently by the machine learning models.

### 5. Feature Alignment

Training and testing datasets are aligned to ensure that both contain the same feature columns.

Duplicate columns are also removed.

---

# 🧠 Feature Engineering

Feature engineering is an important part of this project.

The following additional features are created:

### Income Transformation

```text
LOG_INCOME
INCOME_SQUARED
```

These transformations help capture non-linear relationships between income and credit risk.

### External Score Transformation

```text
EXT_SOURCE_2_SQUARED
```

This allows the model to capture potential non-linear effects of the external score.

### Interaction Features

Several interaction features are created, including:

```text
INCOME × EXT_SOURCE_2
INCOME / EXT_SOURCE_2
OWN_CAR_AGE × EXT_SOURCE_2
CONTRACT_TYPE × EXT_SOURCE_2
CONTRACT_TYPE × INCOME
```

These features allow the models to capture relationships between applicant characteristics.

### Missing Indicators

Additional binary features are created to indicate whether important variables were originally missing.

---

# 📈 Exploratory Data Analysis

The project includes visual analysis of important variables.

Examples include:

* Target-class distribution
* Loan contract type distribution
* Organization type distribution
* External score distribution
* Income distribution
* Car-age distribution

Visualization tools include:

* Matplotlib
* Seaborn

EDA is used to understand the dataset and identify patterns that may be useful during feature engineering.

---

# 🤖 Machine Learning Models

Multiple machine learning algorithms are evaluated.

## 1. Logistic Regression

Logistic Regression is used as a baseline linear classification model.

Before training, numerical features are standardized using `StandardScaler`.

---

## 2. Random Forest

Random Forest is used to capture non-linear relationships and interactions between features.

The model uses:

* Multiple decision trees
* Class balancing
* Parallel processing

---

## 3. AdaBoost

AdaBoost is tested as a boosting-based classification algorithm.

It combines multiple weak learners to create a stronger predictive model.

---

## 4. XGBoost

XGBoost is used as a gradient boosting model capable of learning complex non-linear patterns.

---

## 5. CatBoost

CatBoost is evaluated as another gradient boosting approach with strong performance on structured/tabular datasets.

---

## 6. LightGBM

LightGBM is included because of its computational efficiency and strong performance on large tabular datasets.

Feature names are sanitized before training to ensure compatibility with LightGBM.

---

# 📏 Model Evaluation

The primary evaluation metric is:

## ROC-AUC

ROC-AUC measures how effectively the model ranks positive cases above negative cases.

A higher ROC-AUC indicates better ranking performance.

The project evaluates models using a stratified train-validation split.

Example:

```text
Training Set → 70%
Validation Set → 30%
```

Stratification is used to maintain the target-class distribution across both datasets.

---

# 🔧 Hyperparameter Tuning

Random Forest hyperparameters are optimized using `GridSearchCV`.

Because the dataset contains a large number of observations, a compact parameter search is used to reduce computational cost.

The current search focuses on:

```text
n_estimators
max_depth
```

Cross-validation is used to compare different parameter combinations using ROC-AUC.

---

# 🤝 Ensemble Learning

The project also experiments with ensemble predictions.

Predictions from different models can be combined to create a final probability estimate.

For example:

```text
Final Prediction =
    Logistic Regression Prediction
    +
    Random Forest Prediction
    +
    Gradient Boosting Predictions
```

The ensemble approach aims to combine models that learn different patterns from the data.

---

# 🏗️ Project Workflow

The complete workflow can be summarized as:

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning
     │
     ▼
Missing Value Treatment
     │
     ▼
Categorical Encoding
     │
     ▼
Feature Engineering
     │
     ▼
Train / Validation Split
     │
     ├───────────────┐
     ▼               ▼
Scaling           Tree Models
     │               │
     ▼               ├── Random Forest
Logistic Regression  ├── AdaBoost
                     ├── XGBoost
                     ├── CatBoost
                     └── LightGBM
                             │
                             ▼
                       Model Evaluation
                             │
                             ▼
                      Hyperparameter Tuning
                             │
                             ▼
                         Ensembles
                             │
                             ▼
                    Final Model Training
                             │
                             ▼
                     Test Predictions
                             │
                             ▼
                  Competition Submission
```

---

# 📁 Project Structure

Recommended project structure:

```text
Competition/
│
├── input/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── output/
│   └── submission.csv
│
├── tutorial.ipynb
│
├── tutorial.zip
│
├── README.md
│
└── requirements.txt
```

---

# 💻 Technologies Used

The project is implemented in Python.

### Programming Language

* Python 3.x

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* CatBoost
* LightGBM

### Development Environment

* Visual Studio Code
* Jupyter Notebook

---

# ▶️ How to Run

### Step 1 — Prepare the Dataset

Place the competition CSV files inside:

```text
input/
```

Required files:

```text
train.csv
test.csv
sample_submission.csv
```

### Step 2 — Open the Notebook

Open:

```text
tutorial.ipynb
```

using Visual Studio Code with the Jupyter extension installed.

### Step 3 — Select the Python Environment

Select the Python environment containing all required packages.

### Step 4 — Run the Notebook

Run the cells sequentially from:

```text
CELL 01
```

through the final cell.

Running cells in order is important because later cells depend on variables created during earlier preprocessing and feature-engineering steps.

### Step 5 — Generate Submission

The final prediction file is saved as:

```text
output/submission.csv
```

---

# 📤 Submission Format

The final submission contains the required identifier and predicted probability.

Example:

```text
SK_ID_CURR,TARGET
100001,0.084532
100005,0.173921
100013,0.062841
```

The `TARGET` column contains probability predictions rather than hard class labels.

---

# 🔬 Reproducibility

Random seeds are fixed where applicable to improve reproducibility.

For example:

```python
random_state=0
```

The same preprocessing and feature-engineering steps are applied consistently to the training and test datasets.


---


#  Conclusion

This project implements a complete machine learning pipeline for credit-risk prediction, beginning with raw applicant data and progressing through preprocessing, feature engineering, model development, evaluation, tuning, ensemble learning, and final submission generation.

The project also demonstrates how multiple machine learning algorithms can be systematically compared and combined to develop a robust solution for a real-world tabular classification problem.
