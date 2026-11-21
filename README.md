# Purchase Prediction: Logistic Regression

A binary classifier that predicts whether a customer will make a purchase, based on age, salary, ad clicks and time spent on site (Python, scikit-learn).

## Overview
Given a customer's profile and browsing behaviour, the model predicts `Purchased = 1` (bought) or `0` (did not buy). It serves as a baseline classification model, evaluated with accuracy, a confusion matrix and precision/recall.

## Dataset
- **File:** `data/purchase_data.csv` (500 rows, no missing values)
- **Source:** Practice dataset used for learning purposes

| Column | Description |
|---|---|
| `Age` | Customer age |
| `Salary` | Customer salary |
| `Ad_Clicks` | Number of ad clicks |
| `Time_On_Site` | Time spent on the site (minutes) |
| `Purchased` | Target: 1 = purchased, 0 = did not purchase |

## Approach
1. Load the data and check shape, types, missing values and target classes
2. Separate features (`Age`, `Salary`, `Ad_Clicks`, `Time_On_Site`) from the target (`Purchased`)
3. Split 80/20 (`random_state=42`): 400 training rows, 100 test rows
4. Train `LogisticRegression(max_iter=1000)`
5. Predict on the test set and compare actual vs predicted
6. Evaluate with accuracy, confusion matrix and classification report
7. Visualise the confusion matrix

## Results
**Accuracy: 76%** on 100 test rows.

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 32 | 9 |
| **Actual: Yes** | 15 | 44 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not purchased (0) | 0.68 | 0.78 | 0.73 | 41 |
| Purchased (1) | 0.83 | 0.75 | 0.79 | 59 |

The model is more reliable when it predicts a purchase (83% precision) than when it flags a non-purchase (68%). It misses 15 of the 59 actual buyers.

## Possible Improvements
- Scale the features (Salary is on a much larger scale than the others)
- Use cross-validation instead of a single split
- Tune the classification threshold if missing buyers is costly
- Compare against other models (decision tree, random forest)

## Getting Started
```bash
git clone https://github.com/kshitijnareda/purchase-prediction-logistic-regression.git
cd purchase-prediction-logistic-regression
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook
```
Open `logistic_regression_purchase_data.ipynb` and run all cells.

## Repository Structure
```
├── data/
│   └── purchase_data.csv
├── logistic_regression_purchase_data.ipynb
└── README.md
```

## Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter

## Author
**Kshitij Nareda**
GitHub: [@kshitijnareda](https://github.com/kshitijnareda)
