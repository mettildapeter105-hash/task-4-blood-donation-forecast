# Task 4 - Blood Donation Forecast

Machine learning project that predicts whether a blood donor will donate again. It uses data from the Blood Transfusion Service Center in Hsin-Chu City, Taiwan (748 donors), a common example of the forecasting problem blood collection managers face.

## Dataset (`transfusion.data`)
| Column | Meaning |
|---|---|
| Recency | months since the last donation |
| Frequency | total number of donations |
| Monetary | total blood donated (c.c.) |
| Time | months since the first donation |
| Target | 1 if the donor donated in March 2007, otherwise 0 |

Only about 24% of donors have target = 1, so the classes are imbalanced.

## Approach
1. Load and inspect the data (no missing values).
2. Found that `Monetary` is always exactly 250 x `Frequency`, so it was dropped as redundant.
3. Explored the features and target balance (`eda_plots.png`).
4. Split the data into 75% train / 25% test with a stratified split.
5. Compared four models with 5-fold cross-validated ROC AUC: logistic regression (raw and log-transformed features), random forest and gradient boosting. Class weights were balanced for the first three models to handle the imbalance.
6. Evaluated the best model on the held-out test set and saved plots.

## Setup and run
```bash
pip install -r requirements.txt
python blood_donation_forecast.py
```
Place `transfusion.data` next to the script (or in a `datasets/` folder). Outputs are saved to `outputs/`.

## Results
| Model | Cross-validated ROC AUC |
|---|---|
| Logistic Regression (raw) | 0.7372 |
| Logistic Regression (log) | 0.7351 |
| Random Forest | 0.7151 |
| Gradient Boosting | 0.7022 |

Best model: **Logistic Regression**. On the test set (187 donors):

- **ROC AUC: 0.785**
- Accuracy: 70.6%
- Donors (class 1): precision 0.435, recall 0.841, F1 0.574
- Non-donors (class 0): precision 0.931, recall 0.664, F1 0.776

Because only 24% of donors donate again, a model that always predicts "no donation" would score 76.5% accuracy but find no donors at all. The balanced model trades some accuracy to correctly identify 84% of the donors who do return, which is more useful for planning blood collection. ROC AUC is the main metric here.

Example forecast: a donor with recency = 2 months, frequency = 10 donations and time = 30 months has about a 75% predicted chance of donating.

## Result images
- `eda_plots.png` - target balance, feature distributions and correlations
- `model_comparison.png` - cross-validated AUC of each model
- `roc_curve.png` - ROC curve on the test set
- `confusion_matrix.png` - test-set confusion matrix

## Project structure
```
blood_donation_forecast.py
requirements.txt
transfusion.data
eda_plots.png
model_comparison.png
roc_curve.png
confusion_matrix.png
README.md
```
