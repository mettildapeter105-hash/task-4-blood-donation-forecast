"""
Blood Donation Forecast (scikit-learn)

Predicts whether a donor will donate blood again, using the RFMT features of the
Blood Transfusion Service Center dataset (Hsin-Chu City, Taiwan; 748 donors).

Features
    recency    months since the last donation
    frequency  total number of donations
    monetary   total blood donated in c.c.
    time       months since the first donation
Target
    1 if the donor donated blood in March 2007, otherwise 0

Steps: load -> inspect -> explore -> split -> compare models with cross-validation
-> evaluate the best one on a held-out test set -> save plots and the model.

Run:
    python blood_donation_forecast.py
    python blood_donation_forecast.py --data datasets/transfusion.data --out-dir outputs
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

COLUMNS = ["recency", "frequency", "monetary", "time", "target"]
SEED = 42


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = COLUMNS
    return df


def inspect(df):
    print("Shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nSummary statistics:")
    print(df.describe().round(2).to_string())
    print("\nTarget incidence (share of each class):")
    print(df["target"].value_counts(normalize=True).round(3).to_string())
    print("\nVariance of each feature:")
    print(df.drop(columns="target").var().round(2).to_string())


def drop_redundant(df):
    """Monetary is exactly 250 x frequency in this dataset, so it adds no information."""
    if (df["monetary"] == 250 * df["frequency"]).all():
        print("\nmonetary == 250 x frequency for every donor -> dropping 'monetary' (redundant).")
        return df.drop(columns="monetary")
    return df


def plot_eda(df, path):
    features = [c for c in df.columns if c != "target"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    counts = df["target"].value_counts().sort_index()
    axes[0, 0].bar(["Did not donate (0)", "Donated (1)"], counts.values, color=["#9aa5b1", "#d1495b"])
    axes[0, 0].set_title("Target balance")
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v, str(v), ha="center", va="bottom")

    for ax, col in zip(axes.ravel()[1:4], features):
        for label, colour in ((0, "#9aa5b1"), (1, "#d1495b")):
            ax.hist(df.loc[df["target"] == label, col], bins=20, alpha=0.65, color=colour, label=f"target={label}")
        ax.set_title(f"Distribution of {col}")
        ax.legend()

    corr = df.corr()
    im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(corr)))
    axes[1, 1].set_yticks(range(len(corr)))
    axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha="right")
    axes[1, 1].set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[1, 1].set_title("Correlation matrix")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models():
    """Candidate models. The log transform tames the skewed count features, and
    class_weight='balanced' stops the models ignoring the minority (donor) class."""
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    return {
        "Logistic Regression (raw)": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Logistic Regression (log)": Pipeline([
            ("log", log),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=SEED),
    }


def compare_models(models, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    print("\nModel comparison (5-fold cross-validated ROC AUC on the training set):")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = scores
        print(f"  {name:<28} mean AUC = {scores.mean():.4f}  (+/- {scores.std():.4f})")
    return results


def plot_model_comparison(results, path):
    names = list(results)
    means = [results[n].mean() for n in names]
    stds = [results[n].std() for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(names, means, xerr=stds, color="#3d6fb6", capsize=4)
    ax.set_xlim(0.5, max(means) + 0.08)
    ax.set_xlabel("Cross-validated ROC AUC (higher is better)")
    ax.set_title("Model comparison")
    ax.invert_yaxis()
    for i, m in enumerate(means):
        ax.text(0.51, i, f"AUC {m:.3f}", va="center", color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Blood donation forecast")
    parser.add_argument("--data", default=os.path.join(here, "datasets", "transfusion.data"))
    parser.add_argument("--out-dir", default=os.path.join(here, "outputs"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Fall back to a transfusion.data file placed next to the script
    if not os.path.exists(args.data) and os.path.exists(os.path.join(here, "transfusion.data")):
        args.data = os.path.join(here, "transfusion.data")
    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\n"
                         "Put transfusion.data in a 'datasets' folder next to this script, "
                         "or pass its path with --data.")

    df = load_data(args.data)
    inspect(df)
    df = drop_redundant(df)
    plot_eda(df, os.path.join(args.out_dir, "eda_plots.png"))

    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y)
    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}  "
          f"(donation rate train={y_train.mean():.3f}, test={y_test.mean():.3f})")

    models = build_models()
    results = compare_models(models, X_train, y_train)
    plot_model_comparison(results, os.path.join(args.out_dir, "model_comparison.png"))

    best_name = max(results, key=lambda n: results[n].mean())
    best = models[best_name].fit(X_train, y_train)
    print(f"\nBest model by cross-validation: {best_name}")

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    baseline = 1 - y_test.mean()
    print(f"\nTest ROC AUC : {auc:.4f}")
    print(f"Test accuracy: {acc:.4f}   (always predicting 'no donation' scores {baseline:.4f})")
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["no donation", "donation"], digits=3))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["no donation", "donation"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    joblib.dump(best, os.path.join(args.out_dir, "best_model.joblib"))

    # Example forecast for a new donor
    example = pd.DataFrame([{"recency": 2, "frequency": 10, "time": 30}])
    print("\nExample forecast - donor with recency=2, frequency=10, time=30 months:")
    print(f"  probability of donating = {best.predict_proba(example)[0, 1]:.2%}")
    print(f"\nSaved plots and model to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
