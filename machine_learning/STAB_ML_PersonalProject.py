"""
Predicting Program Application Outcomes
Author: Ryan Moh
Description:
    Machine learning pipeline predicting application acceptance based on student demographic and
    program attributes. Includes: 
        • Data cleaning and preprocessing
        • Encoding categorical and ordinal features
        • Feature scaling
        • Model training using RandomForest
        • Cross-validation and performance evaluation
"""

# 1. Import Required Libraries
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt



# 2. Load and Inspect Dataset ( I recently learned type hinting)
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load CSV data, clean column names, and drop irrelevant fields."""
    df = pd.read_csv(filepath)
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )

    # Drop non-essential or redundant columns
    remove_cols: List[str] = [
        "college",
        "college_of_primary_major",
        "u_of_a_primary_dept",
        "hometown",
        "home_state_(or_country_if_not_usa)",
        "who_in_your_immediate_family_has_completed_a_bachelor_degree?",
        "what_is_your_level?",
        "what_is_your_expected_date_of_graduation?",
        "what_is_the_primary_college_of_your_major(s)?",
        "please_list_your_primary_major:",
        "are_you_enrolled_in_the_honors_college?",
        "program_type",
        "program_group",
    ]
    df = df.drop(columns=remove_cols, axis=1)
    df.drop(columns="application_status_alias", inplace=True, errors="ignore")
    return df


# 3. Feature Cleaning and Encoding
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, normalize levels, and encode binary target."""
    # Fill missing categorical data
    fill_cols: List[str] = [
        "level",
        "major",
        "u_of_a_department",
        "u_of_a_primary_school",
        "ethnicity",
        "citizenship_country",
        "type_of_program",
        "gender",
        "who_in_your_immediate_family_has_completed_an_associate_or_bachelor_degree?",
    ]
    df[fill_cols] = df[fill_cols].fillna("unknown")

    # Keep only valid gender entries
    df = df[df["gender"].astype(str).str.lower().isin({"f", "m"})]

    # Encode target variable (1 = accepted/committed, 0 = otherwise)
    status_positive = {"accepted", "committed"}
    df["y"] = (
        df["application_status"].astype(str).str.lower().isin(status_positive).astype(int)
    )

    # Normalize 'level' column categories
    df["level"] = (
        df["level"]
        .str.lower()
        .str.strip()
        .replace(
            {
                r"^freshman$": "freshman",
                r"^sophomore$": "sophomore",
                r"^junior$": "junior",
                r".*senior.*": "senior",
                r"(.*master.*|.*professional.*|.*graduate.*)": "graduate",
                r"^doctorate.*": "doctorate",
                r"^unknown.*": "unknown",
            },
            regex=True,
        )
    )

    # Simplify first-gen indicator mapping
    df.rename(
        columns={
            "who_in_your_immediate_family_has_completed_an_associate_or_bachelor_degree?": "first_gen_ind"
        },
        inplace=True,
    )
    mapping = {
        "neither have completed a college degree": 1,
        "both parents": 0,
        "father": 0,
        "mother": 0,
    }
    df["first_gen"] = df["first_gen_ind"].astype(str).str.lower().str.strip().map(mapping)

    return df


# 4. Build Preprocessing Pipeline
def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    ordinal_features: List[str],
    ordinal_categories: List[List[str]],
) -> ColumnTransformer:
    """Define preprocessing pipelines."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", TargetEncoder()),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("ord", ordinal_transformer, ordinal_features),
        ]
    )
    return preprocessor


# 5. Train and Evaluate Model
def train_and_evaluate_model(df: pd.DataFrame) -> None:
    """Split data, train RandomForest pipeline, and evaluate performance."""
    X = df.drop(columns=["y", "application_status"])
    y = df["y"]

    # Define feature groups
    numeric_features = ["program_year"]
    ordinal_features = ["level"]
    ordinal_categories = [["freshman", "sophomore", "junior", "senior", "graduate", "doctorate"]]
    categorical_features = [
        c for c in df.columns if df[c].dtype == "object" and c not in ordinal_features + ["application_status"]
    ]

    # Build preprocessing and modeling pipeline
    preprocessor = create_preprocessor(
        numeric_features, categorical_features, ordinal_features, ordinal_categories
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Split dataset with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Evaluate performance
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"\nCV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Train Accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test  Accuracy: {model.score(X_test, y_test):.3f}")

    # Confusion matrix visualization
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.show()

    # K-Fold validation for robustness
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    print("\nCross-Validation Results (Accuracy):")
    for i, result in enumerate(scores, 1):
        print(f"  Fold {i}: {result * 100:.2f}%")
    print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")


# 6. Execute
if __name__ == "__main__":
    filepath: str = r""
    df = load_and_clean_data(filepath)
    df = preprocess_data(df)
    train_and_evaluate_model(df)

# 7. My learnings and model insights

#Model is overfitting because training data score is 97.8% while test is 80%

#Bias and Variance 
#   -Bias refers to when the model is too rigid to capture complex relationships,leading to underfitting. 
#   -Variance happens when the model fits noise in training data too closely,leading to overfitting 

#Detecting Overfitting
#  -Compare training and validation/test performance,  a large performance gap indicates overfitting.
#  -Cross-validation ensures every sample has a chance to be tested, reducing the risk of memorization.
#  -Stratified sampling helps preserve class balance in imbalanced datasets.

#Cross-Validation Techniques
#   -Hold-Out Validation: Split dataset into train/test subsets (simple but may be data-dependent).
#   -K-Fold CV: Rotate validation sets for more reliable performance estimates.
#   -Stratified K-Fold: Maintains target class proportions, useful for imbalanced labels.
#   -Leave-One-Out (LOOCV): Each observation acts as its own test set, only good for small dataset.



# 8. Next steps

#Mitigating Overfitting 
#   -Regularization (L1/L2): Penalize large coefficients to control model complexity.
#   -Early Stopping: Halt training when validation error stops improving (deep learning context).
#   -Feature Selection: Remove redundant or noisy variables using RFE or feature importance.
#   -Cross-Validation: Improves generalization by exposing the model to diverse data splits.

#Hyperparameter Tuning
#   -Use GridSearchCV or RandomizedSearchCV to optimize parameters like depth, estimators, and learning rate.
#   -Evaluate performance using ROC-AUC, F1, or precision-recall for imbalanced datasets.
#   -Interpret model performance using feature importance or SHAP/LIME to explain key predictors.

#Model Deployment Concepts
#   -Once validated, pipelines can be exported using joblib or pickle for deployment.
#coming soon

