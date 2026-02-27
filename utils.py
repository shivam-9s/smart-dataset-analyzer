import numpy as np
import pandas as pd


def detect_problem_type(y):
    if len(np.unique(y)) < 20:
        return "Classification"
    else:
        return "Regression"


def analyze_dataset(X, y, problem_type):

    analysis_report = ""

    if problem_type == "Regression":

        correlations = X.corrwith(y)
        avg_corr = np.mean(np.abs(correlations))

        analysis_report += f"📊 Average absolute correlation with target: {round(avg_corr,3)}\n\n"

        if avg_corr > 0.5:
            recommended_model = "Linear Regression"
            analysis_report += """
            🔎 Observation:
            - Strong linear relationship detected between features and target.
            - Linear Regression assumptions appear reasonable.
            - Dataset likely follows linear trend.

            ✅ Recommendation:
            Linear Regression is suitable because the data shows strong linear correlation.
            """
        else:
            recommended_model = "Random Forest"
            analysis_report += """
            🔎 Observation:
            - Weak linear correlation detected.
            - Data may contain non-linear patterns.
            - Linear model may underfit.

            ✅ Recommendation:
            Random Forest Regressor is preferred because it can capture non-linear relationships.
            """

    else:  # Classification

        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()

        analysis_report += f"📊 Class imbalance ratio: {round(imbalance_ratio,2)}\n\n"

        if imbalance_ratio < 1.5:
            recommended_model = "Logistic Regression"
            analysis_report += """
            🔎 Observation:
            - Classes are fairly balanced.
            - Likely linear separation possible.
            - Dataset is small/medium sized.

            ✅ Recommendation:
            Logistic Regression is appropriate due to balanced classes and possible linear boundary.
            """
        else:
            recommended_model = "Random Forest"
            analysis_report += """
            🔎 Observation:
            - Class imbalance detected.
            - Data may have complex decision boundaries.

            ✅ Recommendation:
            Random Forest is preferred because it handles imbalance and non-linearity better.
            """

    return recommended_model, analysis_report
