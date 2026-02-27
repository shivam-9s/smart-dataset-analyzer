import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

def evaluate_model(problem_type, y_test, predictions):

    st.subheader("📊 Model Evaluation")

    if problem_type == "Classification":
        acc = accuracy_score(y_test, predictions)
        st.write("Accuracy:", acc)

        st.text("Classification Report:")
        st.text(classification_report(y_test, predictions))

        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        return acc

    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        st.write("MSE:", mse)
        st.write("MAE:", mae)
        st.write("R2 Score:", r2)

        return r2
